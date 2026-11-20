"""Export a merged fp16 model to 4-bit ONNX for Transformers.js (WebGPU or WASM).

Output follows the onnx-community repo layout Transformers.js loads from:

    <output>/config.json (with "transformers.js_config"), generation_config.json, tokenizer files
    <output>/onnx/model_q4f16.onnx (+ model_q4f16.onnx_data)   --dtype q4f16 (default): WebGPU
    <output>/onnx/model_q4.onnx (+ model_q4.onnx_data)         --dtype q4: WASM fallback

q4f16 has fp16 activations and KV cache (needs WebGPU with shader-f16); q4
has the same 4-bit weights with fp32 activations, which the WASM (CPU)
backend runs. Both can sit in one output directory.

The graph is built by onnxruntime-genai's model builder, the tool behind the
onnx-community repos: fused contrib ops (MultiHeadAttention, RotaryEmbedding,
SkipSimplifiedLayerNormalization), ~2.6k nodes for Qwen2.5-Coder-3B, and with
`shared_embeddings` the tied embedding and lm_head share one 4-bit
GatherBlockQuantized weight (1.74GB). A plain torch/optimum export followed by
MatMulNBits quantization also runs in the browser, but its graph is unfused
(~9.1k nodes) and decoded ~6x slower (~7 vs ~45 tokens/s on an RTX 3060, see
SQLocal.md), so that path was dropped.

The builder reads the fp16 merged model; an fp32 export of the 3B model would
need ~12GB for the weights alone, more free RAM than the machine this runs on
reliably has.

`--verify N` runs the zero-shot dev evaluation through onnxruntime (CUDA) on
the first N dev examples, optionally comparing SQL against an evaluate.py
results file.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from text2sql.utils.config import load_config, resolve_path
from text2sql.utils.logging import get_logger

logger = get_logger(__name__)

_ROOT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)

# Same as onnx-community/Qwen2.5-Coder-3B-Instruct: tells Transformers.js the
# weights live in an external file and the KV cache is fp16.
_TRANSFORMERS_JS_CONFIG = {
    "kv_cache_dtype": {"q4f16": "float16", "fp16": "float16"},
    "dtype": "q4f16",
    "use_external_data_format": True,
}


def _save_onnx(model, path: Path) -> None:
    """Save with all weights in one external `<name>_data` file, as Transformers.js expects."""
    import onnx

    data_name = path.name + "_data"
    (path.parent / data_name).unlink(missing_ok=True)
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
    )


def embed_chat_template(output_dir: Path) -> None:
    """Copy chat_template.jinja into tokenizer_config.json.

    transformers>=4.43 saves the chat template as a separate
    chat_template.jinja, but Transformers.js reads it only from
    tokenizer_config.json["chat_template"]; without it,
    apply_chat_template() fails in the browser.
    """
    jinja = output_dir / "chat_template.jinja"
    cfg_path = output_dir / "tokenizer_config.json"
    if not jinja.exists():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["chat_template"] = jinja.read_text(encoding="utf-8")
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Embedded chat_template.jinja into %s", cfg_path)


def write_root_files(merged_dir: Path, output_dir: Path) -> None:
    """Config + tokenizer files from the merged model, adjusted for Transformers.js."""
    for name in _ROOT_FILES:
        if (merged_dir / name).exists():
            shutil.copy2(merged_dir / name, output_dir / name)
    embed_chat_template(output_dir)
    cfg_path = output_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["transformers.js_config"] = _TRANSFORMERS_JS_CONFIG
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# Runs the onnxruntime-genai builder CLI with one fix: transformers' Qwen2
# config always has a `rope_scaling` attribute (None when unused), and the
# 0.20 builder iterates over it whenever the attribute exists, crashing with
# "'NoneType' object is not iterable". Treat None as "no scaling" ({}).
#
# Second fix: for WebGPU the builder picks GroupQueryAttention, which expects
# onnxruntime-genai's pre-allocated, shared past/present KV buffer and fails
# under Transformers.js' growing KV cache ("Input 'past_key' dimension 3
# should match the packed KV head dimension, got 0"). Reporting GQA and packed
# Attention as unsupported makes it fall back to MultiHeadAttention + a
# separate RotaryEmbedding + a `position_ids` input -- the same structure as
# the onnx-community/Qwen2.5-Coder-3B-Instruct q4f16 export that runs in
# Transformers.js.
#
# builder.py imports its submodules as top-level packages (`from builders
# import ...`, with its own directory on sys.path), so the class to patch is
# `builders.base.Model`, not `onnxruntime_genai.models.builders.base.Model`.
_BUILDER_SHIM = """
import os, runpy, sys
import onnxruntime_genai.models as _models
sys.path.insert(0, os.path.dirname(_models.__file__))
import builders.base as base
_orig = base.Model.make_config_init
def _patched(self, config):
    if getattr(config, "rope_scaling", {}) is None:
        config.rope_scaling = {}
    return _orig(self, config)
base.Model.make_config_init = _patched
base.Model.is_gqa_supported = lambda self: False
base.Model.is_packed_attn_supported = lambda self: False
sys.argv = ["builder"] + sys.argv[1:]
runpy.run_module("onnxruntime_genai.models.builder", run_name="__main__")
"""


def standard_ops_embedding(model):
    """Replace GatherBlockQuantized (the 4-bit embedding lookup) with standard ONNX ops.

    onnxruntime-web's WASM build has no GatherBlockQuantized kernel ("Could
    not find an implementation for GatherBlockQuantized(1)"), so the q4 model
    would not load on the CPU fallback. The same lookup, with the same 4-bit
    weights (shared with lm_head) and the same numbers:

        bytes  = Gather(packed [vocab, hidden/2], ids)          two 4-bit values per byte
        values = interleave(bytes % 16, bytes // 16) - 8        low nibble first, zero point 8
        embed  = values.reshape(..., hidden/block, block) * Gather(scales, ids)[..., None]
    """
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    graph = model.graph
    nodes = [n for n in graph.node if n.op_type == "GatherBlockQuantized"]
    if not nodes:
        return model
    (node,) = nodes
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    assert attrs.get("bits") == 4 and attrs.get("gather_axis", 0) == 0 and attrs.get("quantize_axis") == 1, attrs
    assert len(node.input) == 3, "zero points are not handled"
    packed, ids, scales = node.input
    block = attrs["block_size"]
    n_blocks = next(list(i.dims) for i in graph.initializer if i.name == scales)[1]
    hidden = n_blocks * block
    p = "/model/embed_tokens/std/"

    def const(name, array):
        graph.initializer.append(numpy_helper.from_array(np.asarray(array), p + name))
        return p + name

    sixteen, eight = const("sixteen", np.int32(16)), const("eight", np.float32(8))
    last = const("last_axis", np.array([-1], dtype=np.int64))
    shape_blocks = const("shape_blocks", np.array([0, 0, n_blocks, block], dtype=np.int64))
    shape_hidden = const("shape_hidden", np.array([0, 0, hidden], dtype=np.int64))
    new = [
        helper.make_node("Gather", [packed, ids], [p + "bytes"], axis=0),
        helper.make_node("Cast", [p + "bytes"], [p + "ints"], to=TensorProto.INT32),
        helper.make_node("Mod", [p + "ints", sixteen], [p + "low"]),
        helper.make_node("Div", [p + "ints", sixteen], [p + "high"]),
        helper.make_node("Unsqueeze", [p + "low", last], [p + "low1"]),
        helper.make_node("Unsqueeze", [p + "high", last], [p + "high1"]),
        helper.make_node("Concat", [p + "low1", p + "high1"], [p + "pairs"], axis=-1),
        helper.make_node("Cast", [p + "pairs"], [p + "floats"], to=TensorProto.FLOAT),
        helper.make_node("Sub", [p + "floats", eight], [p + "centered"]),
        helper.make_node("Reshape", [p + "centered", shape_blocks], [p + "blocks"]),
        helper.make_node("Gather", [scales, ids], [p + "scales"], axis=0),
        helper.make_node("Unsqueeze", [p + "scales", last], [p + "scales1"]),
        helper.make_node("Mul", [p + "blocks", p + "scales1"], [p + "scaled"]),
        helper.make_node("Reshape", [p + "scaled", shape_hidden], list(node.output)),
    ]
    index = list(graph.node).index(node)
    graph.node.remove(node)
    for offset, n in enumerate(new):
        graph.node.insert(index + offset, n)
    return model


# Transformers.js file name and builder execution provider for each dtype.
_DTYPES = {"q4f16": ("model_q4f16.onnx", "webgpu"), "q4": ("model_q4.onnx", "cpu")}


def export_genai(merged_dir: Path, output_dir: Path, block_size: int, shared_embeddings: bool, dtype: str = "q4f16") -> Path:
    import onnx

    with tempfile.TemporaryDirectory(dir=output_dir.parent) as tmp:
        tmp = Path(tmp)
        extra = [f"block_size={block_size}", "op_types_to_quantize=MatMul/Gather"]
        if shared_embeddings:
            extra.append("shared_embeddings=true")
        cmd = [
            sys.executable, "-c", _BUILDER_SHIM,
            "-i", str(merged_dir), "-o", str(tmp / "out"), "-p", "int4", "-e", _DTYPES[dtype][1],
            "-c", str(tmp / "cache"), "--extra_options", *extra,
        ]
        logger.info("Running onnxruntime-genai model builder: %s", " ".join(cmd[3:]))
        subprocess.run(cmd, check=True)
        (output_dir / "onnx").mkdir(parents=True, exist_ok=True)
        target = output_dir / "onnx" / _DTYPES[dtype][0]
        # Re-save rather than rename: the graph stores its external data file
        # name (the builder writes model.onnx.data; Transformers.js wants
        # <file>.onnx_data).
        model = onnx.load(str(tmp / "out" / "model.onnx"))
        if dtype == "q4":
            model = standard_ops_embedding(model)  # the WASM backend lacks GatherBlockQuantized
        _save_onnx(model, target)
        shutil.copy2(tmp / "out" / "genai_config.json", output_dir / f"genai_config_{dtype}.json")
    write_root_files(merged_dir, output_dir)
    logger.info("Wrote %s", target)
    return target


def verify(output_dir: Path, file_name: str, n: int, reference: Path | None, data_config: dict) -> dict:
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoConfig, AutoTokenizer

    from text2sql.eval.baseline_eval import run_baseline

    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = ORTModelForCausalLM.from_pretrained(
        str(output_dir),
        subfolder="onnx",
        file_name=file_name,
        config=AutoConfig.from_pretrained(str(output_dir)),
        provider="CUDAExecutionProvider",
        use_io_binding=True,
    )
    with open(resolve_path(data_config["dev_file"]), encoding="utf-8") as f:
        dev = json.load(f)[:n]
    out = run_baseline(
        "zero_shot", dev, {}, model, tokenizer, resolve_path(data_config["raw_dir"]), 0, data_config["max_seq_length"]
    )
    summary = {"file": file_name, "n": len(dev), **out["metrics"]}
    if reference:
        with open(reference, encoding="utf-8") as f:
            ref = json.load(f)["results"][:n]
        assert [r["question"] for r in ref] == [r["question"] for r in out["results"]]
        summary["same_sql_as_reference"] = sum(a["pred"] == b["pred"] for a, b in zip(ref, out["results"])) / len(dev)
        summary["reference_execution_accuracy"] = sum(bool(r["execution_accuracy"]) for r in ref) / len(dev)
    logger.info("Verification (%s): %s", file_name, summary)
    with open(output_dir / f"verify_dev_{file_name.removesuffix('.onnx')}.json", "w", encoding="utf-8") as f:
        json.dump({**summary, "results": out["results"]}, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merged", required=True, help="merged fp16 model directory (scripts/merge_adapter.py)")
    parser.add_argument("--output", required=True, help="output directory, onnx-community layout")
    parser.add_argument("--dtype", choices=list(_DTYPES), default="q4f16", help="q4f16 for WebGPU, q4 for the WASM fallback")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--no-shared-embeddings", action="store_true", help="keep separate embedding and lm_head weights")
    parser.add_argument("--skip-export", action="store_true", help="reuse the existing ONNX files, e.g. to only --verify")
    parser.add_argument("--verify", type=int, default=0, help="evaluate on the first N dev examples")
    parser.add_argument("--verify-files", nargs="+", default=None, help="default: the file for --dtype")
    parser.add_argument("--reference", default=None, help="evaluate.py results JSON to compare SQL against")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    args = parser.parse_args()

    merged_dir = resolve_path(args.merged)
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_export:
        export_genai(merged_dir, output_dir, args.block_size, not args.no_shared_embeddings, args.dtype)
    if args.verify:
        data_config = load_config(args.data_config)
        reference = resolve_path(args.reference) if args.reference else None
        for name in args.verify_files or [_DTYPES[args.dtype][0]]:
            verify(output_dir, name, args.verify, reference, data_config)


if __name__ == "__main__":
    main()
