"""Merge a LoRA adapter into a full-precision base model for export.

Training ran QLoRA on Unsloth's pre-quantized 4-bit base, so every
checkpoint's `adapter_config.json` points at the bnb-4bit repo. Merging into
that 4-bit model would bake quantization error into the weights and give an
unexportable bitsandbytes graph, so the adapter is merged into the original
fp16 weights of the same model instead (`merge_and_unload`), and the result
is saved as a plain transformers checkpoint that ONNX export can read.

This module deliberately does not import Unsloth: the merged model must be a
stock transformers model, not one with Unsloth's patched forward passes.

`--verify N` re-runs the zero-shot dev evaluation on the first N dev
examples with the merged model, and, given the 4-bit + adapter results file
from scripts/evaluate.py, reports how often the two produce the same SQL.
"""

import argparse
import json
from pathlib import Path

from text2sql.utils.config import load_config, resolve_path
from text2sql.utils.logging import get_logger

logger = get_logger(__name__)

# Unsloth's pre-quantized repos carry the same weights as the originals, just
# stored in 4-bit; map them back to the full-precision source.
_FULL_PRECISION_BASES = {
    "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
}


def resolve_base_model(adapter_dir: Path) -> str:
    with open(adapter_dir / "adapter_config.json", encoding="utf-8") as f:
        base = json.load(f)["base_model_name_or_path"]
    if base in _FULL_PRECISION_BASES:
        return _FULL_PRECISION_BASES[base]
    if "bnb-4bit" in base:
        raise ValueError(f"No full-precision base known for {base}; pass --base explicitly")
    return base


def merge(adapter_dir: Path, output_dir: Path, base: str | None, device: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = base or resolve_base_model(adapter_dir)
    logger.info("Loading base %s in fp16 on %s", base, device)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map=device)

    logger.info("Applying adapter %s and merging", adapter_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()

    # The adapter checkpoint's tokenizer carries the chat template used in
    # training; keep it rather than the base repo's copy.
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved merged fp16 model to %s", output_dir)
    return model, tokenizer


def verify(model, tokenizer, n: int, reference: Path | None, data_config: dict, max_seq_length: int):
    from text2sql.eval.baseline_eval import run_baseline

    with open(resolve_path(data_config["dev_file"]), encoding="utf-8") as f:
        dev = json.load(f)[:n]
    model.eval()
    out = run_baseline(
        "zero_shot", dev, {}, model, tokenizer, resolve_path(data_config["raw_dir"]), 0, max_seq_length
    )
    summary = {"n": len(dev), **out["metrics"]}

    if reference:
        with open(reference, encoding="utf-8") as f:
            ref = json.load(f)["results"][:n]
        merged = out["results"]
        assert [r["question"] for r in ref] == [r["question"] for r in merged]
        summary["same_sql_as_reference"] = sum(a["pred"] == b["pred"] for a, b in zip(ref, merged)) / len(dev)
        summary["reference_execution_accuracy"] = sum(bool(r["execution_accuracy"]) for r in ref) / len(dev)
    logger.info("Verification on first %d dev examples: %s", n, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", required=True, help="LoRA checkpoint directory")
    parser.add_argument("--output", required=True, help="where to save the merged model")
    parser.add_argument("--base", default=None, help="full-precision base model (default: derived from adapter_config.json)")
    parser.add_argument("--device", default="cuda", help="device to merge on (fp16 3B needs ~6.5GB)")
    parser.add_argument("--verify", type=int, default=0, help="evaluate the merged model on the first N dev examples")
    parser.add_argument("--reference", default=None, help="evaluate.py results JSON for the same checkpoint, to compare SQL")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    args = parser.parse_args()

    model, tokenizer = merge(resolve_path(args.adapter), resolve_path(args.output), args.base, args.device)
    if args.verify:
        data_config = load_config(args.data_config)
        summary = verify(
            model,
            tokenizer,
            args.verify,
            resolve_path(args.reference) if args.reference else None,
            data_config,
            data_config["max_seq_length"],
        )
        with open(resolve_path(args.output) / "verify_dev.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
