"""Measure actual peak VRAM for one QLoRA training step, instead of guessing.

Supports two backends:
  - "peft": plain transformers + bitsandbytes + peft. Useful as a worst-case
    baseline, but Qwen's large (152K) vocabulary makes the loss computation
    materialize huge fp32 logits, which peft/transformers doesn't optimize
    away -- this backend was measured to peak at 16.5GB (batch=2) / 13.0GB
    (batch=1) at max_seq_length=1536, over the 12GB target GPU either way.
  - "unsloth": the backend this project actually plans to train with. Its
    fused kernels (notably chunked/fused cross-entropy) specifically target
    the large-vocab logits blowup above.

Builds a real batch from the longest examples in data/processed/train.json
(near-worst-case for the configured max_seq_length), runs one
forward+backward+optimizer step, and reports the measured peak. Run once per
--batch-size/--backend combination (as separate processes) for a clean
reading -- reusing one process across configs leaves stale allocator state
that skews the next measurement.
"""

import argparse
import json

from text2sql.models.load_model import QUANTIZED_MODEL_ID
from text2sql.utils.config import load_config, resolve_path
from text2sql.utils.logging import get_logger

logger = get_logger(__name__)


def _longest_examples(train_file, tokenizer, n: int) -> list[list[dict]]:
    with open(train_file, encoding="utf-8") as f:
        data = json.load(f)
    scored = [(len(tokenizer.apply_chat_template(ex["messages"], tokenize=True)), ex["messages"]) for ex in data]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    lengths = [length for length, _ in scored[:n]]
    logger.info("Selected %d longest examples, token lengths: %s", n, lengths)
    return [messages for _, messages in scored[:n]]


def _load_peft(base_model: str, lora_cfg: dict, max_seq_length: int):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(QUANTIZED_MODEL_ID, device_map={"": 0}, dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    return model, tokenizer


def _load_unsloth(base_model: str, lora_cfg: dict, max_seq_length: int):
    # Delegates to the same loader the real training run uses
    # (text2sql.train.train_qlora) -- the point of this measurement is to
    # know what that run will actually do, not a lookalike.
    from text2sql.models.load_model import load_model_for_training

    return load_model_for_training(lora_cfg, max_seq_length)


def run(batch_size: int, backend: str, data_config_path: str, train_config_path: str) -> dict:
    import torch

    data_config = load_config(data_config_path)
    train_config = load_config(train_config_path)
    max_seq_length = data_config["max_seq_length"]
    base_model = train_config["base_model"]
    lora_cfg = train_config["lora"]

    logger.info("Loading tokenizer + 4-bit model via %s backend (%s)", backend, QUANTIZED_MODEL_ID)
    loader = _load_unsloth if backend == "unsloth" else _load_peft
    model, tokenizer = loader(base_model, lora_cfg, max_seq_length)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable params: %d / %d (%.3f%%)", trainable, total, 100 * trainable / total)

    train_file = resolve_path(data_config["train_file"])
    conversations = _longest_examples(train_file, tokenizer, batch_size)

    encoded = tokenizer.apply_chat_template(
        conversations,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=train_config["training"]["learning_rate"]
    )

    torch.cuda.reset_peak_memory_stats()
    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    result = {
        "backend": backend,
        "batch_size": batch_size,
        "seq_len": input_ids.shape[1],
        "loss": outputs.loss.item(),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }
    logger.info(
        "backend=%s batch_size=%d seq_len=%d -> peak_allocated=%.2fGB peak_reserved=%.2fGB",
        result["backend"],
        result["batch_size"],
        result["seq_len"],
        result["peak_allocated_gb"],
        result["peak_reserved_gb"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure peak VRAM for one QLoRA training step")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--backend", choices=["peft", "unsloth"], default="unsloth")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    args = parser.parse_args()

    run(args.batch_size, args.backend, args.data_config, args.train_config)


if __name__ == "__main__":
    main()
