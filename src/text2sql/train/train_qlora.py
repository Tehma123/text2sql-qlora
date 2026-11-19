"""QLoRA fine-tuning of the base model on the Spider train split.

Trains only on the assistant's SQL response, not the schema/question prompt
(`unsloth.chat_templates.train_on_responses_only`): the prompt is several
times longer than the response (see reports/data_preparation.md), so an
unmasked loss would spend most of its gradient signal relearning schema
tokens instead of the actual question -> SQL mapping. Qwen2.5's own chat
template has no `{% generation %}` tags, so trl's built-in
`assistant_only_loss` silently produces an all-masked (all -100) label set
on this tokenizer -- checked directly, not assumed -- which is why this
uses Unsloth's marker-based masking instead.

Model loading (base model + LoRA adapter) is shared with
`text2sql.train.vram_check` via `text2sql.models.load_model.load_model_for_training`,
so the VRAM measured ahead of time reflects what this script actually runs.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from text2sql.utils.config import load_config, resolve_path
from text2sql.utils.logging import get_logger

logger = get_logger(__name__)

# Qwen2.5's chat template renders turns as ChatML -- verified directly against
# the real tokenizer (`tokenizer.apply_chat_template`), not assumed:
#   <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>\n
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def _load_dataset(path: Path, tokenizer):
    """Load a processed split and render its chat `messages` to a flat `text` column.

    Unsloth's compiled SFTTrainer (unsloth_compiled_cache/UnslothSFTTrainer.py)
    doesn't auto-detect trl's "conversational" (`messages`-column) dataset
    format the way vanilla trl does -- verified directly, not assumed: passing
    a raw `messages` dataset raised "You must specify a `formatting_func`",
    and that hook's own contract for a *single* test example doesn't line up
    with the batched call it makes for real tokenization. Applying the chat
    template once, ourselves, into a plain "text" column sidesteps both.
    """
    from datasets import Dataset

    with open(path, encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)
    dataset = Dataset.from_list(records)
    return dataset.map(
        lambda examples: {"text": [tokenizer.apply_chat_template(convo, tokenize=False) for convo in examples["messages"]]},
        batched=True,
        remove_columns=dataset.column_names,
    )


def _resolve_report_to(train_config: dict) -> str:
    """Use wandb only if it's both configured *and* actually usable.

    Falls back to "none" (with a warning, not a crash) rather than assuming
    the environment is set up -- WANDB_API_KEY isn't set and the wandb
    package isn't installed in this environment as of writing.
    """
    tracking = train_config.get("experiment_tracking", {})
    if tracking.get("provider") != "wandb":
        return "none"
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning(
            "experiment_tracking.provider is 'wandb' but WANDB_API_KEY is not set -- "
            "disabling experiment tracking (report_to='none'). Export WANDB_API_KEY to enable."
        )
        return "none"
    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning(
            "experiment_tracking.provider is 'wandb' but the wandb package isn't installed -- "
            "disabling experiment tracking. `pip install wandb` to enable."
        )
        return "none"
    os.environ.setdefault("WANDB_PROJECT", tracking.get("project", "text2sql-qlora"))
    return "wandb"


def run(
    train_config_path: str,
    data_config_path: str,
    resume_from_checkpoint: str | bool | None = None,
    max_steps: int | None = None,
    train_limit: int | None = None,
) -> dict[str, Any]:
    import unsloth  # noqa: F401 -- must be imported before trl/transformers/peft, or trl gets stale
    # unpatched classes: SFTTrainer.__init__ then round-trips a valid SFTConfig
    # through TrainingArguments.to_dict() (isinstance check fails against the
    # patched class), whose redaction of any "*_token" field name -- real
    # secret or not -- replaces eos_token=None with the literal string
    # "<EOS_TOKEN>", which then fails the vocabulary lookup. Reproduced and
    # confirmed by import order alone before landing on this fix.
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    from text2sql.models.load_model import load_model_for_training, quantized_model_id

    train_config = load_config(train_config_path)
    data_config = load_config(data_config_path)

    max_seq_length = data_config["max_seq_length"]
    lora_cfg = train_config["lora"]
    training_cfg = train_config["training"]
    load_in_4bit = train_config.get("quantization", {}).get("load_in_4bit", True)

    model, tokenizer = load_model_for_training(
        lora_cfg, max_seq_length, load_in_4bit=load_in_4bit, model_id=quantized_model_id(train_config["base_model"])
    )
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable params: %d / %d (%.3f%%)", trainable, total, 100 * trainable / total)

    train_file = resolve_path(data_config["train_file"])
    train_dataset = _load_dataset(train_file, tokenizer)
    if train_limit:
        train_dataset = train_dataset.select(range(min(train_limit, len(train_dataset))))
    logger.info("Loaded %d training examples from %s", len(train_dataset), train_file)

    dev_file = resolve_path(data_config["dev_file"])
    eval_dataset = _load_dataset(dev_file, tokenizer)
    logger.info("Loaded %d dev examples from %s", len(eval_dataset), dev_file)

    output_dir = resolve_path(train_config["output_dir"])
    report_to = _resolve_report_to(train_config)
    early_stopping_patience = training_cfg.get("early_stopping_patience")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        max_length=max_seq_length,
        packing=False,
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        num_train_epochs=training_cfg["num_train_epochs"],
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=training_cfg["warmup_ratio"],
        logging_steps=training_cfg["logging_steps"],
        save_strategy=training_cfg["save_strategy"],
        save_steps=training_cfg.get("save_steps", 500),
        eval_strategy=training_cfg.get("eval_strategy", "epoch"),
        eval_steps=training_cfg.get("eval_steps"),
        # EarlyStoppingCallback requires these; final/ then holds the lowest-eval_loss checkpoint
        load_best_model_at_end=early_stopping_patience is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_eval_batch_size=training_cfg["per_device_train_batch_size"],
        optim=training_cfg.get("optim", "adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        bf16=True,
        seed=3407,
        report_to=report_to,
        dataloader_num_workers=0,  # Windows: worker processes on a spawned dataloader are more trouble than they save here
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience)] if early_stopping_patience is not None else None,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    effective_batch_size = training_cfg["per_device_train_batch_size"] * training_cfg["gradient_accumulation_steps"]
    logger.info(
        "Starting training: %d examples, %s epoch(s), effective batch size %d, max_seq_length=%d",
        len(train_dataset),
        training_cfg["num_train_epochs"],
        effective_batch_size,
        max_seq_length,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved final LoRA adapter to %s", final_dir)

    metrics = dict(train_result.metrics)
    metrics["global_step"] = trainer.state.global_step
    metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    metrics["best_eval_loss"] = trainer.state.best_metric
    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote training metrics to %s: %s", metrics_path, metrics)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune the base model on the Spider train split")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        help="resume training; pass a checkpoint dir path, or omit the value to auto-resume from the latest checkpoint in output_dir",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (overrides num_train_epochs); for smoke tests")
    parser.add_argument("--limit", type=int, default=None, help="only train on the first N examples; for smoke tests")
    args = parser.parse_args()

    run(
        args.train_config,
        args.data_config,
        resume_from_checkpoint=args.resume_from_checkpoint,
        max_steps=args.max_steps,
        train_limit=args.limit,
    )


if __name__ == "__main__":
    main()
