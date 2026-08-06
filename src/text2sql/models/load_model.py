"""Load a model for inference or training.

Baseline evaluation needs the un-finetuned base model; loading a fine-tuned
checkpoint (post-tuning evaluation, demo) is a separate concern to add once
training actually produces one.
"""

from text2sql.utils.logging import get_logger

logger = get_logger(__name__)

# Pre-quantized checkpoint matching train_config.yaml's base_model -- avoids
# downloading the full-precision original just to quantize it locally.
#
# Was unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit -- switched to 3B after a
# real training run (not just vram_check.py's single-step measurement) on
# the 7B model pushed dedicated VRAM to ~11.3GB/12GB and, more specifically,
# BAR1 memory to ~16.35GB/16.38GB (nearly exhausted), which throttled
# Unsloth's CPU-offloaded gradient-checkpointing transfers and slowed
# training to ~5x the originally-measured rate (~154s/it vs ~30s/it), on top
# of ~36 minutes of accumulated thermal slowdown over a ~4h partial run. See
# reports/qlora_training.md for the full incident writeup.
QUANTIZED_MODEL_ID = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"


def load_base_model(max_seq_length: int):
    """Load the un-finetuned base model + tokenizer, ready for generation."""
    import unsloth  # noqa: F401 -- must be imported before transformers/peft
    from unsloth import FastLanguageModel

    logger.info("Loading base model %s (max_seq_length=%d)", QUANTIZED_MODEL_ID, max_seq_length)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=QUANTIZED_MODEL_ID,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_model_for_training(lora_cfg: dict, max_seq_length: int, load_in_4bit: bool = True):
    """Load the base model + tokenizer wrapped with a trainable LoRA adapter (QLoRA).

    Shared by the real training run (`text2sql.train.train_qlora`) and the
    VRAM measurement script (`text2sql.train.vram_check`) so the two don't
    drift -- the whole point of measuring VRAM ahead of time is to know what
    the actual training run will do.
    """
    import unsloth  # noqa: F401 -- must be imported before transformers/peft
    from unsloth import FastLanguageModel

    logger.info(
        "Loading trainable model %s (max_seq_length=%d, load_in_4bit=%s)",
        QUANTIZED_MODEL_ID,
        max_seq_length,
        load_in_4bit,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=QUANTIZED_MODEL_ID,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        use_gradient_checkpointing="unsloth",
        max_seq_length=max_seq_length,
        random_state=3407,
    )
    return model, tokenizer
