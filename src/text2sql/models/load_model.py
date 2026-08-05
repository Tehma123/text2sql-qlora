"""Load a model for inference.

Baseline evaluation needs the un-finetuned base model; loading a fine-tuned
checkpoint (post-tuning evaluation, demo) is a separate concern to add once
training actually produces one.
"""

from text2sql.utils.logging import get_logger

logger = get_logger(__name__)

# Pre-quantized checkpoint matching train_config.yaml's base_model -- avoids
# downloading the ~15GB full-precision original just to quantize it locally.
QUANTIZED_MODEL_ID = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"


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
