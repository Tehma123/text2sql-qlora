"""Post-tuning evaluation: run fine-tuned checkpoints against a held-out split.

Reuses `baseline_eval.run_baseline` rather than duplicating its generation
loop. Two modes:
  - zero_shot (default): matches the fine-tuned model's own training format
    (`build_messages`, no few-shot demos -- see
    `text2sql.prompts.templates.build_training_messages`), so it's the
    correct comparison to `results/baselines/zero_shot.json`.
  - few_shot: combines fine-tuning with the same live per-database
    demonstrations the few-shot baseline gets, to test whether that closes
    the gap to `results/baselines/few_shot.json` -- an experiment, not
    something the model was trained to expect.

Checkpoint *selection* across epochs should run against `dev.json` (the
same split `eval_loss` was computed on during training) so the choice of
"best" checkpoint isn't biased by peeking at `test.json`. Once a checkpoint
is chosen, evaluate it once against `test.json` for the number that's
actually comparable to the baselines.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from text2sql.eval.baseline_eval import _sample_few_shot_pool, run_baseline
from text2sql.models.load_model import load_finetuned_model
from text2sql.utils.config import load_config, resolve_path
from text2sql.utils.logging import get_logger

logger = get_logger(__name__)


def _load_split(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_checkpoint(
    checkpoint_dir: Path,
    eval_data: list[dict[str, Any]],
    raw_dir: Path,
    max_seq_length: int,
    mode: str = "zero_shot",
    few_shot_pool: dict[str, list] | None = None,
    num_examples: int = 0,
) -> dict[str, Any]:
    model, tokenizer = load_finetuned_model(str(checkpoint_dir), max_seq_length)
    return run_baseline(
        mode,
        eval_data,
        few_shot_pool=few_shot_pool or {},
        model=model,
        tokenizer=tokenizer,
        raw_dir=raw_dir,
        num_examples=num_examples,
        max_seq_length=max_seq_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one or more fine-tuned checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="checkpoint directories to evaluate")
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="which split to evaluate against")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N examples")
    parser.add_argument("--output-dir", default="results/finetuned")
    args = parser.parse_args()

    eval_config = load_config(args.eval_config)
    data_config = load_config(args.data_config)

    split_file = resolve_path(data_config["dev_file"] if args.split == "dev" else eval_config["test_file"])
    eval_data = _load_split(split_file)
    if args.limit:
        eval_data = eval_data[: args.limit]
    logger.info("Loaded %d %s examples from %s", len(eval_data), args.split, split_file)

    raw_dir = resolve_path(data_config["raw_dir"])
    max_seq_length = data_config["max_seq_length"]

    num_examples = eval_config.get("baselines", {}).get("few_shot", {}).get("num_examples", 5)
    few_shot_pool = _sample_few_shot_pool(eval_data, num_examples) if args.mode == "few_shot" else {}

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for ckpt in args.checkpoints:
        ckpt_path = resolve_path(ckpt)
        name = ckpt_path.name
        logger.info("Evaluating checkpoint %s on %s split (%s)", name, args.split, args.mode)
        result = evaluate_checkpoint(
            ckpt_path, eval_data, raw_dir, max_seq_length, mode=args.mode, few_shot_pool=few_shot_pool, num_examples=num_examples
        )
        summary[name] = result["metrics"]
        out_path = out_dir / f"{name}_{args.split}_{args.mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote %s", out_path)

    logger.info("Summary (%s split, %s): %s", args.split, args.mode, summary)


if __name__ == "__main__":
    main()
