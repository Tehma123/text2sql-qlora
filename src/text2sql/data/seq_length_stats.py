"""Measure how many processed examples exceed a given max_seq_length.

`preprocess.py` already drops over-budget examples at whatever
`max_seq_length` is currently configured. This script exists to explore
*candidate* budgets (via `--max-seq-length`) against already-built splits
before committing to one in the config and re-running preprocessing --
see reports/data_preparation.md for how 1536 was chosen over 1024.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from text2sql.utils.config import load_config, resolve_path
from text2sql.utils.logging import get_logger

logger = get_logger(__name__)


def _load_split(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_lengths(examples: list[dict[str, Any]], tokenizer) -> list[int]:
    return [len(tokenizer.apply_chat_template(ex["messages"], tokenize=True)) for ex in examples]


def summarize(lengths: list[int], max_seq_length: int) -> dict[str, Any]:
    n = len(lengths)
    s = sorted(lengths)
    over = sum(1 for length in lengths if length > max_seq_length)

    def pct(p: int) -> int:
        return s[min(n - 1, int(p / 100 * n))]

    return {
        "n": n,
        "min": s[0],
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": s[-1],
        "over_max_seq_length": over,
        "over_pct": round(100 * over / n, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure token-length distribution of processed splits against max_seq_length"
    )
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml", help="source of base_model")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument(
        "--max-seq-length", type=int, default=None, help="override data_config's max_seq_length"
    )
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    train_config = load_config(args.train_config)
    max_seq_length = args.max_seq_length or data_config["max_seq_length"]
    base_model = train_config["base_model"]

    from transformers import AutoTokenizer

    logger.info("Loading tokenizer for %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    for split in args.splits:
        path = resolve_path(data_config[f"{split}_file"])
        examples = _load_split(path)
        lengths = compute_lengths(examples, tokenizer)
        stats = summarize(lengths, max_seq_length)
        logger.info(
            "%-5s n=%-5d over %d tokens: %d (%.2f%%) | p50=%d p90=%d p95=%d p99=%d max=%d",
            split,
            stats["n"],
            max_seq_length,
            stats["over_max_seq_length"],
            stats["over_pct"],
            stats["p50"],
            stats["p90"],
            stats["p95"],
            stats["p99"],
            stats["max"],
        )


if __name__ == "__main__":
    main()
