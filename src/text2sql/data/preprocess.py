"""Build instruction-formatted train/dev/test splits from raw Spider data.

Spider ships a genuinely held-out test split: `train_spider.json` +
`train_others.json` -> `train`, `dev.json` -> `dev` (used to monitor training
and for baseline comparisons), and `test.json` -> `test` (never trained on;
40 of its 206 databases don't appear anywhere in train/dev, so it also
measures cross-domain generalization). train/dev share `tables.json` +
`database/`; test has its own `test_tables.json` + `test_database/` because
its extra databases aren't described in `tables.json`.

Examples whose full chat sequence exceeds `max_seq_length` (tokenized with the
real base-model tokenizer) are dropped here rather than left for the trainer
to truncate -- truncating from the right would clip the gold SQL answer, not
the schema, teaching the model on incomplete queries. See
reports/data_preparation.md for the token-length distribution that motivated
this and the choice of 1536 over the originally-planned 1024.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from text2sql.data.schema_utils import build_schema_index
from text2sql.prompts.templates import build_training_messages
from text2sql.utils.config import load_config, resolve_path
from text2sql.utils.logging import get_logger

logger = get_logger(__name__)


def _load_json(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _format_example(raw: dict[str, Any], schema_index: dict[str, str], db_dir: str) -> dict[str, Any]:
    db_id = raw["db_id"]
    question = raw["question"]
    query = raw["query"]
    schema = schema_index[db_id]
    return {
        "db_id": db_id,
        "db_dir": db_dir,  # which raw_dir subfolder holds this db's .sqlite file
        "question": question,
        "query": query,
        "schema": schema,
        "messages": build_training_messages(question, schema, query),
    }


def _filter_by_length(
    examples: list[dict[str, Any]], tokenizer, max_seq_length: int, split_name: str
) -> list[dict[str, Any]]:
    kept = []
    dropped = 0
    for ex in examples:
        length = len(tokenizer.apply_chat_template(ex["messages"], tokenize=True))
        if length > max_seq_length:
            dropped += 1
            continue
        kept.append(ex)
    if dropped:
        logger.info(
            "%s: dropped %d/%d examples over %d tokens", split_name, dropped, len(examples), max_seq_length
        )
    return kept


def build_dataset(config: dict[str, Any], base_model: str) -> dict[str, int]:
    raw_dir = resolve_path(config["raw_dir"])
    processed_dir = resolve_path(config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    max_seq_length = config["max_seq_length"]

    trainval_schema_index = build_schema_index(raw_dir / "tables.json")
    test_schema_index = build_schema_index(raw_dir / "test_tables.json")

    train_raw = _load_json(raw_dir / "train_spider.json") + _load_json(raw_dir / "train_others.json")
    dev_raw = _load_json(raw_dir / "dev.json")
    test_raw = _load_json(raw_dir / "test.json")

    splits = {
        "train": [_format_example(r, trainval_schema_index, "database") for r in train_raw],
        "dev": [_format_example(r, trainval_schema_index, "database") for r in dev_raw],
        "test": [_format_example(r, test_schema_index, "test_database") for r in test_raw],
    }

    from transformers import AutoTokenizer

    logger.info("Loading tokenizer for %s to filter by sequence length", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    counts = {}
    for split_name, examples in splits.items():
        examples = _filter_by_length(examples, tokenizer, max_seq_length, split_name)
        out_path = resolve_path(config[f"{split_name}_file"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2)
        counts[split_name] = len(examples)
        logger.info("Wrote %d examples to %s", len(examples), out_path)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Spider into instruction-formatted splits")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument(
        "--train-config",
        default="configs/train_config.yaml",
        help="source of base_model for the length-filtering tokenizer",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_config = load_config(args.train_config)
    counts = build_dataset(config, train_config["base_model"])
    logger.info("Done: %s", counts)


if __name__ == "__main__":
    main()
