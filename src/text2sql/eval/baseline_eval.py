"""Zero-shot / few-shot baseline evaluation of the un-finetuned base model.

Measures where the base model stands before any fine-tuning, so the value of
QLoRA fine-tuning is demonstrated against real numbers rather than assumed
(see README: "Why Fine-tuning?"). Two baselines, per configs/eval_config.yaml:
  - zero_shot: question + schema only
  - few_shot: a handful of worked (question, SQL) examples in the prompt

Few-shot examples are drawn from *other test questions on the same
database*, not from train: `test.json`'s 40 databases are entirely disjoint
from train/dev (see reports/data_preparation.md), so there is no train
example sharing a schema with any test question to draw a demonstration
from. Showing a few worked examples on an unfamiliar database before asking
the real question is itself standard practice for cross-domain few-shot
prompting.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from text2sql.data.schema_utils import get_db_path
from text2sql.eval.metrics import aggregate, exact_match, execution_accuracy, normalize_sql
from text2sql.models.load_model import load_base_model
from text2sql.prompts.templates import build_few_shot_messages, build_messages
from text2sql.utils.config import load_config, resolve_path
from text2sql.utils.logging import get_logger

logger = get_logger(__name__)


def _load_test_set(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sample_few_shot_pool(test_data: list[dict[str, Any]], seed: int = 42) -> dict[str, list]:
    """Every example of each database, in a fixed random order; `select_demos`
    takes the first ones that don't give away the answer."""
    rng = random.Random(seed)
    by_db = defaultdict(list)
    for ex in test_data:
        by_db[ex["db_id"]].append(ex)

    pool = {}
    for db_id, examples in by_db.items():
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        pool[db_id] = [examples[i] for i in indices]
    return pool


def select_demos(ex: dict[str, Any], few_shot_pool: dict[str, list], num_examples: int) -> list[dict[str, str]]:
    """First `num_examples` pool entries on `ex`'s database that are neither
    `ex` itself nor carry its gold SQL.

    Spider asks most gold queries in two paraphrased forms, so excluding only
    the identical question text let the paraphrase through as a demo with
    the exact answer (164 of 2,147 test examples before this check; see
    reports/qlora_training.md). Queries are compared after `normalize_sql`,
    the same canonicalization exact_match uses.
    """
    gold = normalize_sql(ex["query"])
    demos = []
    for c in few_shot_pool[ex["db_id"]]:
        if c["question"] == ex["question"] or normalize_sql(c["query"]) == gold:
            continue
        demos.append({"question": c["question"], "query": c["query"]})
        if len(demos) == num_examples:
            break
    return demos


def _extract_sql(generated_text: str) -> str:
    """Strip whitespace/markdown fences the model might add despite instructions."""
    text = generated_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("sql"):
            text = text[3:]
    return text.strip().rstrip(";").strip()


def generate_sql(
    model, tokenizer, messages: list[dict[str, str]], max_seq_length: int, max_new_tokens: int = 256
) -> str:
    import torch

    # If a prompt is still over budget (e.g. an unusually large schema),
    # truncate from the left so the real question + generation prompt at the
    # end survives -- the default right-truncation would instead cut off the
    # part of the prompt that actually asks the question.
    tokenizer.truncation_side = "left"
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return _extract_sql(tokenizer.decode(new_tokens, skip_special_tokens=True))


def run_baseline(
    mode: str,
    test_data: list[dict[str, Any]],
    few_shot_pool: dict[str, list],
    model,
    tokenizer,
    raw_dir: Path,
    num_examples: int,
    max_seq_length: int,
) -> dict[str, Any]:
    results = []
    for i, ex in enumerate(test_data):
        if mode == "zero_shot":
            messages = build_messages(ex["question"], ex["schema"])
        else:
            demos = select_demos(ex, few_shot_pool, num_examples)
            messages = build_few_shot_messages(ex["question"], ex["schema"], demos)

        pred_sql = generate_sql(model, tokenizer, messages, max_seq_length)
        db_path = get_db_path(ex["db_id"], raw_dir, ex["db_dir"])

        results.append(
            {
                "db_id": ex["db_id"],
                "question": ex["question"],
                "gold": ex["query"],
                "pred": pred_sql,
                "exact_match": exact_match(pred_sql, ex["query"]),
                "execution_accuracy": execution_accuracy(pred_sql, ex["query"], db_path),
            }
        )
        if (i + 1) % 50 == 0 or (i + 1) == len(test_data):
            running = aggregate(results)
            logger.info(
                "%s: %d/%d | exact_match=%.3f execution_accuracy=%.3f",
                mode,
                i + 1,
                len(test_data),
                running["exact_match"],
                running["execution_accuracy"],
            )

    return {"mode": mode, "metrics": aggregate(results), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot / few-shot baseline evaluation")
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--modes", nargs="+", choices=["zero_shot", "few_shot"], default=None)
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N test examples")
    parser.add_argument("--output-dir", default="results/baselines")
    args = parser.parse_args()

    eval_config = load_config(args.eval_config)
    data_config = load_config(args.data_config)

    test_data = _load_test_set(resolve_path(eval_config["test_file"]))
    if args.limit:
        test_data = test_data[: args.limit]
    logger.info("Loaded %d test examples", len(test_data))

    baselines_cfg = eval_config["baselines"]
    modes = args.modes or [m for m in ("zero_shot", "few_shot") if baselines_cfg.get(m, {}).get("enabled")]
    num_examples = baselines_cfg.get("few_shot", {}).get("num_examples", 5)

    raw_dir = resolve_path(data_config["raw_dir"])
    few_shot_pool = _sample_few_shot_pool(test_data) if "few_shot" in modes else {}

    max_seq_length = data_config["max_seq_length"]
    model, tokenizer = load_base_model(max_seq_length)

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for mode in modes:
        result = run_baseline(mode, test_data, few_shot_pool, model, tokenizer, raw_dir, num_examples, max_seq_length)
        summary[mode] = result["metrics"]
        out_path = out_dir / f"{mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote %s", out_path)

    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
