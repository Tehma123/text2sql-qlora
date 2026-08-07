"""Break down evaluation results by SQL query complexity.

Categories (`configs/eval_config.yaml: error_analysis.categories`) are
derived from the gold SQL via keyword heuristics, matching Spider's own
established complexity taxonomy (presence of a join / subquery /
aggregation). Not mutually exclusive -- a query can be both a join and an
aggregation -- except "select", which means none of the other three apply
(the simple baseline category).

Works on any `run_baseline`-shaped results file (baseline_eval.py or
evaluate.py output: `{"metrics": ..., "results": [{"gold": ..., "pred": ...,
"exact_match": ..., "execution_accuracy": ...}, ...]}`), so the same
breakdown applies uniformly to baselines and fine-tuned checkpoints.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from text2sql.utils.config import load_config, resolve_path
from text2sql.utils.logging import get_logger

logger = get_logger(__name__)

_AGG_FUNCS = ("COUNT(", "SUM(", "AVG(", "MIN(", "MAX(")


def categorize_query(sql: str) -> list[str]:
    sql_upper = sql.upper()
    categories = []
    if "JOIN" in sql_upper:
        categories.append("join")
    if sql_upper.count("SELECT") > 1:
        categories.append("subquery")
    if any(fn in sql_upper for fn in _AGG_FUNCS) or "GROUP BY" in sql_upper:
        categories.append("aggregation")
    if not categories:
        categories.append("select")
    return categories


def breakdown_by_category(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        for cat in categorize_query(r["gold"]):
            buckets[cat].append(r)

    breakdown = {}
    for cat, items in buckets.items():
        n = len(items)
        breakdown[cat] = {
            "n": n,
            "exact_match": sum(1 for i in items if i["exact_match"]) / n,
            "execution_accuracy": sum(1 for i in items if i["execution_accuracy"]) / n,
        }
    return breakdown


def _load_results(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["results"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Break down evaluation results by query complexity")
    parser.add_argument("--results", nargs="+", required=True, help="one or more results JSON files (from baseline_eval.py or evaluate.py)")
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    parser.add_argument("--output-dir", default="results/error_analysis")
    args = parser.parse_args()

    eval_config = load_config(args.eval_config)
    categories = eval_config.get("error_analysis", {}).get("categories", ["select", "join", "subquery", "aggregation"])

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for results_arg in args.results:
        path = resolve_path(results_arg)
        results = _load_results(path)
        breakdown = breakdown_by_category(results)
        name = path.stem
        summary[name] = breakdown
        logger.info("%s (n=%d total):", name, len(results))
        for cat in categories:
            if cat in breakdown:
                b = breakdown[cat]
                logger.info(
                    "  %-12s n=%4d  exact_match=%.3f  execution_accuracy=%.3f",
                    cat,
                    b["n"],
                    b["exact_match"],
                    b["execution_accuracy"],
                )

    out_path = out_dir / "breakdown.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
