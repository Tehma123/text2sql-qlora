"""Exact-match and execution-accuracy metrics for generated SQL.

Exact match compares *structure*, not raw strings -- two queries that differ
only in whitespace, casing, or quoting style should still count as a match
(see README: "Exact Match checks structural correctness"). Execution
accuracy runs both queries against the real per-database SQLite file and
compares result sets, since equivalent SQL can be written many different
ways syntactically.
"""

import sqlite3
import time
from pathlib import Path
from typing import Any

import sqlglot

DEFAULT_TIMEOUT = 5.0


def normalize_sql(sql: str) -> str:
    """Canonicalize SQL so structurally-equivalent queries compare equal.

    Falls back to plain whitespace/case normalization if sqlglot can't parse
    the query -- a genuinely malformed prediction should count as a
    mismatch, not crash the eval loop.
    """
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        return tree.sql(dialect="sqlite", normalize=True)
    except Exception:
        return " ".join(sql.strip().lower().rstrip(";").split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_sql(pred) == normalize_sql(gold)


def execute_query(sql: str, db_path: str | Path, timeout: float = DEFAULT_TIMEOUT) -> list[tuple] | None:
    """Run `sql` read-only against the SQLite db. Returns None on any failure
    (invalid SQL, timeout, or an attempted write -- gold Spider queries are
    always read-only, so a write attempt is itself a wrong prediction)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    start = time.monotonic()

    def _abort_if_slow(*_args) -> int:
        return 1 if time.monotonic() - start > timeout else 0

    con.set_progress_handler(_abort_if_slow, 1000)
    try:
        cur = con.cursor()
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return None
    finally:
        con.close()


def _result_set(rows: list[tuple]) -> set:
    return {tuple(row) for row in rows}


def execution_accuracy(pred: str, gold: str, db_path: str | Path) -> bool:
    pred_rows = execute_query(pred, db_path)
    if pred_rows is None:
        return False
    gold_rows = execute_query(gold, db_path)
    if gold_rows is None:
        # Gold should always execute; if it doesn't, the db/gold pairing is
        # broken -- don't silently count that as a correct prediction.
        return False
    return _result_set(pred_rows) == _result_set(gold_rows)


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    n = len(results)
    if n == 0:
        return {"n": 0, "exact_match": 0.0, "execution_accuracy": 0.0}
    return {
        "n": n,
        "exact_match": sum(r["exact_match"] for r in results) / n,
        "execution_accuracy": sum(r["execution_accuracy"] for r in results) / n,
    }
