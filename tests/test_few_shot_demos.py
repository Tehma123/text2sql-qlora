import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.eval.baseline_eval import _sample_few_shot_pool, select_demos


def _ex(question, query, db_id="db"):
    return {"db_id": db_id, "question": question, "query": query}


def test_excludes_paraphrase_with_same_gold_sql():
    target = _ex("How many heads are there?", "SELECT count(*) FROM head")
    pool = {
        "db": [
            _ex("Count the heads.", "select COUNT(*) from head;"),
            _ex("List all names.", "SELECT name FROM head"),
            target,
            _ex("Who is the oldest?", "SELECT name FROM head ORDER BY age DESC LIMIT 1"),
        ]
    }
    demos = select_demos(target, pool, num_examples=5)
    assert [d["question"] for d in demos] == ["List all names.", "Who is the oldest?"]


def test_takes_first_n_in_pool_order():
    target = _ex("q0", "SELECT 0")
    pool = {"db": [_ex(f"q{i}", f"SELECT {i}") for i in range(1, 9)]}
    demos = select_demos(target, pool, num_examples=5)
    assert [d["question"] for d in demos] == ["q1", "q2", "q3", "q4", "q5"]


def test_pool_is_per_database_and_complete():
    data = [_ex(f"a{i}", f"SELECT {i}", "a") for i in range(10)] + [_ex("b0", "SELECT 0", "b")]
    pool = _sample_few_shot_pool(data)
    assert sorted(pool) == ["a", "b"]
    assert sorted(e["question"] for e in pool["a"]) == sorted(f"a{i}" for i in range(10))
    assert pool == _sample_few_shot_pool(data)
