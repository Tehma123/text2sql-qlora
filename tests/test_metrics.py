import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.eval.metrics import exact_match, execution_accuracy, aggregate


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "toy.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE head (head_id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    con.executemany(
        "INSERT INTO head VALUES (?, ?, ?)",
        [(1, "Alice", 60), (2, "Bob", 45), (3, "Carol", 70)],
    )
    con.commit()
    con.close()
    return path


class TestExactMatch:
    def test_identical_queries_match(self):
        assert exact_match("SELECT count(*) FROM head", "SELECT count(*) FROM head")

    def test_whitespace_and_case_differences_match(self):
        assert exact_match(
            "select   COUNT(*)   from Head",
            "SELECT count(*) FROM head",
        )

    def test_different_predicate_does_not_match(self):
        assert not exact_match(
            "SELECT count(*) FROM head WHERE age > 56",
            "SELECT count(*) FROM head WHERE age > 60",
        )

    def test_unparseable_prediction_does_not_crash(self):
        assert not exact_match("this is not sql", "SELECT count(*) FROM head")


class TestExecutionAccuracy:
    def test_syntactically_different_equivalent_queries_match(self, db_path):
        pred = "SELECT COUNT(*) FROM head WHERE age  >  56"
        gold = "select count(*) from head where age>56"
        assert execution_accuracy(pred, gold, db_path)

    def test_wrong_result_does_not_match(self, db_path):
        pred = "SELECT COUNT(*) FROM head WHERE age > 60"
        gold = "SELECT COUNT(*) FROM head WHERE age > 56"
        assert not execution_accuracy(pred, gold, db_path)

    def test_invalid_predicted_sql_does_not_match(self, db_path):
        assert not execution_accuracy("SELECT * FROM nonexistent_table", "SELECT count(*) FROM head", db_path)

    def test_row_order_does_not_matter(self, db_path):
        pred = "SELECT name FROM head ORDER BY age ASC"
        gold = "SELECT name FROM head ORDER BY age DESC"
        assert execution_accuracy(pred, gold, db_path)

    def test_write_attempts_are_rejected(self, db_path):
        assert execution_accuracy("DELETE FROM head", "SELECT count(*) FROM head", db_path) is False
        # confirm the db_path was genuinely opened read-only, not just that
        # the prediction happened to fail for some other reason
        con = sqlite3.connect(db_path)
        assert con.execute("SELECT count(*) FROM head").fetchone()[0] == 3
        con.close()


class TestAggregate:
    def test_empty_results(self):
        assert aggregate([]) == {"n": 0, "exact_match": 0.0, "execution_accuracy": 0.0}

    def test_averages_over_results(self):
        results = [
            {"exact_match": True, "execution_accuracy": True},
            {"exact_match": False, "execution_accuracy": True},
            {"exact_match": False, "execution_accuracy": False},
            {"exact_match": False, "execution_accuracy": False},
        ]
        metrics = aggregate(results)
        assert metrics["n"] == 4
        assert metrics["exact_match"] == pytest.approx(0.25)
        assert metrics["execution_accuracy"] == pytest.approx(0.5)
