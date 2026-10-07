import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.data.schema_utils import read_primary_keys, schema_to_ddl

# Mirrors how Spider's tables.json describes a composite key: `primary_keys`
# lists only its first column (student.id here, enrollment.student_id), and
# `log` has no primary key at all.
SCHEMA = {
    "db_id": "toy",
    "table_names_original": ["student", "enrollment", "log"],
    "column_names_original": [
        [-1, "*"],
        [0, "id"],
        [0, "name"],
        [1, "student_id"],
        [1, "course_id"],
        [1, "grade"],
        [2, "message"],
    ],
    "column_types": ["text", "number", "text", "number", "number", "text", "text"],
    "primary_keys": [1, 3],
    "foreign_keys": [[3, 1]],
}


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "toy.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE student (id INTEGER PRIMARY KEY, name TEXT)")
    con.execute(
        "CREATE TABLE enrollment (student_id INTEGER, course_id INTEGER, grade TEXT, "
        "PRIMARY KEY (student_id, course_id), FOREIGN KEY (student_id) REFERENCES student(id))"
    )
    con.execute("CREATE TABLE log (message TEXT)")
    con.commit()
    con.close()
    return path


def test_read_primary_keys(db_path):
    assert read_primary_keys(db_path) == {
        "student": ["id"],
        "enrollment": ["student_id", "course_id"],
        "log": [],
    }


def test_composite_key_rendered_as_table_constraint(db_path):
    ddl = schema_to_ddl(SCHEMA, db_path)
    assert '"student_id" NUMBER,' in ddl  # no inline PRIMARY KEY on the first key column
    assert 'PRIMARY KEY ("student_id", "course_id")' in ddl


def test_single_key_and_no_key_unchanged(db_path):
    ddl = schema_to_ddl(SCHEMA, db_path)
    assert '"id" NUMBER PRIMARY KEY' in ddl
    log_table = ddl.split('CREATE TABLE "log"')[1]
    assert "PRIMARY KEY" not in log_table


def test_table_missing_from_db_keeps_tables_json_key(db_path):
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE enrollment")
    con.commit()
    con.close()
    ddl = schema_to_ddl(SCHEMA, db_path)
    assert '"student_id" NUMBER PRIMARY KEY' in ddl


def test_falls_back_to_tables_json_without_db(tmp_path):
    ddl = schema_to_ddl(SCHEMA, tmp_path / "missing.sqlite")
    assert '"student_id" NUMBER PRIMARY KEY' in ddl
    assert ddl == schema_to_ddl(SCHEMA)
