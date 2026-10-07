"""Turn Spider's tables.json schema entries into DDL-style strings for prompts.

Spider ships one entry per database in tables.json, with columns/tables addressed
by parallel index arrays rather than names -- see
https://github.com/taoyds/spider#tables. This module converts that indexed form
into a `CREATE TABLE` string per table, which is the schema representation fed
into the model's prompt (see `text2sql.prompts.templates`).

Primary keys are read from the real SQLite file when one is available, not
from tables.json: its `primary_keys` field keeps only the *first* column of a
composite key (checked against `PRAGMA table_info` across every database --
132/876 tables in tables.json and 161/1,056 in test_tables.json are affected,
e.g. college_2.takes is keyed on (ID, course_id, sec_id, semester, year) but
tables.json lists only ID). Rendering that as a single-column PRIMARY KEY
tells the model a non-unique column is unique.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from text2sql.utils.logging import get_logger

logger = get_logger(__name__)

Schema = dict[str, Any]


def load_tables(tables_path: str | Path) -> dict[str, Schema]:
    """Load tables.json and index it by db_id."""
    with open(tables_path, encoding="utf-8") as f:
        tables = json.load(f)
    return {entry["db_id"]: entry for entry in tables}


def get_db_path(db_id: str, raw_dir: str | Path, db_dir: str = "database") -> Path:
    """Path to the per-database SQLite file.

    `db_dir` is "database" for train/dev examples and "test_database" for test
    examples (see `text2sql.data.preprocess`) -- the original Spider release
    keeps them in separate top-level folders.
    """
    return Path(raw_dir) / db_dir / db_id / f"{db_id}.sqlite"


def read_primary_keys(db_path: str | Path) -> dict[str, list[str]]:
    """Map each table name (lowercased) to its primary-key columns, in key order.

    Tables without a declared primary key map to an empty list.
    """
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        tables = [row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        pks = {}
        for table in tables:
            # PRAGMA table_info rows: (cid, name, type, notnull, default, pk); pk is the
            # 1-based position within the primary key, 0 if the column isn't part of it
            info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
            pks[table.lower()] = [row[1] for row in sorted(info, key=lambda row: row[5]) if row[5] > 0]
        return pks
    finally:
        con.close()


def _primary_keys_by_table(
    schema: Schema, columns_by_table: dict[int, list[int]], db_path: str | Path | None
) -> dict[int, list[int]]:
    """Primary-key column indices per table: from the SQLite file if given, else tables.json."""
    column_names = schema["column_names_original"]
    from_json: dict[int, list[int]] = {}
    for col_idx in schema["primary_keys"]:
        from_json.setdefault(column_names[col_idx][0], []).append(col_idx)
    if db_path is None or not Path(db_path).exists():
        return from_json

    from_db = read_primary_keys(db_path)
    result = {}
    for table_idx, table_name in enumerate(schema["table_names_original"]):
        db_key = from_db.get(table_name.lower())
        idx_by_name = {column_names[c][1].lower(): c for c in columns_by_table[table_idx]}
        mapped = None if db_key is None else [idx_by_name.get(col.lower()) for col in db_key]
        if mapped is None or None in mapped:
            logger.warning(
                "%s.%s: table or its primary-key columns %s not found in the SQLite file; keeping tables.json's",
                schema["db_id"], table_name, db_key,
            )
            mapped = from_json.get(table_idx, [])
        if mapped:
            result[table_idx] = mapped
    return result


def schema_to_ddl(schema: Schema, db_path: str | Path | None = None) -> str:
    """Render one tables.json entry as CREATE TABLE statements.

    Column/table names use the *original* (unnormalized) casing so the DDL
    matches what a query against the real SQLite database would need.
    `db_path` is the database's .sqlite file, used as the source of truth for
    primary keys (see module docstring); without it, tables.json's are used.
    A single-column key is rendered inline, a composite key as a table-level
    `PRIMARY KEY (...)` clause.
    """
    table_names = schema["table_names_original"]
    column_names = schema["column_names_original"]  # [(table_idx, col_name), ...], table_idx -1 is "*"
    column_types = schema["column_types"]
    foreign_keys = schema["foreign_keys"]  # [[col_idx, other_col_idx], ...]

    # Group column indices (skipping the "*" placeholder at index 0) by table
    columns_by_table: dict[int, list[int]] = {i: [] for i in range(len(table_names))}
    for col_idx, (table_idx, _) in enumerate(column_names):
        if table_idx == -1:
            continue
        columns_by_table[table_idx].append(col_idx)

    pk_by_table = _primary_keys_by_table(schema, columns_by_table, db_path)
    fk_by_col = {col_idx: other_idx for col_idx, other_idx in foreign_keys}

    statements = []
    for table_idx, table_name in enumerate(table_names):
        pk = pk_by_table.get(table_idx, [])
        lines = []
        for col_idx in columns_by_table[table_idx]:
            _, col_name = column_names[col_idx]
            col_type = column_types[col_idx].upper()
            suffix = " PRIMARY KEY" if len(pk) == 1 and col_idx == pk[0] else ""
            lines.append(f'  "{col_name}" {col_type}{suffix}')

        if len(pk) > 1:
            pk_cols = ", ".join(f'"{column_names[c][1]}"' for c in pk)
            lines.append(f"  PRIMARY KEY ({pk_cols})")

        for col_idx in columns_by_table[table_idx]:
            if col_idx not in fk_by_col:
                continue
            _, col_name = column_names[col_idx]
            other_table_idx, other_col_name = _resolve_column(column_names, fk_by_col[col_idx])
            other_table = table_names[other_table_idx]
            lines.append(f'  FOREIGN KEY ("{col_name}") REFERENCES "{other_table}"("{other_col_name}")')

        body = ",\n".join(lines)
        statements.append(f'CREATE TABLE "{table_name}" (\n{body}\n);')

    return "\n\n".join(statements)


def _resolve_column(column_names: list[list], col_idx: int) -> tuple[int, str]:
    table_idx, col_name = column_names[col_idx]
    return table_idx, col_name


def build_schema_index(
    tables_path: str | Path, raw_dir: str | Path | None = None, db_dir: str = "database"
) -> dict[str, str]:
    """Map every db_id to its rendered DDL schema string.

    With `raw_dir`, primary keys come from each database's SQLite file under
    `raw_dir/db_dir` (see `get_db_path`) instead of tables.json.
    """
    tables = load_tables(tables_path)
    return {
        db_id: schema_to_ddl(schema, get_db_path(db_id, raw_dir, db_dir) if raw_dir is not None else None)
        for db_id, schema in tables.items()
    }
