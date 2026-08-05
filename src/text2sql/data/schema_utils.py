"""Turn Spider's tables.json schema entries into DDL-style strings for prompts.

Spider ships one entry per database in tables.json, with columns/tables addressed
by parallel index arrays rather than names -- see
https://github.com/taoyds/spider#tables. This module converts that indexed form
into a `CREATE TABLE` string per table, which is the schema representation fed
into the model's prompt (see `text2sql.prompts.templates`).
"""

import json
from pathlib import Path
from typing import Any

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


def schema_to_ddl(schema: Schema) -> str:
    """Render one tables.json entry as CREATE TABLE statements.

    Column/table names use the *original* (unnormalized) casing so the DDL
    matches what a query against the real SQLite database would need.
    """
    table_names = schema["table_names_original"]
    column_names = schema["column_names_original"]  # [(table_idx, col_name), ...], table_idx -1 is "*"
    column_types = schema["column_types"]
    primary_keys = set(schema["primary_keys"])
    foreign_keys = schema["foreign_keys"]  # [[col_idx, other_col_idx], ...]

    # Group column indices (skipping the "*" placeholder at index 0) by table
    columns_by_table: dict[int, list[int]] = {i: [] for i in range(len(table_names))}
    for col_idx, (table_idx, _) in enumerate(column_names):
        if table_idx == -1:
            continue
        columns_by_table[table_idx].append(col_idx)

    fk_by_col = {col_idx: other_idx for col_idx, other_idx in foreign_keys}

    statements = []
    for table_idx, table_name in enumerate(table_names):
        lines = []
        for col_idx in columns_by_table[table_idx]:
            _, col_name = column_names[col_idx]
            col_type = column_types[col_idx].upper()
            suffix = " PRIMARY KEY" if col_idx in primary_keys else ""
            lines.append(f'  "{col_name}" {col_type}{suffix}')

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


def build_schema_index(tables_path: str | Path) -> dict[str, str]:
    """Map every db_id to its rendered DDL schema string."""
    tables = load_tables(tables_path)
    return {db_id: schema_to_ddl(schema) for db_id, schema in tables.items()}
