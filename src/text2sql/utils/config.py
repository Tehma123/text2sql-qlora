"""YAML config loading shared by all scripts."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, resolving relative paths against the project root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Resolve a path from a config value against the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
