#!/usr/bin/env python
"""CLI entry point: run zero-shot / few-shot baseline evaluation.

Usage:
    python scripts/run_baseline_eval.py [--limit N] [--modes zero_shot few_shot]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.eval.baseline_eval import main

if __name__ == "__main__":
    main()
