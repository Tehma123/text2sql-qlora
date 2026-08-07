#!/usr/bin/env python
"""CLI entry point: break down evaluation results by query complexity.

Usage:
    python scripts/error_analysis.py --results results/finetuned/checkpoint-536_test.json results/finetuned/checkpoint-536_test_few_shot.json results/baselines/zero_shot.json results/baselines/few_shot.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.eval.error_analysis import main

if __name__ == "__main__":
    main()
