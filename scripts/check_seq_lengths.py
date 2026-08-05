#!/usr/bin/env python
"""CLI entry point: report what fraction of processed examples exceed max_seq_length.

Usage:
    python scripts/check_seq_lengths.py [--data-config configs/data_config.yaml] [--train-config configs/train_config.yaml]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.data.seq_length_stats import main

if __name__ == "__main__":
    main()
