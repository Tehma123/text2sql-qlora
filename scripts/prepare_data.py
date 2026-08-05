#!/usr/bin/env python
"""CLI entry point: build instruction-formatted train/dev/test splits in data/processed/.

Usage:
    python scripts/prepare_data.py [--config configs/data_config.yaml]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.data.preprocess import main

if __name__ == "__main__":
    main()
