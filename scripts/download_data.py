#!/usr/bin/env python
"""CLI entry point: download the Spider dataset into data/raw/.

Usage:
    python scripts/download_data.py [--config configs/data_config.yaml] [--force]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.data.download_spider import main

if __name__ == "__main__":
    main()
