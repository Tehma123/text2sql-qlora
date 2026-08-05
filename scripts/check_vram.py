#!/usr/bin/env python
"""CLI entry point: measure actual peak VRAM for one QLoRA training step.

Usage:
    python scripts/check_vram.py [--batch-size 2] [--data-config ...] [--train-config ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.train.vram_check import main

if __name__ == "__main__":
    main()
