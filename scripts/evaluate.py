#!/usr/bin/env python
"""CLI entry point: evaluate fine-tuned checkpoints.

Usage:
    python scripts/evaluate.py --checkpoints checkpoints/checkpoint-536 checkpoints/checkpoint-1072 checkpoints/checkpoint-1608 --split dev --limit 300
"""

import sys
from pathlib import Path

# See scripts/train.py: Unsloth prints emoji straight to stdout, which
# crashes on Windows' default cp1252 console codepage.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.eval.evaluate import main

if __name__ == "__main__":
    main()
