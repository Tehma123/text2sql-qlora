#!/usr/bin/env python
"""CLI entry point: QLoRA fine-tune the base model on the Spider train split.

Usage:
    python scripts/train.py [--resume-from-checkpoint PATH] [--max-steps N] [--limit N]
"""

import sys
from pathlib import Path

# Unsloth prints emoji (e.g. sloth U+1F9A5) straight to stdout; Windows'
# default console codepage (cp1252) can't encode them and crashes the run
# right after the multi-GB model load. Force UTF-8 so training doesn't die
# on a print statement.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.train.train_qlora import main

if __name__ == "__main__":
    main()
