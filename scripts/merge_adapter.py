#!/usr/bin/env python
"""CLI entry point: merge a LoRA checkpoint into the fp16 base model for ONNX export.

Usage:
    python scripts/merge_adapter.py --adapter checkpoints/v2/checkpoint-600 \
        --output checkpoints/v2/merged-600-fp16 \
        [--verify 200 --reference results/finetuned/v2_retrain/checkpoint-600_dev_zero_shot.json]
"""

import sys
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.models.merge_adapter import main

if __name__ == "__main__":
    main()
