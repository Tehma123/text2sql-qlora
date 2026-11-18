#!/usr/bin/env python
"""CLI entry point: export a merged fp16 model to q4f16 ONNX for Transformers.js (WebGPU).

Usage:
    python scripts/export_onnx.py --merged checkpoints/v2/merged-600-fp16 \
        --output checkpoints/v2/onnx-600-genai \
        [--verify 200 --reference results/finetuned/v2_retrain/checkpoint-600_dev_zero_shot.json]
"""

import sys
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.models.export_onnx import main

if __name__ == "__main__":
    main()
