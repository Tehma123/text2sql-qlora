# Text2SQL-Tune

> Fine-tuning an open-source LLM for Text-to-SQL, fully reproducible on a single consumer GPU (RTX 3060 12GB).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#requirements)
[![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)](#roadmap)

## Table of Contents

- [Overview](#overview)
- [Why Fine-tuning?](#why-fine-tuning)
- [Approach](#approach)
- [Tech Stack](#tech-stack)
- [Hardware Requirements](#hardware-requirements)
- [Evaluation](#evaluation)
- [Results](#results)
- [Demo](#demo)
- [Browser models](#browser-models)
- [Success Criteria](#success-criteria)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview

**Text2SQL-Tune** is an open-source project that fine-tunes a small open-weight LLM to translate natural language questions into executable SQL queries.

The goal is to provide a fully reproducible, end-to-end pipeline — from raw dataset to a working demo — that anyone can run on a single consumer-grade GPU, without relying on cloud compute or paid APIs.

```
"What is the average revenue per region in Q1?"
                    │
                    ▼
        SELECT region, AVG(revenue)
        FROM sales
        WHERE quarter = 'Q1'
        GROUP BY region;
```

This project is meant to be a reference implementation: clear enough to learn from, and rigorous enough (proper baselines, error analysis, reproducible scripts) to be trusted as more than a demo.

## Why Fine-tuning?

Before reaching for fine-tuning, it's worth asking whether prompting or RAG alone would be enough. For Text-to-SQL specifically, they tend to fall short:

| Approach | Limitation for Text-to-SQL |
|---|---|
| **Prompt engineering (few-shot)** | Struggles with complex schemas (many tables, foreign keys); prone to hallucinating column/table names. Stuffing more examples into the prompt increases cost and latency without fixing the core issue. |
| **RAG (schema retrieval)** | Helps reduce schema noise, but doesn't teach the model SQL-specific patterns — it still tends to fail on nested subqueries and multi-table joins. |
| **Fine-tuning** | The model learns the question → SQL mapping directly from thousands of examples, improving syntactic correctness and generalization to new questions on similar schemas. |

This project treats that comparison as a first-class deliverable, not an assumption: baselines are measured and reported alongside the fine-tuned model, so the value of fine-tuning is demonstrated, not just claimed.

## Approach

### Pipeline

```
Dataset (Spider)
      │
      ▼
Preprocessing & instruction formatting
      │
      ▼
Baseline evaluation (zero-shot / few-shot on base model)
      │
      ▼
QLoRA fine-tuning (Unsloth)
      │
      ▼
Post-tuning evaluation (compared against baseline)
      │
      ▼
Error analysis
      │
      ▼
Demo (Gradio) + technical report
```

### Design decisions

| Component | Choice | Rationale |
|---|---|---|
| **Dataset** | [Spider](https://yale-lily.github.io/spider) | Standard Text-to-SQL benchmark, ~7,000 training examples, diverse schemas, established leaderboard for comparison |
| **Base model** | Qwen2.5-Coder-3B-Instruct | Strong code/SQL capabilities for its size; started at 7B, but a real multi-hour training run (not just a single-step VRAM check) showed 7B doesn't actually fit *comfortably* in 12GB — it drove dedicated VRAM to ~11.3GB and nearly exhausted BAR1 memory, throttling training to ~5x slower than measured. 3B re-measured at 4.09GB/4.72GB peak, leaving >7GB of headroom. See `reports/qlora_training.md` for the full incident |
| **Fine-tuning method** | QLoRA (4-bit) via Unsloth | VRAM-efficient, faster than plain PEFT, the standard approach for local fine-tuning today |
| **Evaluation** | Exact Match + Execution Accuracy | Exact Match checks structural correctness; Execution Accuracy checks whether the generated SQL returns the correct result against a real database (more meaningful, since equivalent SQL can differ syntactically) |
| **Experiment tracking** | Weights & Biases | Tracks loss curves and enables comparison across hyperparameter runs |
| **Demo** | Gradio | Lightweight interface, deployable for free on Hugging Face Spaces |

## Tech Stack

- Python 3.10+
- PyTorch
- Hugging Face `transformers`, `datasets`, `peft`
- [Unsloth](https://github.com/unslothai/unsloth) for efficient QLoRA training
- `bitsandbytes` for 4-bit quantization
- Weights & Biases for experiment tracking
- Gradio for the demo UI

## Hardware Requirements

This project is designed to run on a single consumer GPU:

- **Minimum**: NVIDIA GPU with 12GB VRAM (e.g. RTX 3060 12GB), 32GB system RAM
- 4-bit quantization keeps the model comfortably within VRAM limits
- Small batch size (1–2) with gradient accumulation (8–16 steps) simulates a larger effective batch
- Max sequence length capped around 1024 tokens (sufficient for question + schema + SQL in Spider)
- LoRA rank starts at 16–32

No cloud GPU is required to reproduce the core results, though the roadmap includes optional larger-scale runs on rented compute.

## Evaluation

Two metrics are used, computed on a held-out test split:

- **Exact Match**: does the generated SQL structurally match the reference query?
- **Execution Accuracy**: does the generated SQL, when executed, return the same result as the reference query?

Two baselines are measured before fine-tuning to make the comparison meaningful:
1. **Zero-shot** — question + schema only, no examples
2. **Few-shot** — 3–5 examples included in the prompt

## Results

| | exact_match | execution_accuracy |
|---|---|---|
| Zero-shot (3B base) | 27.2% | 66.6% |
| Few-shot (3B base, 5-shot) | 50.3% | 73.9% |
| QLoRA fine-tuned v1, zero-shot | 43.0% | 71.9% |
| QLoRA fine-tuned v1 + few-shot | 53.1% | 76.2% |
| QLoRA fine-tuned v2, zero-shot | 44.4% | 71.7% |
| **QLoRA fine-tuned v2 + few-shot** | **54.0%** | **77.4%** |
| Target (lowered from ~80% after the results were known) | — | 75% |

The baselines and v2 use the corrected schema prompts (see "v1 vs. v2" below); v1 was measured on the original prompts, where the baselines scored 66.3% / 74.3%. Every few-shot number is about 1-1.5pt high, because for 164 test questions (7.6%) one few-shot demo is a paraphrase carrying the same gold SQL; on the remaining 1,983 questions v2 + few-shot scores 76.4% and the few-shot baseline 72.5% (details in `reports/qlora_training.md`, "Few-shot caveat"). The demo selection has since been fixed in code, but the few-shot evaluations were not re-run, so the table shows the numbers as measured.

Fine-tuning alone clearly beats zero-shot (v2: +5.1pt execution_accuracy) but does not beat the few-shot baseline. On all questions it trails by 2.2pt, but that gap is mostly the demo leak, which only lifts the few-shot conditions: on the 1,983 clean questions it is 0.6pt (71.9% vs. 72.5%, p=0.56), i.e. no measurable difference. Few-shot is shown 5 worked examples *from the same unseen test database* at inference time (a live demonstration of that schema's naming conventions), which the fine-tuned model, evaluated zero-shot, doesn't get. Combining both — fine-tuning *and* few-shot demonstrations at inference time — beats each baseline individually (v2: +10.8pt over zero-shot, +3.5pt over few-shot; +9.3pt / +3.9pt on the clean questions), confirming they provide complementary rather than redundant signal. That clears the 75% target; it is 2.6pt short of the original ~80%. Error analysis by query complexity shows why: JOIN queries (40% of the test set) are the weak point in every condition, and combining fine-tuning + few-shot helps most exactly there (v1: +6.1pt over few-shot alone), not on the already-easy simple-SELECT category.

**v1 vs. v2:** v2 retrains on schema prompts with corrected composite primary keys (Spider's `tables.json` keeps only a composite key's first column) and selects its checkpoint by dev execution accuracy at 100-step resolution instead of once per epoch. It is not significantly better than v1 (zero-shot 71.7% vs. 71.9%; with few-shot 77.4% vs. 76.2%, p=0.06), but it is the best measured result. Re-running the baselines on the corrected prompts showed the primary-key fix itself has no measurable effect on accuracy (+0.3pt zero-shot, -0.4pt few-shot, neither significant). Full trace and reasoning in `reports/qlora_training.md`.

## Demo

```bash
python demo/app.py
```

Loads `checkpoints/v2/checkpoint-600` and serves a local Gradio UI (default `http://127.0.0.1:7860`): pick one of 6 curated held-out databases, see its schema and real example questions from that database, and ask your own question — generated live using the fine-tuned model + few-shot combination above (the 77.4% row), with the SQL executed against the real SQLite file so you see actual query results, not just generated text.

Note: `checkpoints/v2/checkpoint-600` was retrained on 2026-11-18 with the same config (the original adapter was lost with the old clone). The retrained checkpoint scores 75.1% dev execution accuracy zero-shot, vs. 76.4% for the original; the test-split numbers in [Results](#results) were measured on the original and have not been re-run.

## Browser models

The fine-tuned adapters are also merged and exported to q4f16 ONNX for in-browser inference with Transformers.js on WebGPU. They power **[SQLocal](https://tehma123.github.io/text2sql-qlora/)**: drop a CSV file, ask a question in English, and get the SQL and the result table, with the model and DuckDB-WASM running entirely in the browser (Chrome or Edge on desktop with WebGPU; code in [`web/app/`](web/app/), spec and log in [`SQLocal.md`](SQLocal.md)).

| | Dev execution accuracy | Download | Speed (RTX 3060, Edge) | License |
|---|---|---|---|---|
| [Tehma/text2sql-qwen2.5-coder-1.5b-onnx](https://huggingface.co/Tehma/text2sql-qwen2.5-coder-1.5b-onnx) | 71.4% | 877 MB | ~62 tok/s, 0.35-0.7 s per query | Apache 2.0 |
| [Tehma/text2sql-qwen2.5-coder-3b-onnx](https://huggingface.co/Tehma/text2sql-qwen2.5-coder-3b-onnx) | 75.1% | 1.74 GB | ~46 tok/s, 0.5-1.0 s per query | Qwen Research (non-commercial) |

Accuracy is for the 4-bit + adapter checkpoint on the full dev split (zero-shot); the difference is significant (exact McNemar p=0.006). Through the app itself, on 50 questions about five real CSV files (seaborn's tips, titanic, penguins, mpg and diamonds; `web/e2e/benchmark.py`), the 1.5B model answers 45 and the 3B model 47 correctly. Without WebGPU the app runs a q4 version of the 1.5B model on the CPU (982 MB, about 30 s per question, multi-threaded via a COOP/COEP service worker). Reproduce with `configs/train_config_1.5b.yaml` (1.5B) or `configs/train_config.yaml` (3B), then:

```bash
python scripts/merge_adapter.py --adapter checkpoints/1.5b/checkpoint-600 --output checkpoints/1.5b/merged-600-fp16
pip install --no-deps onnxruntime-genai==0.20.0
python scripts/export_onnx.py --merged checkpoints/1.5b/merged-600-fp16 --output checkpoints/1.5b/onnx-600-genai --verify 200
python web/spike/serve.py --model checkpoints/1.5b/onnx-600-genai   # in a second terminal:
python web/spike/run_spike.py --model onnx-600-genai --reference checkpoints/1.5b/onnx-600-genai/verify_dev_model_q4f16.json
```

`--verify 200` re-scores the first 200 dev questions and writes the `verify_dev_model_q4f16.json` that the spike compares against.

The full log (decisions, bugs, measurements) is in the "Việc cần chạy cho M1" section of [`SQLocal.md`](SQLocal.md).

## Success Criteria

- Execution accuracy of the fine-tuned model shows a clear, measurable improvement over both baselines. Target: **75%** on the `test` split, against Qwen2.5-Coder-3B-Instruct's own measured baselines (66.6% zero-shot, 73.9% few-shot on the corrected prompts — see `reports/baseline_evaluation.md`). The target was ~80% until 2026-10-09, when it was lowered to 75% after the test results were known. **Result: 77.4%** (v2 fine-tuned + few-shot; 76.4% without the leaked few-shot demos) — clears both baselines significantly and meets the 75% target, though not by a statistically clear margin (95% database-level interval 73.1-82.0%); 2.6pt short of the original ~80%. See the v2 section of `reports/qlora_training.md`
- Error analysis breaks down performance by query complexity (simple SELECT, JOIN, subquery, aggregation) — **done**, see `reports/qlora_training.md`'s "Error analysis" section: JOIN queries are the weak point across every condition and the largest category (40% of test set)
- The full pipeline is reproducible — running the provided scripts on the same dataset yields comparable results

## Roadmap

- [x] Define problem scope and approach
- [x] Data preparation (download Spider, explore schema, build instruction-formatted dataset)
- [x] Baseline evaluation (zero-shot / few-shot)
- [x] QLoRA fine-tuning setup and training runs
- [x] Post-tuning evaluation and comparison
- [x] Error analysis
- [x] Gradio demo
- [x] Technical write-up
- [x] v2: fix composite primary keys in schema prompts, retrain, select the checkpoint by dev execution accuracy
- [x] Re-run both baselines on the corrected schema prompts
- [x] Acceptance review: all reported numbers recomputed from the per-example results; few-shot demo exclusion fixed in code (not re-run); target set to 75%. **Project closed 2026-10-09.**
- [x] Follow-up (2026-11): retrain 3B v2, train a 1.5B model on the same config, export both to q4f16 ONNX for the browser and publish them on the Hugging Face Hub (see [Browser models](#browser-models))
- [x] SQLocal MVP deployed to GitHub Pages: CSV upload, English question, SQL + result table, read-only check, automatic fixes, CSV export
- [x] SQLocal robustness: CPU fallback without WebGPU, charts, plain-English SQL reading, value hints and case fixes when nothing matches, every error case tested; 50-question benchmark on real CSVs
- [ ] SQLocal user testing with 5–10 people (feedback form ready: "Report a problem" in the app)

Full write-up: [`reports/technical_report.md`](reports/technical_report.md). Working reports with the complete step-by-step trace (bugs hit, decisions made, and why) are in [`reports/`](reports/).

## Contributing

The initial pipeline is complete — see [Results](#results) and the [technical report](reports/technical_report.md). Issues and pull requests are welcome, particularly around the join-query weakness identified in error analysis.

## License

MIT — see `LICENSE` for details.
