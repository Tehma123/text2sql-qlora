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
| Zero-shot (3B base) | 26.9% | 66.3% |
| Few-shot (3B base, 5-shot) | 50.9% | 74.3% |
| QLoRA fine-tuned v1, zero-shot | 43.0% | 71.9% |
| QLoRA fine-tuned v1 + few-shot | 53.1% | 76.2% |
| QLoRA fine-tuned v2, zero-shot | 44.4% | 71.7% |
| **QLoRA fine-tuned v2 + few-shot** | **54.0%** | **77.4%** |
| Target | — | ~80% |

Fine-tuning alone clearly beats zero-shot (v2: +5.4pt execution_accuracy) but falls short of the few-shot baseline — the most likely explanation is that few-shot is shown 5 worked examples *from the same unseen test database* at inference time (a live demonstration of that schema's naming conventions), which the fine-tuned model, evaluated zero-shot, doesn't get access to. Combining both — fine-tuning *and* few-shot demonstrations at inference time — beats each baseline individually (v2: +11.1pt over zero-shot, +3.1pt over few-shot), confirming they provide complementary rather than redundant signal. Still 2.6pt short of the ~80% target. Error analysis by query complexity shows why: JOIN queries (40% of the test set) are the weak point in every condition, and combining fine-tuning + few-shot helps most exactly there (v1: +6.1pt over few-shot alone), not on the already-easy simple-SELECT category.

**v1 vs. v2:** v2 retrains on schema prompts with corrected composite primary keys (Spider's `tables.json` keeps only a composite key's first column) and selects its checkpoint by dev execution accuracy at 100-step resolution instead of once per epoch. It is not significantly better than v1 (zero-shot 71.7% vs. 71.9%; with few-shot 77.4% vs. 76.2%, p=0.06), but it is the best measured result. The baselines were measured before the primary-key fix and haven't been re-run on the corrected prompts yet. Full trace and reasoning in `reports/qlora_training.md`.

## Demo

```bash
python demo/app.py
```

Loads `checkpoints/v2/checkpoint-600` and serves a local Gradio UI (default `http://127.0.0.1:7860`): pick one of 6 curated held-out databases, see its schema and real example questions from that database, and ask your own question — generated live using the fine-tuned model + few-shot combination above (the 77.4% row), with the SQL executed against the real SQLite file so you see actual query results, not just generated text.

## Success Criteria

- Execution accuracy of the fine-tuned model shows a clear, measurable improvement over both baselines. Target: **~80%** on the `test` split, against Qwen2.5-Coder-3B-Instruct's own measured baselines (66.3% zero-shot, 74.3% few-shot — see `reports/baseline_evaluation.md`). **Result: 77.4%** (v2 fine-tuned + few-shot) — clears both baselines, falls short of the target by 2.6pt; see the v2 section of `reports/qlora_training.md`
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
- [ ] Re-run both baselines on the corrected schema prompts

Full write-up: [`reports/technical_report.md`](reports/technical_report.md). Working reports with the complete step-by-step trace (bugs hit, decisions made, and why) are in [`reports/`](reports/).

## Contributing

The initial pipeline is complete — see [Results](#results) and the [technical report](reports/technical_report.md). Issues and pull requests are welcome, particularly around the join-query weakness identified in error analysis.

## License

MIT — see `LICENSE` for details.
