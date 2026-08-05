# Text2SQL-Tune

> Fine-tuning an open-source LLM for Text-to-SQL, fully reproducible on a single consumer GPU (RTX 3060 12GB).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#requirements)
[![Status](https://img.shields.io/badge/status-in%20development-orange.svg)](#roadmap)

## Table of Contents

- [Overview](#overview)
- [Why Fine-tuning?](#why-fine-tuning)
- [Approach](#approach)
- [Tech Stack](#tech-stack)
- [Hardware Requirements](#hardware-requirements)
- [Evaluation](#evaluation)
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
| **Base model** | Qwen2.5-Coder-7B-Instruct | Strong code/SQL capabilities; 7B fits comfortably in 12GB VRAM with QLoRA |
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
- 4-bit quantization keeps the 7B model within VRAM limits
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

## Success Criteria

- Execution accuracy of the fine-tuned model shows a clear, measurable improvement over both baselines
- Error analysis breaks down performance by query complexity (simple SELECT, JOIN, subquery, aggregation)
- The full pipeline is reproducible — running the provided scripts on the same dataset yields comparable results

## Roadmap

- [x] Define problem scope and approach
- [x] Data preparation (download Spider, explore schema, build instruction-formatted dataset)
- [x] Baseline evaluation (zero-shot / few-shot)
- [ ] QLoRA fine-tuning setup and training runs
- [ ] Post-tuning evaluation and comparison
- [ ] Error analysis
- [ ] Gradio demo
- [ ] Technical write-up

## Contributing

This project is in early development. Issues and pull requests are welcome once the initial pipeline is in place — check the [Roadmap](#roadmap) for the current stage.

## License

MIT — see `LICENSE` for details.
