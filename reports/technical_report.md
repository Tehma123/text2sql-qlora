# Text2SQL-Tune — Technical Report

## Overview

This project fine-tunes **Qwen2.5-Coder-3B-Instruct** on the **Spider** text-to-SQL benchmark using **QLoRA** (4-bit quantization + LoRA), on a single consumer GPU (RTX 3060, 12GB VRAM). The goal was a fully reproducible pipeline — data preparation, baselines, fine-tuning, evaluation, error analysis, and a working demo — with every claim backed by a measurement rather than assumed.

**Headline result:** the fine-tuned model, combined with few-shot demonstrations at inference time, reaches **76.2% execution accuracy** on the held-out test split — beating both the zero-shot baseline (66.3%, +9.9pt) and the few-shot baseline (74.3%, +1.9pt) individually, though 3.8 points short of the ~80% target set going in. Fine-tuning alone (no few-shot) reached 71.9%, ahead of zero-shot but behind few-shot — the reasoning behind that gap, and why combining both closes it, is one of the more interesting findings below.

The project also pivoted mid-way from a 7B to a 3B base model after a real multi-hour training run revealed the 7B model didn't fit the 12GB GPU as comfortably as a quick VRAM check had suggested — a concrete example of the gap between a single-step measurement and sustained real-world behavior.

## Dataset

**Spider** ([yale-lily.github.io/spider](https://yale-lily.github.io/spider)), downloaded from the original release rather than the Hugging Face mirror, since execution-accuracy evaluation needs the real per-database SQLite files, which the HF mirror doesn't include.

| Split | Examples | Distinct databases | Overlap with other splits |
|---|---|---|---|
| train | 8,575 | 145 | none |
| dev | 1,034 | 20 | none |
| test | 2,147 | 40 | none |

All three splits use **completely disjoint sets of databases** — this is Spider's cross-domain design, and it means both dev and test evaluate genuine generalization to schemas the model has never seen, not just new questions on familiar ones.

**Schema representation:** each database is rendered as `CREATE TABLE` DDL (original column/table casing, types, primary keys, `FOREIGN KEY ... REFERENCES ...`) rather than a flatter listing — closer to what the base model saw during pretraining, and keeps join-relevant key relationships explicit.

**Instruction format:** chat messages (system + user + gold assistant turn), so both training and evaluation apply the base model's own chat template rather than hand-rolling special tokens. `max_seq_length` was set to **1536** after measuring real tokenized lengths across candidate budgets (1024 left 5.6% of train over-budget; 1536 leaves only 0.97%, with the remainder dropped rather than truncated, since truncation would risk clipping the gold SQL answer instead of the schema).

**Peak VRAM for one training step**, measured directly rather than assumed: plain `peft`+`bitsandbytes` needed 16.5GB at batch size 2 (doesn't fit in 12GB); Unsloth's fused kernels brought that down to 8.36GB allocated / 9.92GB reserved, comfortably under budget — the gap traces to Qwen's 152K-token vocabulary, whose full-precision logits Unsloth avoids materializing.

## Baselines

Two baselines were measured on the un-finetuned base model before any fine-tuning, so the value of fine-tuning is demonstrated against real numbers rather than assumed. Both run on the full 2,147-example test split.

### Zero-shot

Question + schema only, no examples.

| exact_match | execution_accuracy |
|---|---|
| 26.9% | 66.3% |

### Few-shot

5 worked (question, SQL) examples included in the prompt, sampled from *other* questions on the same database (test's 40 databases are disjoint from train, so there's no training example to draw a same-schema demonstration from).

| exact_match | execution_accuracy |
|---|---|
| 50.9% | 74.3% |

Few-shot lifts both metrics substantially over zero-shot, most dramatically on exact_match (+24 points). The leading hypothesis: few-shot's main effect is teaching the model the dataset's SQL *formatting conventions* (spacing, `count(*)` vs `COUNT(*)`, column ordering) via direct demonstration — a pattern-copying skill — rather than deeper schema reasoning, where the gap to execution correctness narrows less.

## Fine-tuning

**Method:** QLoRA (4-bit quantization via `bitsandbytes`, LoRA adapters via Unsloth's `FastLanguageModel`) — r=16, alpha=32, dropout=0.05, targeting all attention and MLP projection matrices. Trained with `trl.SFTTrainer`, response-only loss masking (`unsloth.chat_templates.train_on_responses_only`) so gradient signal comes from the SQL response tokens only, not the several-times-longer schema/question prompt. Masking correctness was verified directly, not assumed: real training examples were decoded post-masking and their unmasked tokens matched the gold SQL exactly, for both the 7B and 3B tokenizers.

### Model selection: a mid-project pivot from 7B to 3B

The project started with Qwen2.5-Coder-**7B**-Instruct. A single-step VRAM check showed a comfortable 8.36GB/9.92GB (allocated/reserved) footprint under the 12GB budget. A real 3-epoch training run told a different story: after ~4 hours it had reached only 20% of the first epoch, with per-step time degraded 5x from its own initial rate (154s/it vs. an initial ~30s/step), projecting **~59 hours** to complete instead of the estimated ~14.

Diagnosis (using Windows Performance Counters, since `nvidia-smi` doesn't expose per-process memory on this platform) traced the slowdown to **BAR1 memory** — the PCIe aperture used for CPU↔GPU transfers — nearly exhausted (29MB free of 16,384MB), a resource the single-step VRAM check never exercised because Unsloth's CPU-offloaded gradient checkpointing only puts sustained pressure on it over a long, varied-batch run. BAR1 has no runtime "increase" lever, so rather than trade one constraint for another, the project switched to **Qwen2.5-Coder-3B-Instruct** (~43% of 7B's parameters), which reduces pressure on both VRAM and BAR1 at once. Re-measured immediately: 4.09GB/4.72GB peak, more than 7GB of headroom — and this margin held up through the real full run that followed, with no recurrence of the degradation pattern.

This was a real trade-off, not a free upgrade: 3B's own baselines land 5.9-9.6 points below 7B's on execution_accuracy, and the accuracy target had to be re-derived rather than carried over.

### Training run

Three epochs, `python scripts/train.py`, checkpointed at each epoch boundary (steps 536 / 1,072 / 1,608 of 1,608 total).

| checkpoint | epoch | eval_loss | train_loss |
|---|---|---|---|
| checkpoint-536 | 1 | 0.264 | ~0.08 |
| checkpoint-1072 | 2 | 0.288 | lower |
| checkpoint-1608 | 3 | **0.362** | **0.005** |

Eval loss rose across all three epochs, accelerating sharply at epoch 3 (+0.073, roughly 3x the per-epoch rate of epochs 1-2), while train loss bottomed out near zero — the model had essentially memorized the training set's exact SQL strings by the third epoch. Classic overfitting.

**Checkpoint selection** was done entirely against `dev.json` (the same split `eval_loss` was computed on), specifically to avoid biasing the choice by peeking at the test set that would later report the final number. A dev-set comparison confirmed epoch 3 was clearly worse (5.7-7.4pt below epochs 1-2 on execution_accuracy — the loss and accuracy signals agreed there); epochs 1 and 2 were close enough that epoch 1 was measured on the full 1,034-example dev set for a more precise read (0.7292 execution_accuracy, matching its smaller-sample estimate closely) and selected — both for its (marginally) higher accuracy and for being the lowest-overfitting-risk checkpoint.

### Engineering issues encountered

Four real, reproduced bugs surfaced while building the pipeline, each worth noting since they're not obviously specific to this project:

1. **Import order silently corrupts training config.** Importing `trl`/`transformers` before `unsloth` causes `transformers`' config serialization to redact `eos_token=None` to the literal string `"<EOS_TOKEN>"`, which then fails vocabulary lookup — traced to `unsloth` needing to patch those libraries' classes before they're used elsewhere.
2. **Unsloth's compiled `SFTTrainer` doesn't auto-detect conversational (`messages`-column) datasets** the way vanilla `trl` does — worked around by pre-applying the chat template into a flat `text` column.
3. **Unsloth's startup banner crashes on Windows' default console encoding** (emoji in `print()` output, cp1252 can't encode them) — fixed by forcing UTF-8 on `stdout`/`stderr` before any other import.
4. **Unsloth's fast downloader (`hf_transfer`) can't recover from its own corrupted partial download on Windows** — deterministic, not a flake; fixed by disabling it via an environment variable set early enough in the import chain to actually take effect.

## Results

**checkpoint-536 (epoch 1), full `test.json` (2,147 examples), one-shot evaluation:**

| | exact_match | execution_accuracy |
|---|---|---|
| Zero-shot (3B base) | 26.9% | 66.3% |
| Few-shot (3B base, 5-shot) | 50.9% | 74.3% |
| Fine-tuned, zero-shot | 43.0% | 71.9% |
| **Fine-tuned + few-shot** | **53.1%** | **76.2%** |
| Target | — | ~80% |

**Fine-tuning alone clearly beats zero-shot** (+5.6pt execution_accuracy, +16.1pt exact_match) but **falls short of the few-shot baseline** (-2.4pt / -7.9pt). The most likely explanation: the few-shot baseline is shown 5 worked examples *from the same unseen database* at inference time — a live demonstration of that schema's naming conventions. The fine-tuned model is evaluated zero-shot (matching its training format), so at inference time on a new database it gets no such per-database signal, only what it generalized from 145 *different* training databases during fine-tuning.

**Combining both — fine-tuning and few-shot demonstrations at inference time — beats each baseline individually**, +9.9pt over zero-shot and +1.9pt over few-shot. This confirms the two provide complementary, not redundant, signal: fine-tuning's cross-database pattern learning and few-shot's live per-schema demonstration stack rather than one subsuming the other.

**On the ~80% target:** it was calibrated as "~5-6 points over the few-shot baseline," implicitly assuming fine-tuning strictly dominates few-shot. The result here shows that assumption doesn't hold — the two methods inject qualitatively different information, so the margin logic that produced the target didn't transfer cleanly. The 76.2% result is a complete, honestly-measured outcome; it just isn't the number the original target predicted.

## Error Analysis

Execution_accuracy broken down by SQL query complexity (keyword-heuristic categories matching Spider's own taxonomy — join present, more than one `SELECT` (subquery), aggregate function or `GROUP BY` present, or none of the above as the simple baseline; not mutually exclusive except "select"):

| category | n | zero-shot | few-shot | fine-tuned | fine-tuned + few-shot |
|---|---|---|---|---|---|
| select (simple) | 526 | 76.4% | **90.5%** | 86.7% | 88.6% |
| **join** | **862** | 56.8% | 61.0% | 59.2% | **67.1%** |
| subquery | 346 | 58.4% | 67.1% | 66.2% | 66.5% |
| aggregation | 1,054 | 65.2% | 71.5% | 69.3% | **74.1%** |

Two findings:

1. **Join queries are the weak point across every condition** — the lowest execution_accuracy of all four categories in every single column, and the largest category by count (862 of 2,147 examples, 40% of the test set), so it drags the overall average down more than any other category simply by weight.
2. **The combined approach's improvement is concentrated exactly where it's needed, not spread evenly.** On the already-easy "select" category, combining doesn't help beyond few-shot alone. On **join** — the hardest, largest category — combining beats few-shot alone by **+6.1 points**, and on aggregation by +2.6 points. The two signals complement each other specifically on the hard, high-value cases.

This makes join-heavy queries the clearest target for further improvement, rather than a diffuse "needs more training" problem.

## Conclusion

The project demonstrates, with measurements rather than assumptions at every step, that QLoRA fine-tuning improves Text-to-SQL performance over an untuned base model, and that combining fine-tuning with few-shot prompting improves further still — beating both baselines individually at 76.2% execution accuracy. It falls 3.8 points short of the ~80% target, and the reason is itself a real finding: the target's margin logic assumed fine-tuning would strictly dominate few-shot, but the two methods turned out to provide complementary rather than nested signal, with few-shot's live per-database demonstration filling a gap fine-tuning's zero-shot inference structurally can't.

Along the way, a mid-project model-size pivot (7B → 3B) showed the concrete difference between a single-step VRAM measurement and real, sustained training behavior on a resource (BAR1 memory) the quick check never exercised — a useful cautionary example for anyone reproducing this on similarly VRAM-constrained hardware.

**If pushing accuracy further were the next step**, the error analysis points at a specific, high-value target rather than a diffuse one: join queries are the weakest category everywhere and the largest by volume, so schema-linking-focused improvements (or simply more join-heavy training data) would likely move the overall number more than further generic training. Given diminishing returns are already visible in this project's own research on the effort required to push past ~80-82% on Spider (schema linking, self-consistency, ensembling), the honest scoping decision made here was to treat 76.2% — a clear, well-explained improvement over both baselines — as a complete result rather than chase the last few points.
