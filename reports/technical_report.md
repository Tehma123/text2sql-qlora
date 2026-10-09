# Text2SQL-Tune — Technical Report

## Overview

This project fine-tunes **Qwen2.5-Coder-3B-Instruct** on the **Spider** text-to-SQL benchmark using **QLoRA** (4-bit quantization + LoRA), on a single consumer GPU (RTX 3060, 12GB VRAM). The goal was a fully reproducible pipeline — data preparation, baselines, fine-tuning, evaluation, error analysis, and a working demo — with every claim backed by a measurement rather than assumed.

**Headline result:** the fine-tuned model, combined with few-shot demonstrations at inference time, reaches **77.4% execution accuracy** on the held-out test split — beating both the zero-shot baseline (66.6%, +10.8pt) and the few-shot baseline (73.9%, +3.5pt) measured on the same prompts. That meets the project's 75% target, which was lowered from ~80% after the results were known; against the original ~80% it is 2.6 points short. Fine-tuning alone (no few-shot) reached 71.7%, ahead of zero-shot but not ahead of few-shot — why, and why combining both beats each, is one of the more interesting findings below.

These are the numbers for the **v2** model. The first model (v1) reached 76.2% / 71.9%; after it was finished, the schema prompts turned out to carry wrong primary keys for tables with composite keys, so the model was retrained on corrected prompts with finer-grained checkpoint selection (see "Revision: primary-key fix and retraining (v2)"). v2 is not significantly better than v1, but it is the best measured result. All few-shot numbers in this report run about 1-1.5pt high because of a demo-selection leak that affects every few-shot condition about equally (see "Few-shot caveat" under Results). The project was closed on 2026-10-09 after an acceptance review that recomputed every reported number from the per-example results.

The project also pivoted mid-way from a 7B to a 3B base model after a real multi-hour training run revealed the 7B model didn't fit the 12GB GPU as comfortably as a quick VRAM check had suggested — a concrete example of the gap between a single-step measurement and sustained real-world behavior.

## Dataset

**Spider** ([yale-lily.github.io/spider](https://yale-lily.github.io/spider)), downloaded from the original release rather than the Hugging Face mirror, since execution-accuracy evaluation needs the real per-database SQLite files, which the HF mirror doesn't include.

| Split | Examples | Distinct databases | Overlap with other splits |
|---|---|---|---|
| train | 8,575 | 145 | none |
| dev | 1,034 | 20 | none |
| test | 2,147 | 40 | none |

All three splits use **completely disjoint sets of databases** — this is Spider's cross-domain design, and it means both dev and test evaluate genuine generalization to schemas the model has never seen, not just new questions on familiar ones.

**Schema representation:** each database is rendered as `CREATE TABLE` DDL (original column/table casing, types, primary keys, `FOREIGN KEY ... REFERENCES ...`) rather than a flatter listing — closer to what the base model saw during pretraining, and keeps join-relevant key relationships explicit. Primary keys are read from the SQLite files, not from Spider's `tables.json`: `tables.json` keeps only the first column of a composite key, which had rendered 132 of 876 train/dev tables with a wrong single-column `PRIMARY KEY` in v1's prompts (53% of train examples). Composite keys are now a table-level `PRIMARY KEY (a, b)` clause.

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

Both baselines were measured again after the primary-key fix to the schema prompts: **66.6%** zero-shot and **73.9%** few-shot execution accuracy (exact_match 27.2% / 50.3%). Neither change is significant, and every example that changed is on a schema with a composite-key table — the fix doesn't measurably help the untuned model. The corrected-prompt numbers are the ones the v2 model is compared against.

Few-shot lifts both metrics substantially over zero-shot, most dramatically on exact_match (+24 points). The leading hypothesis: few-shot's main effect is teaching the model the dataset's SQL *formatting conventions* (spacing, `count(*)` vs `COUNT(*)`, column ordering) via direct demonstration — a pattern-copying skill — rather than deeper schema reasoning, where the gap to execution correctness narrows less.

## Fine-tuning

**Method:** QLoRA (4-bit quantization via `bitsandbytes`, LoRA adapters via Unsloth's `FastLanguageModel`) — r=16, alpha=32, dropout=0.05, targeting all attention and MLP projection matrices. Trained with `trl.SFTTrainer`, response-only loss masking (`unsloth.chat_templates.train_on_responses_only`) so gradient signal comes from the SQL response tokens only, not the several-times-longer schema/question prompt. Masking correctness was verified directly, not assumed: real training examples were decoded post-masking and their unmasked tokens matched the gold SQL exactly, for both the 7B and 3B tokenizers.

### Model selection: a mid-project pivot from 7B to 3B

The project started with Qwen2.5-Coder-**7B**-Instruct. A single-step VRAM check showed a comfortable 8.36GB/9.92GB (allocated/reserved) footprint under the 12GB budget. A real 3-epoch training run told a different story: after ~4 hours it had reached only 20% of the first epoch, with per-step time degraded 5x from its own initial rate (154s/it vs. an initial ~30s/step), projecting **~59 hours** to complete instead of the estimated ~14.

Diagnosis (using Windows Performance Counters, since `nvidia-smi` doesn't expose per-process memory on this platform) traced the slowdown to **BAR1 memory** — the PCIe aperture used for CPU↔GPU transfers — nearly exhausted (29MB free of 16,384MB), a resource the single-step VRAM check never exercised because Unsloth's CPU-offloaded gradient checkpointing only puts sustained pressure on it over a long, varied-batch run. BAR1 has no runtime "increase" lever, so rather than trade one constraint for another, the project switched to **Qwen2.5-Coder-3B-Instruct** (~43% of 7B's parameters), which reduces pressure on both VRAM and BAR1 at once. Re-measured immediately: 4.09GB/4.72GB peak, more than 7GB of headroom — and this margin held up through the real full run that followed, with no recurrence of the degradation pattern.

This was a real trade-off, not a free upgrade: 3B's own baselines land 5.9-9.6 points below 7B's on execution_accuracy, and the accuracy target had to be re-derived rather than carried over.

### Training run (v1)

Three epochs, `python scripts/train.py`, checkpointed at each epoch boundary (steps 536 / 1,072 / 1,608 of 1,608 total).

| checkpoint | epoch | eval_loss | train_loss |
|---|---|---|---|
| checkpoint-536 | 1 | 0.264 | ~0.08 |
| checkpoint-1072 | 2 | 0.288 | lower |
| checkpoint-1608 | 3 | **0.362** | **0.005** |

Eval loss rose across all three epochs, accelerating sharply at epoch 3 (+0.073, roughly 3x the per-epoch rate of epochs 1-2), while train loss bottomed out near zero — the model had essentially memorized the training set's exact SQL strings by the third epoch. Classic overfitting.

**Checkpoint selection** was done entirely against `dev.json` (the same split `eval_loss` was computed on), specifically to avoid biasing the choice by peeking at the test set that would later report the final number. A dev-set comparison confirmed epoch 3 was clearly worse (5.7-7.4pt below epochs 1-2 on execution_accuracy — the loss and accuracy signals agreed there); epochs 1 and 2 were close enough that epoch 1 was measured on the full 1,034-example dev set for a more precise read (0.7292 execution_accuracy, matching its smaller-sample estimate closely) and selected — both for its (marginally) higher accuracy and for being the lowest-overfitting-risk checkpoint.

### Revision: primary-key fix and retraining (v2)

Two problems surfaced after v1 was finished: the primary-key bug described under "Dataset", and the per-epoch eval grid being too coarse — v1's eval_loss was already lowest at the first of its three points, so the best checkpoint could have been anywhere before step 1,072.

v2 retrained on the corrected prompts with the same hyperparameters, except evaluation and checkpointing every 100 steps and 2 epochs instead of 3 (epoch 3 had cost v1 7.4pt of dev accuracy). Dev execution_accuracy (n=1,034) per checkpoint:

| step | 100 | 200 | 300 | 500 | **600** | 700 | 800 | 900 |
|---|---|---|---|---|---|---|---|---|
| eval_loss | 0.232 | **0.226** | 0.242 | 0.269 | 0.285 | 0.294 | 0.302 | 0.308 |
| execution_accuracy | 70.5% | 71.8% | 71.7% | 71.9% | **76.4%** | 74.8% | 74.6% | 74.4% |

**eval_loss turned out to be a misleading selection signal.** It bottomed out at step 200 and rose steadily after, while execution accuracy sat at ~72% through step 500 and jumped at step 600 (shortly after epoch 2 began). The first v2 pass used early stopping on eval_loss and stopped at step 600; it was resumed with early stopping off once dev accuracy showed the disagreement. Selecting by eval_loss would have shipped a checkpoint 4.6pt worse on dev.

checkpoint-600 was selected (76.4% dev vs. 73.4% for v1's checkpoint-536 on the same prompts). That dev advantage mostly did not survive to test: checkpoint-600 was the best of eight dev evaluations, and the dev set's 20 databases make per-example significance tests overconfident, since questions on the same database don't succeed or fail independently. On test, v2 matches v1 zero-shot (71.7% vs. 71.9%, p=0.84) and is 1.2pt ahead with few-shot (77.4% vs. 76.2%, p=0.061). The primary-key fix on its own is a small effect — v1 gains only 0.5pt on dev from the corrected prompts, the base model moves +0.3pt zero-shot and -0.4pt few-shot on test (neither significant), and v2's few-shot gain on test comes from questions on schemas with no composite-key table.

### Engineering issues encountered

Four real, reproduced bugs surfaced while building the pipeline, each worth noting since they're not obviously specific to this project:

1. **Import order silently corrupts training config.** Importing `trl`/`transformers` before `unsloth` causes `transformers`' config serialization to redact `eos_token=None` to the literal string `"<EOS_TOKEN>"`, which then fails vocabulary lookup — traced to `unsloth` needing to patch those libraries' classes before they're used elsewhere.
2. **Unsloth's compiled `SFTTrainer` doesn't auto-detect conversational (`messages`-column) datasets** the way vanilla `trl` does — worked around by pre-applying the chat template into a flat `text` column.
3. **Unsloth's startup banner crashes on Windows' default console encoding** (emoji in `print()` output, cp1252 can't encode them) — fixed by forcing UTF-8 on `stdout`/`stderr` before any other import.
4. **Unsloth's fast downloader (`hf_transfer`) can't recover from its own corrupted partial download on Windows** — deterministic, not a flake; fixed by disabling it via an environment variable set early enough in the import chain to actually take effect.

## Results

**Full `test.json` (2,147 examples), each model evaluated once per mode:**

| | prompts | exact_match | execution_accuracy |
|---|---|---|---|
| Zero-shot (3B base) | original | 26.9% | 66.3% |
| Zero-shot (3B base) | corrected | 27.2% | 66.6% |
| Few-shot (3B base, 5-shot) | original | 50.9% | 74.3% |
| Few-shot (3B base, 5-shot) | corrected | 50.3% | 73.9% |
| v1 fine-tuned (checkpoint-536), zero-shot | original | 43.0% | 71.9% |
| v1 fine-tuned + few-shot | original | 53.1% | 76.2% |
| v2 fine-tuned (checkpoint-600), zero-shot | corrected | 44.4% | 71.7% |
| **v2 fine-tuned + few-shot** | corrected | **54.0%** | **77.4%** |
| Target (lowered from ~80% after the results were known) | | — | 75% |

The comparisons below pair v2 with the baselines measured on the same corrected prompts.

**Fine-tuning alone clearly beats zero-shot** (v2: +5.1pt execution_accuracy, +17.2pt exact_match, p=8e-8) but **does not beat the few-shot baseline**. On all 2,147 examples it trails by 2.2pt, but the demo leak described below lifts only the few-shot conditions, so it biases exactly this comparison. On the 1,983 leak-free examples the gap is 0.6pt (71.9% vs. 72.5%, 176 vs. 188, p=0.56; 19 databases better, 16 worse), i.e. no measurable difference; v1 alone vs. the few-shot baseline is likewise -1.2pt (p=0.24). Why fine-tuning alone doesn't pull ahead: the few-shot baseline is shown 5 worked examples *from the same unseen database* at inference time — a live demonstration of that schema's naming conventions. The fine-tuned model is evaluated zero-shot (matching its training format), so at inference time on a new database it gets no such per-database signal, only what it generalized from 145 *different* training databases during fine-tuning.

**Combining both — fine-tuning and few-shot demonstrations at inference time — beats each baseline individually**: v2 is +10.8pt over zero-shot and +3.5pt over few-shot (p=4.4e-5), and few-shot adds 5.7pt on top of v2's own zero-shot result. This confirms the two provide complementary, not redundant, signal: fine-tuning's cross-database pattern learning and few-shot's live per-schema demonstration stack rather than one subsuming the other.

**On the target:** the target going in was ~80%, calibrated as "~5-6 points over the few-shot baseline," implicitly assuming fine-tuning strictly dominates few-shot. The result here shows that assumption doesn't hold — the two methods inject qualitatively different information, so the margin logic that produced the target didn't transfer cleanly. On 2026-10-09, with all test results known, the target was lowered to **75%**. 77.4% meets it (76.4% on the leak-free examples), but the change was made after seeing the results, and 75% sits only 1.1pt above the few-shot baseline; the evidence that fine-tuning helps is the significant gain over both baselines, not clearing 75%. The 95% database-level bootstrap interval for 77.4% is 73.1-82.0%, so the result is not statistically distinguishable from either 75% or 80%.

**Few-shot caveat.** Few-shot demonstrations are other test questions on the same database, and the target question was excluded only when its text matched exactly. Spider usually asks each gold query in two paraphrased forms, so for 164 of the 2,147 test examples (7.6%) one demonstration carries the gold SQL (164 counts SQL equal after `normalize_sql`, the sqlglot canonicalization exact_match uses; 160 by raw string), and accuracy on those examples is ~90% in every few-shot condition. This inflates every few-shot number by about 1-1.5pt, roughly equally across few-shot conditions, but not the zero-shot ones. On the 1,983 clean examples: few-shot baseline 72.5%, v1 + few-shot 75.1%, **v2 + few-shot 76.4%**. v2's lead over the few-shot baseline holds there (+3.9pt, p=1.4e-5). The exclusion is now fixed in code (`select_demos` also drops demos with the target's gold SQL, which changes the demos of exactly those 164 examples), but the few-shot evaluations were not re-run before the project was closed, so the numbers above are as measured.

## Error Analysis

Execution_accuracy broken down by SQL query complexity (keyword-heuristic categories matching Spider's own taxonomy — join present, more than one `SELECT` (subquery), aggregate function or `GROUP BY` present, or none of the above as the simple baseline; not mutually exclusive except "select"). This analysis was done on v1, with the baseline columns from the original prompts; the last column adds v2's combined result:

| category | n | zero-shot | few-shot | fine-tuned (v1) | fine-tuned + few-shot (v1) | fine-tuned + few-shot (v2) |
|---|---|---|---|---|---|---|
| select (simple) | 526 | 76.4% | **90.5%** | 86.7% | 88.6% | 89.5% |
| **join** | **862** | 56.8% | 61.0% | 59.2% | 67.1% | **68.0%** |
| subquery | 346 | 58.4% | 67.1% | 66.2% | 66.5% | **70.5%** |
| aggregation | 1,054 | 65.2% | 71.5% | 69.3% | 74.1% | **74.4%** |

Two findings:

1. **Join queries are the weak point across every condition** — the lowest execution_accuracy of all four categories in every single column, and the largest category by count (862 of 2,147 examples, 40% of the test set), so it drags the overall average down more than any other category simply by weight.
2. **The combined approach's improvement is concentrated exactly where it's needed, not spread evenly.** On the already-easy "select" category, combining doesn't help beyond few-shot alone. On **join** — the hardest, largest category — combining beats few-shot alone by **+6.1 points**, and on aggregation by +2.6 points. The two signals complement each other specifically on the hard, high-value cases.

This makes join-heavy queries the clearest target for further improvement, rather than a diffuse "needs more training" problem. v2 doesn't change the picture: join is still its weakest category (68.0%), and its largest gain over v1 is on subqueries (+4.0pt).

## Conclusion

The project demonstrates, with measurements rather than assumptions at every step, that QLoRA fine-tuning improves Text-to-SQL performance over an untuned base model, and that combining fine-tuning with few-shot prompting improves further still — beating both baselines individually at 77.4% execution accuracy. That meets the 75% target the project closed with, and falls 2.6 points short of the ~80% target it started with. The reason for the shortfall against ~80% is itself a real finding: that target's margin logic assumed fine-tuning would strictly dominate few-shot, but the two methods turned out to provide complementary rather than nested signal, with few-shot's live per-database demonstration filling a gap fine-tuning's zero-shot inference structurally can't.

Along the way, a mid-project model-size pivot (7B → 3B) showed the concrete difference between a single-step VRAM measurement and real, sustained training behavior on a resource (BAR1 memory) the quick check never exercised — a useful cautionary example for anyone reproducing this on similarly VRAM-constrained hardware.

The v2 revision added two lessons about checkpoint selection. Token-level eval_loss and execution accuracy disagreed badly on this task, so checkpoints should be chosen by execution accuracy. And a dev set with only 20 databases can make a checkpoint look significantly better than it is, especially when it is the best of several candidates: v2's 3-point zero-shot lead over v1 on dev disappeared on test (71.7% vs. 71.9%). It also surfaced an evaluation detail worth copying for anyone building a few-shot setup on Spider: exclude demonstrations by their gold SQL, not just by question text, or paraphrases leak the answer.

**If pushing accuracy further were the next step**, the error analysis points at a specific, high-value target rather than a diffuse one: join queries are the weakest category everywhere and the largest by volume, so schema-linking-focused improvements (or simply more join-heavy training data) would likely move the overall number more than further generic training. Given diminishing returns are already visible in this project's own research on the effort required to push past ~80-82% on Spider (schema linking, self-consistency, ensembling), the honest scoping decision made here was to treat 77.4% — a clear, well-explained improvement over both baselines — as a complete result rather than chase the last few points.
