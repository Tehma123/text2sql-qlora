# Baseline Evaluation

Status: **done, on the current (3B) base model.** Originally run against
Qwen2.5-Coder-7B-Instruct (both baselines) -- the project has since pivoted
to Qwen2.5-Coder-3B-Instruct (`reports/qlora_training.md`, "Pivot: 7B ->
3B") after a real training run showed 7B doesn't fit the 12GB target GPU as
comfortably as a single-step VRAM check suggested. 7B's baselines no longer
apply as the comparison floor for fine-tuning; both 3B baselines are now
measured below.

**Re-run on corrected prompts (2026-10-09):** both 3B baselines were
measured again after the composite-primary-key fix to the schema prompts
(`reports/data_preparation.md`, "Correction (2026-10-07)"). The results
barely move -- see "Re-run on corrected primary keys" at the end.

## Summary

| Baseline | Model | exact_match | execution_accuracy | wall-clock |
|---|---|---|---|---|
| Zero-shot | 7B (superseded) | 33.7% | 75.9% | ~84 min (~2.3s/example) |
| Few-shot (5 examples) | 7B (superseded) | 50.5% | 80.2% | ~97 min (~2.7s/example) |
| **Zero-shot** | **3B (current)** | **26.9%** | **66.3%** | **~58 min (~1.6s/example)** |
| **Few-shot (5 examples)** | **3B (current)** | **50.9%** | **74.3%** | **~53 min (~1.5s/example)** |

3B zero-shot lands meaningfully below 7B's: -6.8 points exact_match, -9.6
points execution_accuracy. Consistent with what was flagged as a real risk
when the pivot was decided -- smaller models are generally weaker at exactly
what Spider's fully cross-domain test set stresses (generalizing to 40
database schemas never seen in training).

**But few-shot narrows that gap substantially, and closes it entirely on
exact_match:** 3B few-shot (50.9%) is actually marginally *above* 7B
few-shot (50.5%), while the execution_accuracy gap shrinks from -9.6 points
(zero-shot) to -5.9 points (74.3% vs 80.2%). This tracks with a hypothesis
formed while the run was still in progress: few-shot's main measured effect
(both here and in the original 7B run below) is teaching the model the
dataset's SQL *formatting conventions* via demonstration -- a
pattern-copying skill -- rather than deeper schema reasoning. A smaller
model can apparently copy a demonstrated format just as well as a larger
one, even where it lags on zero-shot schema understanding. 3B is also faster
per example throughout (~1.6s zero-shot, ~1.5s few-shot vs 7B's ~2.3s/~2.7s),
though not proportionally to its ~43% parameter share, since generation
latency is more bandwidth-bound than compute-bound.

**Early running numbers during this eval were misleadingly high and are not
representative** -- both 3B runs processed examples in database order, so
the first ~15-20% of progress reflected only 1-2 (easier) databases; few-shot
peaked at 63.1%/83.6% around n=450 before settling down to the final
50.9%/74.3% as harder databases entered the average. Worth remembering next
time an early partial read looks unexpectedly good.

Measures where the un-finetuned base model stands, so the value of QLoRA
fine-tuning (next roadmap phase) is demonstrated against real numbers rather
than assumed (README: "Why Fine-tuning?"). Two baselines per
`configs/eval_config.yaml`: zero-shot (question + schema only) and few-shot
(5 worked examples in the prompt), both run against the full 2,147-example
`test` split.

## Setup

- Model: `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit` (current; was
  `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` before the pivot -- same
  weights `train_config.yaml`'s `base_model` resolves to either way), loaded
  via Unsloth's `FastLanguageModel` for inference -- no LoRA applied, this is
  the stock pretrained model.
- Metrics (`src/text2sql/eval/metrics.py`):
  - **Exact match**: both queries parsed and canonicalized with `sqlglot`
    (dialect `sqlite`) before comparing, so whitespace/casing/quoting
    differences don't count as mismatches -- an unparseable prediction
    counts as a mismatch rather than crashing the eval loop.
  - **Execution accuracy**: both queries run read-only (`file:...?mode=ro`)
    against the real per-database SQLite file, with a progress-handler-based
    timeout (5s) to bound runaway queries (e.g. an accidental cartesian
    product), and compared as an order-insensitive set of result rows.
- 11 unit tests in `tests/test_metrics.py` cover both metrics against a toy
  in-memory-equivalent SQLite fixture, including that write attempts
  (`DELETE FROM ...`) are correctly rejected rather than silently permitted.

## Zero-shot results

### 3B (current)

| n | exact_match | execution_accuracy | wall-clock time |
|---|---|---|---|
| 2,147 | 26.9% | 66.3% | ~58 min (incl. model load) on the RTX 3060 12GB, ~1.6s/example |

Pace was checked throughout the run (not just at the end) specifically
because the prior 7B *training* run had shown a rate that degraded badly
partway through (see `reports/qlora_training.md`) -- this zero-shot
*inference* run showed no such trend: ~1.2-2.4s/example fluctuation
throughout with no worsening pattern, consistent with inference not touching
the CPU-offloaded gradient-checkpointing path that caused the training
slowdown.

### 7B (superseded)

| n | exact_match | execution_accuracy | wall-clock time |
|---|---|---|---|
| 2,147 | 33.7% | 75.9% | ~84 min (incl. model load) on the RTX 3060 12GB, ~2.3s/example |

Execution accuracy well above exact match is expected in both cases -- many
syntactically-different queries return identical results (e.g. `count(*)` vs
`COUNT(*)`, or reordered column lists), which is exactly why the README
treats execution accuracy as "more meaningful."

## Bug found and fixed: few-shot prompts were being silently truncated

(Found during the original 7B run; the fix is in shared, model-agnostic code
and still applies going forward.)

The first few-shot run was killed partway through after its log filled with:

```
Unsloth: Input IDs of shape torch.Size([1, 3478]) with length 3478 > the model's max sequence length of 1536.
We shall truncate it ourselves.
```

Cause: `build_few_shot_messages()` in `src/text2sql/prompts/templates.py`
called `build_user_message(question, schema)` -- which embeds the *full*
schema -- for every demonstration turn AND the real question. With
`num_examples: 5`, that's the schema repeated 6 times in one prompt, pushing
the largest schemas well past `max_seq_length: 1536` (mean ~1,800+, up to
~3,540 tokens observed) even though the schema is identical throughout a
single conversation and only needs stating once.

Fix: state the schema once, in the first turn (whichever turn is first --
the first demonstration, or the real question if there are no
demonstrations); every subsequent turn is just `### Question: ... ### SQL:`.
Verified on 200 test examples before rerunning:

| | mean tokens | max tokens | over 1536 |
|---|---|---|---|
| zero-shot | 386 | 588 | 0 |
| few-shot (buggy) | ~1,800+ | ~3,540 | most of them |
| few-shot (fixed) | 759 | 1,203 | 0 |

Also added a safety net in `generate_sql()` (`src/text2sql/eval/baseline_eval.py`):
explicit `truncation=True, max_length=max_seq_length` with
`tokenizer.truncation_side = "left"`, so if a prompt is ever over budget
again (e.g. one unusually large schema), it truncates from the left --
preserving the real question and generation prompt at the end, rather than
the default right-truncation, which would cut off the part of the prompt
that actually asks the question.

Only the few-shot eval path was affected -- `build_training_messages()`
(used by `preprocess.py` for the actual training data) only ever embeds the
schema once, so nothing in `data/processed/` needed to change.

## Few-shot demonstration sampling

`test.json`'s 40 databases are entirely disjoint from train/dev (see
`reports/data_preparation.md`), so there is no train example sharing a
schema with any test question to draw a few-shot demonstration from.
Demonstrations are instead sampled from *other test questions on the same
database* (`_sample_few_shot_pool` + `select_demos` in `baseline_eval.py`,
seeded) -- showing a few worked examples on an unfamiliar schema before
asking the real question, which is standard practice for cross-domain
few-shot prompting.

**Known leak (found 2026-10-09):** the current question was excluded by
exact question text only. Spider usually asks each gold query in two
paraphrased forms, so for 164 of 2,147 test examples (7.6%) a demo is a
paraphrase carrying the same gold SQL (equal after `normalize_sql`; 160 by
raw string). Accuracy on those examples is ~90%, which lifts every few-shot
number by about 1-1.5pt. On the 1,983 clean examples, the 3B few-shot
baseline scores 72.9% (old prompts) and 72.5% (corrected prompts). The
exclusion is now fixed in code (`select_demos` also skips demos with the
target's gold SQL), but the few-shot runs were not repeated before the
project closed, so every few-shot number in this report is from the old
exclusion. Details in `reports/qlora_training.md` ("Few-shot caveat").

## Few-shot results

### 3B (current)

| n | exact_match | execution_accuracy | wall-clock time |
|---|---|---|---|
| 2,147 | 50.9% | 74.3% | ~53 min (incl. model load) on the RTX 3060 12GB, ~1.5s/example |

Both metrics improved over 3B's own zero-shot, and by a wide margin on exact
match (26.9% -> 50.9%, +24 points -- an even bigger lift than 7B saw). See
the Summary section above for the full 3B-vs-7B comparison and the
"formatting convention" hypothesis for why few-shot closes the gap.

### 7B (superseded)

| n | exact_match | execution_accuracy | wall-clock time |
|---|---|---|---|
| 2,147 | 50.5% | 80.2% | ~97 min (incl. model load) on the RTX 3060 12GB, ~2.7s/example |

Both metrics improved over zero-shot, as expected -- most notably exact
match (33.7% -> 50.5%), since seeing a couple of worked examples on an
unfamiliar schema helps the model match the dataset's SQL formatting
conventions (e.g. spacing around commas, `count(*)` vs `COUNT(*)`) even when
it already had enough schema understanding to get the *result* right
zero-shot. Full per-example predictions for the current model are in
`results/baselines/{zero_shot,few_shot}.json` (3B).

## Re-run on corrected primary keys

Both 3B baselines re-run on 2026-10-09 against the corrected schema prompts,
same model, same 2,147 test questions and same seeded few-shot demos
(`python scripts/run_baseline_eval.py --output-dir results/baselines/v2`,
~57 min zero-shot + ~52 min few-shot including model load):

| Baseline | prompts | exact_match | execution_accuracy |
|---|---|---|---|
| Zero-shot | old | 26.9% | 66.3% |
| Zero-shot | corrected | 27.2% | 66.6% |
| Few-shot (5 examples) | old | 50.9% | 74.3% |
| Few-shot (5 examples) | corrected | 50.3% | 73.9% |

Neither change is significant (exact McNemar on execution_accuracy:
zero-shot 28 vs. 22 examples that only one prompt version gets right,
p=0.48; few-shot 3 vs. 11, p=0.057), and in both cases the changed examples
are all on schemas with a composite-key table. The primary-key fix has no
measurable effect on the untuned model. These corrected-prompt numbers are
the right comparison for the v2 fine-tuned model, which was trained and
tested on the same prompts.
