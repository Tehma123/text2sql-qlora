# Baseline Evaluation

Status: done. Both baselines run on the full 2,147-example `test` split.

## Summary

| Baseline | exact_match | execution_accuracy | wall-clock |
|---|---|---|---|
| Zero-shot | 33.7% | 75.9% | ~84 min (~2.3s/example) |
| Few-shot (5 examples) | **50.5%** | **80.2%** | ~97 min (~2.7s/example) |

Five worked examples lift exact match by ~17 points and execution accuracy
by ~4 points over zero-shot -- a real, measured floor for QLoRA fine-tuning
to beat, not an assumption. Few-shot costs more per example (longer prompts
to prefill) but not proportionally to the accuracy gain.

Measures where the un-finetuned base model stands, so the value of QLoRA
fine-tuning (next roadmap phase) is demonstrated against real numbers rather
than assumed (README: "Why Fine-tuning?"). Two baselines per
`configs/eval_config.yaml`: zero-shot (question + schema only) and few-shot
(5 worked examples in the prompt), both run against the full 2,147-example
`test` split.

## Setup

- Model: `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` (same weights
  `train_config.yaml`'s `base_model` resolves to), loaded via Unsloth's
  `FastLanguageModel` for inference -- no LoRA applied, this is the stock
  pretrained model.
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

| n | exact_match | execution_accuracy | wall-clock time |
|---|---|---|---|
| 2,147 | 33.7% | 75.9% | ~84 min (incl. model load) on the RTX 3060 12GB, ~2.3s/example |

Execution accuracy well above exact match is expected -- many
syntactically-different queries return identical results (e.g. `count(*)` vs
`COUNT(*)`, or reordered column lists), which is exactly why the README
treats execution accuracy as "more meaningful."

## Bug found and fixed: few-shot prompts were being silently truncated

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
database* (`_sample_few_shot_pool` in `baseline_eval.py`, seeded, excludes
the current question) -- showing a few worked examples on an unfamiliar
schema before asking the real question, which is standard practice for
cross-domain few-shot prompting.

## Few-shot results

| n | exact_match | execution_accuracy | wall-clock time |
|---|---|---|---|
| 2,147 | 50.5% | 80.2% | ~97 min (incl. model load) on the RTX 3060 12GB, ~2.7s/example |

Both metrics improved over zero-shot, as expected -- most notably exact
match (33.7% -> 50.5%), since seeing a couple of worked examples on an
unfamiliar schema helps the model match the dataset's SQL formatting
conventions (e.g. spacing around commas, `count(*)` vs `COUNT(*)`) even when
it already had enough schema understanding to get the *result* right
zero-shot. Full per-example predictions for both baselines are in
`results/baselines/{zero_shot,few_shot}.json`.
