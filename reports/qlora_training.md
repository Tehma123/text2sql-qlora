# QLoRA Training

Status: **complete.** The real 3-epoch fine-tune on **Qwen2.5-Coder-3B-Instruct**
finished, a dev-set comparison across all three epoch checkpoints selected
checkpoint-536 (epoch 1; epochs 2-3 overfit), and the final `test.json`
evaluation puts it at **exact_match 43.0% / execution_accuracy 71.9%**
zero-shot -- clearly ahead of the zero-shot baseline (66.3%) but short of
the few-shot baseline (74.3%). Combining fine-tuning with few-shot
demonstrations at inference time (an experiment run afterward) closes that
gap and then some: **exact_match 53.1% / execution_accuracy 76.2%**, beating
both baselines individually, though still 3.8pt short of the ~80% target.
See "Real 3-epoch training run" and "Post-tuning evaluation" below for the
full trace, and "Final result" for the interpretation. This followed an
earlier pivot: the originally-chosen Qwen2.5-Coder-7B-Instruct ran for ~4h
before it became clear it would take ~59h total, not the ~14h estimated from
a smoke test -- traced to VRAM/BAR1 pressure specific to running a 7B model
this close to the 12GB limit (full incident below).

## Pivot: 7B -> 3B

### What happened

A real 3-epoch training run was started on the 7B model (`python
scripts/train.py`, no flags -- the full configured run, per the user's
decision to launch it manually after reviewing the ~14h estimate below). It
ran for ~3h54m and reached step 326/1608 (20%, mid-epoch-1) before being
stopped for investigation. The progress bar's own extrapolated rate had
degraded to **154.05s/it**, versus the ~43s/step the run averaged over its
own first 326 steps, and versus the ~30.4s/step originally measured in a
10-step smoke test. At that rate the full run projected to **~59 hours**,
not ~14.

### Diagnosis

`nvidia-smi` doesn't expose per-process GPU memory on Windows' WDDM driver
model (every process reported `N/A`) -- worked around with Windows'
Performance Counters (`Get-Counter '\GPU Process Memory(*)\Local Usage'`),
which Task Manager itself uses and does report per-process. That showed the
training process alone using **~11.3GB of the 12GB card**, with everything
else running (desktop compositor, browser, Discord, VS Code, etc.) totaling
under 200MB combined -- ruling out background apps as the cause, and
correcting an earlier guess in this investigation that closing a live
wallpaper app would help (it wouldn't have; it wasn't a meaningful
consumer).

A detailed `nvidia-smi -q -d PERFORMANCE,MEMORY` read at that point showed:
- **Not actively thermal-throttled at that instant** (`HW/SW Thermal
  Slowdown`, `HW Power Brake`: all "Not Active"; 57C; only 76.87W of a 170W
  cap; clock at 1942/2115MHz).
- **But real accumulated thermal throttling during the run**: the `SW
  Thermal Slowdown` counter showed ~35.8 minutes accumulated over the ~4h
  run -- consistent with an earlier spot-check that had caught 86C.
- **The more specific finding: BAR1 memory nearly exhausted** -- 16,355 of
  16,384 MiB used (only 29MB free). BAR1 is the PCIe-mapped aperture used
  for CPU<->GPU transfers. Unsloth's `use_gradient_checkpointing="unsloth"`
  (`load_model.py`) offloads checkpointed activations to CPU RAM and copies
  them back during backward-pass recompute -- a transfer path that runs
  through BAR1. Low power draw + high reported "utilization" + a nearly-full
  transfer aperture is the signature of a process waiting on data transfers,
  not one that's compute-bound, which fits: dedicated VRAM at 128MB free
  (99% full) and BAR1 at 29MB free (99.8% full) simultaneously.

Net conclusion: `vram_check.py`'s single-step measurement (8.36GB allocated
/ 9.92GB reserved, "comfortable headroom" under 12GB) held for one isolated
step but **underpredicted what a real, hours-long, varied-batch run actually
needed** -- both on dedicated VRAM and, more specifically, on BAR1. Neither
resource has a runtime "increase" lever: BAR1's 16GB aperture (a fixed
PCI power-of-two sized at boot, already larger than the card's own 12GB
VRAM) is strong evidence Resizable BAR is already enabled and already at its
practical ceiling for this card -- there was no bigger setting to reach for.

### Decision

Switch to `Qwen2.5-Coder-3B-Instruct` (~43% of 7B's parameters) rather than
tune batch size / sequence length down on 7B, since a smaller model reduces
pressure on *both* constrained resources (VRAM and BAR1) at once, and
addresses the root cause rather than trading one constraint for another.
Confirmed available as a matching pre-quantized checkpoint,
`unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`, same naming pattern already
used for 7B. Re-measured immediately after the switch (`scripts/check_vram.py
--backend unsloth --batch-size 2`): **4.09GB allocated / 4.72GB reserved**
peak for one step -- under half of 7B's footprint, with over 7GB of the 12GB
card free. See `configs/train_config.yaml` and `load_model.py` for where
this landed in code.

**This is a real trade-off, not a free upgrade** -- flagging plainly:
- The measured baselines in `reports/baseline_evaluation.md` (75.9%
  zero-shot, 80.2% few-shot) are 7B-specific and no longer apply; 3B needs
  its own baseline run before any fine-tuning result can be compared against
  anything.
- The **≥85% target** (see below) was calibrated against 7B's baselines and
  7B-specific published research -- now provisional pending the new 3B
  baseline.
- Smaller models are generally weaker at exactly what Spider's fully
  cross-domain test set stresses (generalizing to 40 schemas never seen in
  training), so this may lower the ceiling as well as the floor.
- README's "Design decisions" table originally justified 7B as "fits
  comfortably in 12GB VRAM" -- this incident is direct empirical evidence
  that claim didn't hold under sustained real training, which is itself
  the reason for revisiting it rather than a reason to avoid doing so.

## What runs

```bash
python scripts/train.py                                    # full run, per configs/train_config.yaml
python scripts/train.py --limit 20 --max-steps 2            # smoke test (no config changes needed)
python scripts/train.py --resume-from-checkpoint            # auto-resume from the latest checkpoint in output_dir
python scripts/train.py --resume-from-checkpoint checkpoints/checkpoint-536   # resume from a specific one
```

`src/text2sql/train/train_qlora.py` loads the base model + LoRA adapter
(`text2sql.models.load_model.load_model_for_training` -- shared with
`vram_check.py` so the VRAM measured ahead of time reflects what actually
trains), formats `data/processed/{train,dev}.json` into chat text, and trains
with `trl.SFTTrainer` against `configs/train_config.yaml`. Checkpoints and the
final LoRA adapter go to `checkpoints/` (gitignored; empty as of the pivot --
the 7B run never reached its first epoch-boundary checkpoint at step 536, so
nothing was lost by stopping it). `dev.json` is wired in as `eval_dataset`
(`eval_strategy: epoch`) purely for loss visibility across epochs -- not for
checkpoint selection; Spider's train split is modest (8,575 examples)
relative to the model, so watching for overfitting across the 3 configured
epochs seemed worth the eval cost (see reports/data_preparation.md's note on
why `dropout: 0.05` was kept for the same reason).

## Response-only loss masking

The prompt (schema + question) is several times longer than the response (the
gold SQL query), so training on the full sequence would spend most of the
gradient signal relearning schema tokens. `unsloth.chat_templates.train_on_responses_only`
masks everything outside the assistant turn to `-100`, given the literal
ChatML markers Qwen2.5 uses (`<|im_start|>user\n` / `<|im_start|>assistant\n`,
confirmed directly against the real tokenizer's `apply_chat_template` output,
not assumed from docs).

trl's own built-in `assistant_only_loss` was tried first since it's a native
`SFTConfig` flag -- rejected after checking directly: it depends on the chat
template containing a `{% generation %}` tag for `return_assistant_tokens_mask`
to work, and Qwen2.5-Coder's template doesn't have one. Confirmed with a
standalone call: it silently returns an all-zero mask, which would have
masked every token and produced zero training signal (or NaN loss) without
ever raising an error.

**Verified the masking is exactly correct**, not just "didn't crash" --
against the 7B model, before the pivot: ran the real pipeline through
`train_on_responses_only` on 5 real training examples and decoded only the
unmasked (non -100) label tokens. All 5 decoded to *exactly* the gold SQL
query, no more and no less:

```
[0] unmasked_text='SELECT count(*) FROM head WHERE age  >  56<|im_end|>\n'
    gold_sql     ='SELECT count(*) FROM head WHERE age  >  56'
[3] unmasked_text='SELECT max(budget_in_billions) ,  min(budget_in_billions) FROM department<|im_end|>\n'
    gold_sql     ='SELECT max(budget_in_billions) ,  min(budget_in_billions) FROM department'
```

**Re-verified directly against 3B (2026-08-06)**, not just assumed to carry
over from the same model family: ran the exact same check -- 4 real training
examples through the real `train_qlora.run()` pipeline (same
`load_model_for_training`, same `_load_dataset`, same
`train_on_responses_only` call) -- and decoded the unmasked labels. All 4
matched the gold SQL exactly (plus `<|im_end|>`), confirming the same
markers (`INSTRUCTION_PART`/`RESPONSE_PART`) still resolve correctly against
the 3B tokenizer:

```
[0] unmasked_text='SELECT count(*) FROM head WHERE age  >  56<|im_end|>\n'
    gold_sql     ='SELECT count(*) FROM head WHERE age  >  56'
[3] unmasked_text='SELECT max(budget_in_billions) ,  min(budget_in_billions) FROM department<|im_end|>\n'
    gold_sql     ='SELECT max(budget_in_billions) ,  min(budget_in_billions) FROM department'
```

No blockers remain before launching the real 3-epoch run.

## Real 3-epoch training run

Launched manually (`python scripts/train.py`, per the established pattern of
the user launching multi-hour GPU-bound runs themselves). Reached all three
epoch-boundary checkpoints:

| checkpoint | epoch | train_loss (at that point) | eval_loss | wall-clock timestamp |
|---|---|---|---|---|
| checkpoint-536 | 1.0 | ~0.08 (logged range 0.077-0.086 near boundary) | 0.2635 | 2026-08-06 22:15 |
| checkpoint-1072 | 2.0 | lower, not logged exactly | 0.2883 | 2026-08-07 01:38 |
| checkpoint-1608 | 3.0 | **0.00467** (final, from `train_metrics.json`) | **0.3618** | 2026-08-07 10:46 |

**Eval loss rises across all three epochs, and the rate of increase
accelerates** (epoch 0->1: +0.019, epoch 1->2: +0.025, epoch 2->3: +0.073 --
roughly 3x the per-epoch rate of the first two). Combined with train_loss
bottoming out near zero by epoch 3 (the model has essentially memorized the
training set's exact SQL strings), this is a clear overfitting signature by
the third epoch -- see "Post-tuning evaluation" below for whether it actually
costs execution_accuracy, which isn't guaranteed to move in lockstep with
token-level loss on a task with many valid SQL phrasings for one answer.

**A methodological caveat on run duration, checked directly rather than
trusted from the trainer's own summary:** `train_metrics.json` reports
`train_runtime: 8114.76s` (~2h15m) for the "full" run, which looks like great
news -- far under the ~8h estimate -- but this number is misleading.
Checkpoint directory timestamps show large gaps between epoch boundaries
(22:15 -> 01:38 = 3h22m, then 01:38 -> 10:46 = **9h08m**) that don't match
sustained training at the observed ~14-18s/step pace (536 steps at that pace
is ~2-2.5h, not 3.3h or 9.1h). The likely explanation: the run was paused and
resumed at least once, consistent with the plan discussed mid-run to
possibly stop after epoch 1. When `trainer.train(resume_from_checkpoint=...)`
is invoked as a fresh process, `train_runtime` in the returned metrics
reflects only that invocation's own active wall-clock time, not the
cumulative time across separate invocations -- and 8114.76s lines up closely
with one epoch's worth of steps at the live-observed rate (536 x ~15.1s/step
approx 8093s), consistent with the final invocation covering only the
last resumed segment. **The reliable throughput numbers remain the ones
measured directly during live epoch 1** (~14-18s/step, matching the
pre-run smoke-test estimate of ~17.5s/step) -- `train_metrics.json`'s
headline `train_runtime` should not be read as "the run only took 2.25h."
Total active GPU compute across all 3 epochs is better estimated at roughly
**~7-8h**, matching the original projection; the ~14h46m of wall-clock
calendar time it actually spanned includes real pauses, not just training.

## Post-tuning evaluation

New pipeline, mirroring `baseline_eval.py`'s structure rather than
duplicating it: `text2sql.models.load_model.load_finetuned_model` loads a
saved checkpoint directory (adapter_config.json + adapter_model.safetensors)
via Unsloth's `FastLanguageModel.from_pretrained(model_name=checkpoint_dir,
...)`, which reads `base_model_name_or_path` from the adapter config and
quantizes+merges automatically -- same pattern as `load_base_model`.
`text2sql.eval.evaluate.evaluate_checkpoint` then calls
`baseline_eval.run_baseline(mode="zero_shot", ...)` directly against that
model: the fine-tuned model was trained on exactly the `build_messages`
format (system + user turn, no few-shot demos -- see
`prompts/templates.build_training_messages`), so "zero_shot" mode is the
correct comparison, not a separate code path. `scripts/evaluate.py` is the
CLI entry point (`--checkpoints ... --split {dev,test} --limit N`).

**Methodology, decided before running anything:** checkpoint *selection*
(which epoch to ship) runs against `dev.json` -- the same split `eval_loss`
was already computed on during training -- so the choice isn't biased by
peeking at `test.json`. `test.json` gets evaluated exactly once, on
whichever single checkpoint wins the dev comparison, since that's what's
actually comparable to the baselines in `reports/baseline_evaluation.md`
(66.3%/74.3% execution_accuracy) and the ~80% target.

**Checkpoint comparison, 300-example dev slice** (`results/finetuned/checkpoint-*_dev.json`):

| checkpoint | epoch | exact_match | execution_accuracy |
|---|---|---|---|
| checkpoint-536 | 1 | 0.380 | **0.727** |
| checkpoint-1072 | 2 | 0.417 | 0.710 |
| checkpoint-1608 | 3 | 0.403 | 0.653 |

**Epoch 3 is out**: 5.7-7.4 points below epochs 1 and 2 on execution_accuracy,
directly confirming the eval_loss-predicted overfitting -- the accuracy
metric and the loss metric agree here.

**Epoch 1 vs. epoch 2, resolved with the full 1,034-example dev set:**
checkpoint-536 evaluated at execution_accuracy **0.7292** (exact_match
0.4449) on the full dev set -- landing right where its 300-example estimate
predicted (0.727), confirming that estimate wasn't noise. checkpoint-1072
was only measured at n=300 (0.710); given checkpoint-536's stability at
4x the sample size and its additional advantage of being the
lowest-overfitting-risk checkpoint (lowest eval_loss, fewest epochs),
**checkpoint-536 (epoch 1) was selected** without spending the ~35min to
also re-run checkpoint-1072 at full size -- the practical case for epoch 1
was strong enough without it.

## Final result

**checkpoint-536, full `test.json` (2,147 examples, one-shot evaluation as
planned):**

| | exact_match | execution_accuracy |
|---|---|---|
| Zero-shot (3B base, no tuning) | 26.9% | 66.3% |
| Few-shot (3B base, 5-shot) | 50.9% | 74.3% |
| **Fine-tuned (checkpoint-536)** | **43.0%** | **71.9%** |
| Target | -- | ~80% |

**Beats zero-shot clearly** (+5.6pt execution_accuracy, +16.1pt exact_match)
-- fine-tuning demonstrably helps over the untuned model. **Falls short of
both few-shot (-2.4pt execution_accuracy, -7.9pt exact_match) and the ~80%
target (-8.1pt).** This isn't a broken pipeline; every step from data
masking through checkpoint selection was verified directly along the way.
It's a real result that the original target didn't anticipate.

**Why fine-tuning didn't beat few-shot, most likely explanation:** the
few-shot baseline is shown 5 worked (question, SQL) examples *from the same
unseen test database* at inference time -- a live demonstration of that
specific schema's naming conventions and query style. The fine-tuned model
is evaluated zero-shot (matching its training format exactly), so at
inference time on a brand-new database it gets no examples for that specific
schema at all -- it has to generalize purely from patterns learned across
the 145 *different* training databases. This lines up with the baseline
report's own finding that few-shot's main effect looks like teaching
formatting conventions via direct demonstration rather than deeper schema
reasoning -- exactly the kind of signal a live in-context example provides
that a zero-shot-evaluated fine-tuned model structurally can't access.
Secondary contributors: checkpoint-536 is only epoch 1 (selected for lowest
overfitting risk, not because it was the most converged), and 8,575 training
examples across 145 databases is a modest amount of data for learning fully
general cross-database schema-linking.

**What this means for the ~80% target:** it was calibrated as "~5-6 points
over the few-shot baseline," implicitly assuming fine-tuning strictly
dominates few-shot (i.e. that fine-tuning is a superset of what few-shot
achieves). This result suggests that assumption doesn't hold -- few-shot and
fine-tuning appear to provide different, not strictly nested, kinds of
signal (per-database live demonstration vs. cross-database pattern
learning). The target wasn't miscalibrated due to an error; the underlying
assumption itself doesn't hold up against real data.

### Follow-on experiment: fine-tuning + few-shot combined

Since nothing about fine-tuning precludes also using in-context examples at
inference time, ran checkpoint-536 with the same 5-shot demonstrations the
few-shot baseline gets (`scripts/evaluate.py --mode few_shot`, reusing
`baseline_eval.run_baseline`'s existing few-shot code path and
`_sample_few_shot_pool`, sampled fresh from each eval split). A quick
300-example dev check showed a real improvement (execution_accuracy 0.727 ->
0.757, exact_match 0.380 -> 0.457) before committing to the full run.

**Full `test.json` result (2,147 examples):**

| | exact_match | execution_accuracy |
|---|---|---|
| Zero-shot (3B base) | 26.9% | 66.3% |
| Few-shot (3B base, 5-shot) | 50.9% | 74.3% |
| Fine-tuned, zero-shot | 43.0% | 71.9% |
| **Fine-tuned + few-shot** | **53.1%** | **76.2%** |
| Target | -- | ~80% |

**Beats both baselines individually**: +9.9pt execution_accuracy /
+26.2pt exact_match over zero-shot, +1.9pt / +2.2pt over few-shot, +4.3pt /
+10.1pt over fine-tuned alone. This directly confirms the hypothesis above:
fine-tuning's cross-database pattern learning and few-shot's live
per-schema demonstration are complementary signals, not redundant ones --
stacking them adds up rather than one subsuming the other. Still 3.8pt
short of the ~80% target, but this is the strongest result produced and the
one that should anchor the final write-up.

## Error analysis

New module, `src/text2sql/eval/error_analysis.py` + `scripts/error_analysis.py`,
works on any `run_baseline`-shaped results file (uniform across baselines
and fine-tuned checkpoints). `categorize_query()` classifies each gold SQL
by keyword heuristics matching Spider's own established complexity
taxonomy -- `JOIN` present, more than one `SELECT` (subquery), any of
`COUNT/SUM/AVG/MIN/MAX`/`GROUP BY` (aggregation), or none of the above
("select" = simple baseline). Not mutually exclusive except "select".

**execution_accuracy by category, all 4 conditions, full `test.json`:**

| category | n | zero-shot | few-shot | fine-tuned | fine-tuned+few-shot |
|---|---|---|---|---|---|
| select (simple) | 526 | 76.4% | **90.5%** | 86.7% | 88.6% |
| **join** | **862** | 56.8% | 61.0% | 59.2% | **67.1%** |
| subquery | 346 | 58.4% | 67.1% | 66.2% | 66.5% |
| aggregation | 1054 | 65.2% | 71.5% | 69.3% | **74.1%** |

Two findings:

1. **Join queries are the weak point across every condition** -- lowest
   execution_accuracy of all four categories in every single row, and the
   largest category by count (862/2,147 = 40% of the test set), so it's
   dragging the overall average down more than any other category simply by
   weight. This is the highest-value target if pushing accuracy further
   later (see "Follow-on experiment" note above about diminishing returns
   elsewhere).
2. **The combined approach's improvement is concentrated exactly where it's
   needed, not spread evenly.** On "select" (already near-ceiling at 90.5%
   for few-shot alone), combining doesn't help (88.6%, within noise of
   few-shot alone, arguably a small regression). On subquery, combining is
   flat vs. few-shot alone (66.5% vs 67.1%). But on **join** -- the hardest,
   largest category -- combining beats few-shot alone by **+6.1pt** (61.0%
   -> 67.1%), and on aggregation by +2.6pt (71.5% -> 74.1%). The two signals
   complement each other specifically on the hard, high-value cases, not on
   the easy ones where either alone was already sufficient -- a more precise
   version of the "complementary, not redundant" finding above.

Full breakdown: `results/error_analysis/breakdown.json`.

## Bugs found while wiring this up

Four real, reproduced-not-assumed issues, in the order they were hit:

### 1. Unsloth import order silently corrupts `SFTConfig`

First smoke test failed with:
```
ValueError: The specified `eos_token` ('<EOS_TOKEN>') is not found in the vocabulary
```
`eos_token` was never set anywhere in this project's code -- `sft_config.eos_token`
printed `None` right before the failing call. Root cause, tracked down by
reproducing each step in isolation: `trl`/`transformers` got imported (via
`from trl import SFTConfig, SFTTrainer`) *before* `import unsloth`, so
`SFTTrainer.__init__` ends up round-tripping a validly-constructed `SFTConfig`
through `TrainingArguments.to_dict()` (an `isinstance` check against the
Unsloth-patched class fails, since the config object was built from the
pre-patch class). `to_dict()` unconditionally redacts every field whose name
ends in `_token` -- real secret or not -- replacing it with the literal string
`f"<{FIELD_NAME.upper()}>"` (`transformers/training_args.py`, `to_dict()`).
`eos_token=None` becomes `"<EOS_TOKEN>"`, which then fails the vocabulary
lookup. This is exactly the ordering Unsloth's own runtime warning names
("Unsloth should be imported before [trl, transformers, peft]") -- the
warning was visible in the log but easy to dismiss as cosmetic since nothing
in this project's code does that import directly (`unsloth.chat_templates`
does, one line below where the warning fires). Fixed by moving `import
unsloth` to the first line of `run()`, before any `trl`/`transformers`
import.

### 2. Unsloth's compiled `SFTTrainer` doesn't auto-detect conversational datasets

Vanilla `trl.SFTTrainer` auto-detects a `messages`-column dataset (verified by
reading `trl/trainer/sft_trainer.py` directly) and applies the tokenizer's
chat template itself. Unsloth compiles and caches its own drop-in replacement
(`unsloth_compiled_cache/UnslothSFTTrainer.py`, generated on first use) that
doesn't replicate this -- passing the raw `messages` dataset raised `"Unsloth:
You must specify a formatting_func"`. Worked around by applying the chat
template ourselves into a plain `text` column before handing the dataset to
`SFTTrainer` (`_load_dataset()` in `train_qlora.py`), which sidesteps the
compiled hook's `formatting_func` contract entirely (its own test call passes
a single non-batched example to a function whose real invocation is batched
-- inspected directly, not reliable enough to depend on).

### 3. Unsloth's startup banner crashes on Windows' default console encoding

`unsloth`'s own `print()` calls include emoji (e.g. U+1F9A5 "sloth"). Windows'
default console codepage (cp1252) can't encode them, crashing the process
*after* the multi-GB model finishes loading:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f9a5'
```
Fixed in `scripts/train.py` by reconfiguring `sys.stdout`/`sys.stderr` to
UTF-8 before importing anything else -- cheap, and avoids losing a 20s+ model
load to a print statement on every run on this OS.

### 4. Unsloth's fast downloader (`hf_transfer`) can't recover from its own corrupted partial download, on Windows

Hit while downloading the 3B model for the first time after the pivot:
```
Exception: Error while removing corrupted file: The process cannot access the
file because it is being used by another process. (os error 32)
```
Deterministic, not a one-off flake -- retried twice (including manually
deleting the leftover `.incomplete` blob from the HF cache between attempts)
and hit the identical error both times. `hf_transfer` (Unsloth's default
"fast downloading", a Rust-based accelerated downloader) fails to clean up
its own corrupted partial-download blob on Windows when a retry needs to
replace it. Worked around by disabling it (`HF_HUB_ENABLE_HF_TRANSFER=0`),
which falls back to the standard, slower-but-robust downloader. Fixed
permanently in `src/text2sql/__init__.py` rather than in `load_model.py`:
`huggingface_hub` caches this as a module-level constant
(`huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER`, read once via
`os.environ.get()` at that module's own import time), and by the time
`load_model.py` runs, `unsloth`/`transformers`/`huggingface_hub` have
already been imported (e.g. `train_qlora.py` does `import unsloth` before
importing anything from this project) -- so setting the env var there would
already be too late. Every entry point imports some `text2sql` submodule
first, which always executes `text2sql/__init__.py` before descending into
it, making that the one place early enough to guarantee it lands in time.
Verified directly (not assumed): imported `text2sql` first exactly as the
real entry points do, confirmed the env var was set before `huggingface_hub`
was ever imported, then imported `huggingface_hub.constants` and confirmed
its cached value reflected the fix.

## VRAM

**7B (superseded by the pivot above; kept as historical record):**
`vram_check.py` measured one isolated forward+backward+optimizer step at
8.36GB allocated / 9.92GB reserved -- looked comfortable under the 12GB
target, but a real multi-hour run still drove usage to ~11.3GB dedicated VRAM
and nearly exhausted BAR1 memory (see Pivot section above for the full
diagnosis). The real `SFTTrainer`-based run adds one wrinkle not present in
that isolated single-step test: Unsloth auto-enables "padding-free" batching
for `SFTTrainer` specifically (`Unsloth: Padding-free auto-enabled`), which
changes the memory profile somewhat from the raw manual loop `vram_check.py`
runs -- a real, confirmed methodological difference, though its exact
magnitude wasn't isolated.

**3B (current):** re-measured immediately after the pivot, same command and
methodology (`scripts/check_vram.py --backend unsloth --batch-size 2`):

| | 7B | 3B |
|---|---|---|
| peak allocated | 8.36GB | **4.09GB** |
| peak reserved | 9.92GB | **4.72GB** |
| free at reserved (of 12GB) | ~2GB | **~7.3GB** |

The much larger margin (3.5x the free headroom 7B had) makes it unlikely that
3B will run into the same underprediction the 7B single-step measurement did
-- but that's exactly the kind of claim this project has learned not to trust
until a real, hours-long run confirms it.

## Throughput

Re-measured against 3B (`python scripts/train.py --max-steps 10`, real full
training data, same method as the superseded 7B number):

| | 7B (superseded) | 3B (current) |
|---|---|---|
| time per optimizer step | ~30.4s | **~17.5s** |
| dev-set eval pass (1,034 examples) | ~385s (~6.4min) | **~202s (~3.4min)** |
| steps/epoch (8,575 examples / effective batch 16) | ~536 | ~536 (unchanged -- same batch config) |
| **estimated full run (3 epochs, 1,608 steps)** | ~13.6h training + ~19min eval ≈ **~14h** | **~7.8h training + ~10min eval ≈ ~8h** |

Real speedup (~1.74x), though not proportional to the ~2.5x parameter-count
ratio (7.66B/3.12B) -- consistent with the pattern seen throughout this
project that fixed overheads (data loading, `Trainer` bookkeeping,
gradient-checkpointing recompute) don't scale down linearly with model size.

**Unlike the 7B measurement, this one shows no sign of the degradation
pattern that caused the pivot**: per-step time over the 10 real steps (after
excluding step 1's one-time compilation/warmup overhead) was stable at
~11-19s with no accelerating trend, consistent with 3B's much larger VRAM
margin (`~7.3GB free vs 7B's ~2GB, see VRAM section above) not putting the
same pressure on BAR1. Still just a 10-step measurement, not the full
1,608-step run -- carrying forward the same caveat as before, just with
much more headroom behind it this time.

## Target

Set at **execution accuracy ≥85%** on `test.json` -- **provisional,
calibrated against the 7B model, pending re-derivation now that the base
model has changed.** Original reasoning, kept for context: web research
(2026-08-06) found that Qwen2.5-Coder-7B-Instruct specifically showed a
small *regression*, not an improvement, from plain LoRA SFT fine-tuning on
Spider in at least one published write-up, since the model already carries
strong SQL competence from code-heavy pretraining. A more sophisticated
method (schema-aware pre-synthesized queries, not plain SFT) only reached
82.1% execution accuracy on Spider test with the same 7B base model, and
leaderboard systems above 90% rely on schema linking, self-consistency, and
ensembling well beyond a single LoRA run -- 85% was set as a defensible
"clear, measurable improvement" over 7B's 80.2% few-shot baseline rather
than a higher round number.

None of the 7B-specific research this was based on has been re-checked
against a 3B model, and the baseline it was measured relative to (80.2%) no
longer applies. Once 3B's own baselines are measured, this target should be
explicitly revisited rather than carried over by default.

**Both 3B baselines are now measured** (`reports/baseline_evaluation.md`):

| | 3B | 7B (superseded) | Δ |
|---|---|---|---|
| zero-shot execution_accuracy | 66.3% | 75.9% | -9.6 |
| few-shot execution_accuracy | 74.3% | 80.2% | -5.9 |
| few-shot exact_match | 50.9% | 50.5% | +0.4 |

Unexpected finding: few-shot closes most of the zero-shot gap, and fully
closes it on exact_match (3B few-shot slightly exceeds 7B's).

**Revised target (set 2026-08-06): execution accuracy ~80%** on `test.json`.
Same margin logic as the original 85% figure ("~5 points over the few-shot
baseline"), applied to 3B's own 74.3% few-shot baseline instead of 7B's
80.2%. Decided pragmatically rather than re-grounded in fresh 3B-specific
research the way the original 85% was -- the web research behind that
figure was specific to 7B's published fine-tuning behavior and hasn't been
checked against 3B at all. If 3B's fine-tuning behavior turns out to differ
meaningfully from 7B's (plausible, given how much closer 3B already runs to
7B on few-shot than zero-shot), this number may need revisiting once real
training results exist -- but ~80% is now the working target, not a
placeholder extrapolation.

## Next steps

1. ~~Run baseline evaluation (`scripts/run_baseline_eval.py`) against
   `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`~~ -- done: zero-shot
   66.3%/26.9%, few-shot 74.3%/50.9%.
2. ~~Revisit the ≥85% target against the new 3B baselines~~ -- done: ~80%,
   see above.
3. ~~Re-verify response-only masking against 3B specifically~~ -- done, see
   above: 4 examples decoded, all matched gold SQL exactly.
4. ~~Re-measure real training throughput on 3B~~ -- done: ~17.5s/step,
   ~8h full-run estimate, no accelerating-slowdown pattern observed.
5. ~~Run `python scripts/train.py` for the real 3-epoch fine-tune~~ -- done:
   `checkpoints/checkpoint-{536,1072,1608}` + `checkpoints/final`.
6. ~~Wire up post-tuning evaluation (`scripts/evaluate.py`)~~ -- done, see
   above.
7. ~~Resolve checkpoint-536 vs. checkpoint-1072 on the full dev set~~ --
   done: checkpoint-536 selected (0.7292 execution_accuracy at n=1034,
   stable vs. its own n=300 estimate; also lowest-overfitting-risk option).
8. ~~Run the winning checkpoint once against `test.json`~~ -- done: exact_match
   43.0%, execution_accuracy 71.9%. See "Final result" above.
9. ~~Follow-on experiment: evaluate checkpoint-536 with few-shot examples
   added at inference time~~ -- done: exact_match 53.1%, execution_accuracy
   76.2%, beats both baselines. See "Follow-on experiment" above.
10. ~~Error analysis~~ -- done: join queries are the weak point across every
    condition (largest category too); combining fine-tuning + few-shot
    helps most exactly there (+6.1pt) and on aggregation (+2.6pt), not on
    the already-easy select category. See "Error analysis" above.
11. ~~Gradio demo~~ -- done: `demo/app.py` already had a full UI shell built
    (curated 6-database picker, schema + example questions, live SQL
    execution) with a mock gold-lookup standing in for the model; swapped
    `predict_sql()` for real generation via `checkpoint-536` + few-shot
    demos from the selected database (the 76.2% combination). Verified via
    `gradio_client` against the live server, not just the underlying
    functions: `/on_db_change` and `/on_generate` both confirmed working
    end-to-end, including a hand-typed question outside the example set
    generating and executing correctly.
12. ~~Technical write-up~~ -- done: `reports/technical_report.md` pulls
    together dataset, baselines, fine-tuning, results, and error analysis
    into one polished document.

All roadmap items complete. Project closed out -- see README's Roadmap
section and `reports/technical_report.md` for the final write-up.
