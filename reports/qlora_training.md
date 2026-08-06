# QLoRA Training

Status: pipeline built, smoke-tested, and verified correct end-to-end on the
real model/data/GPU -- **but the base model changed mid-project**. A real
training run on the originally-chosen Qwen2.5-Coder-7B-Instruct ran for
~4h before it became clear it would take ~59h total, not the ~14h estimated
from a smoke test -- traced to VRAM/BAR1 pressure specific to running a 7B
model this close to the 12GB limit (full incident below). The project has
pivoted to **Qwen2.5-Coder-3B-Instruct**. The training run itself has not
been (re-)executed yet; baseline evaluation against the new 3B base model is
next, before another real training attempt.

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
5. Run `python scripts/train.py` for the real 3-epoch fine-tune -- no
   remaining blockers.
6. Once it produces `checkpoints/final`, wire up post-tuning evaluation
   (`scripts/evaluate.py`, currently an empty placeholder) against the same
   `test.json` split, for an apples-to-apples comparison.
