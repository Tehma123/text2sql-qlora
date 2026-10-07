# Data Preparation

Status: done. Produced by `scripts/download_data.py` + `scripts/prepare_data.py`.

## Getting the dataset

Spider is distributed two ways:

| Source | Has SQL pairs | Has schema (`tables.json`) | Has per-DB SQLite files |
|---|---|---|---|
| HF `datasets` (`xlangai/spider`) | Yes | No | No |
| Original release ([yale-lily.github.io/spider](https://yale-lily.github.io/spider), Google Drive) | Yes | Yes | Yes |

Execution-accuracy evaluation (this project's primary metric, per the README) needs
real SQLite databases to run generated queries against, so the pipeline downloads
the **original release**, not the HF mirror.

```bash
python scripts/download_data.py   # -> data/raw/
python scripts/prepare_data.py    # -> data/processed/
```

`download_data.py` fetches the zip from Google Drive via `gdown` and extracts it
into `data/raw/`. Google Drive periodically rate-limits this file; if the
automated download fails, the script prints manual fallback steps (download by
hand, drop the zip at `data/raw/spider.zip`, re-run).

## What's actually in the archive

The archive turned out to include more than the README draft assumed going in —
worth recording since it changed the split design:

- `train_spider.json` + `train_others.json` — training pairs, 166 databases (`tables.json`, `database/`)
- `dev.json` — validation pairs, same 166 databases
- `test.json` — a **genuinely held-out test set** with its own `test_tables.json` + `test_database/`

Spider's leaderboard test set is famously hidden, so the initial plan (before
inspecting the download) was to fake a test split by carving a slice out of
train and repurposing `dev.json` as "test". That turned out to be unnecessary:
this release ships a real `test.json` whose questions are drawn from 40
databases that **do not appear anywhere in train or dev** — true cross-domain
held-out evaluation, not just unseen questions on familiar schemas.

## Resulting splits

Final counts, after the sequence-length filter described below (84 train
examples dropped for exceeding `max_seq_length`):

| Split | Examples | Distinct DBs | Overlaps train/dev DBs? |
|---|---|---|---|
| train | 8,575 (of 8,659 raw) | 146 | — |
| dev | 1,034 | 20 | no overlap with train's 146 |
| test | 2,147 | 40 | no — fully disjoint from train+dev |

## Schema representation

Each database's schema is rendered as `CREATE TABLE` DDL (original column/table
casing, types, primary keys, foreign keys as `FOREIGN KEY ... REFERENCES ...`)
by `src/text2sql/data/schema_utils.py`. DDL was chosen over a flatter
"table: col1, col2" listing because it's closer to what the base model has
seen during pretraining and keeps primary/foreign-key relationships explicit,
which Spider questions frequently depend on (joins).

**Correction (2026-10-07):** this section originally said no Spider database
has a composite primary key, so every PK could be rendered inline. That was
checked against `tables.json` only, and `tables.json` turns out to keep just
the *first* column of each composite key. Against the real SQLite files
(`PRAGMA table_info`), 132 of 876 tables in `tables.json` and 161 of 1,056 in
`test_tables.json` have a composite key -- e.g. `college_2.takes` is keyed on
`(ID, course_id, sec_id, semester, year)` but was rendered as
`"ID" TEXT PRIMARY KEY`, telling the model a non-unique column is unique.
Every other table matched (649 single-column keys, 95 tables with no key).
That reached 4,554/8,575 train (53%), 387/1,034 dev and 1,091/2,147 test
examples. `schema_utils.py` now reads primary keys from the SQLite file and
renders a composite key as a table-level `PRIMARY KEY (a, b)` clause;
single-column keys are still inline, so those tables' DDL is unchanged.
Foreign keys were checked the same way and match (all 792 in `tables.json`
exist in SQLite). The first training run (`checkpoints/checkpoint-*`) used
the old prompts, kept in `data/processed/v1_partial_pk/`.

## Instruction format

`src/text2sql/prompts/templates.py` builds chat messages (system + user + gold
assistant turn) rather than a hand-rolled flat string, so both the SFT trainer
and eval scripts apply the base model's own chat template instead of
duplicating special-token handling in two places. The same module also builds
few-shot messages for the baseline evaluation (worked examples drawn from the
same database as the target question, so they don't leak schema information
the model wouldn't otherwise have).

Processed records (`data/processed/{train,dev,test}.json`) carry `db_id`,
`db_dir` (`database` vs `test_database` — needed later to locate the right
`.sqlite` file for execution-accuracy scoring), `question`, `query`, the
rendered `schema`, and the full `messages` list.

## Verified

- Extracted archive layout matches expectations (`train_spider.json`,
  `train_others.json`, `dev.json`, `tables.json`, `database/`, `test.json`,
  `test_tables.json`, `test_database/`).
- A sample SQLite database (`department_management`) opens and executes a
  query correctly.
- `prepare_data.py` ran end-to-end and produced the split sizes above.

## Sequence length vs. max_seq_length

Measured with the real `Qwen/Qwen2.5-Coder-7B-Instruct` tokenizer
(`scripts/check_seq_lengths.py`, applied to the full chat-formatted
`messages`, i.e. system + schema/question + gold SQL).

At the originally-planned budget of 1024, dev/test were nearly unaffected but
~5.6% of train exceeded it (large-schema databases produce long DDL). Since
truncating from the right would clip the gold SQL answer rather than the
schema -- training the model on incomplete queries -- the choice was between
dropping those examples or raising the budget. Re-checked at higher
candidate budgets before deciding:

| `max_seq_length` | train over-budget | dev / test over-budget |
|---|---|---|
| 1024 | 481 (5.55%) | 0 / 4 |
| 1280 | 164 (1.89%) | 0 / 0 |
| 1536 | 84 (0.97%) | 0 / 0 |

**Applied: `max_seq_length: 1536`** in both `configs/data_config.yaml` and
`configs/train_config.yaml`, with `preprocess.py` now dropping any example
still over budget after that bump (84 train outliers, up to 3,067 tokens;
0 in dev/test). Re-verified post-filter: max sequence length across all three
splits is now 1,506 tokens, safely under 1,536.

Longer sequences cost more VRAM/step time, which raised the question of
whether `per_device_train_batch_size: 2` still fits on the 12GB target GPU.
Rather than guess, this was measured directly (`src/text2sql/train/vram_check.py`,
`scripts/check_vram.py`) on the actual RTX 3060 12GB this environment has:
load the 4-bit base model, wrap it with the configured LoRA setup, build a
batch from the two longest examples in the processed train split (1506/1505
tokens -- near-worst-case for `max_seq_length: 1536`), and measure peak VRAM
over one real forward+backward+optimizer step.

| Backend | batch_size | peak allocated | peak reserved | Fits in 12GB? |
|---|---|---|---|---|
| plain `peft` + `bitsandbytes` | 2 | 16.50GB | 18.01GB | No |
| plain `peft` + `bitsandbytes` | 1 | 12.20GB | 12.97GB | No (reserved > 12GB even at batch 1) |
| **Unsloth** (`FastLanguageModel`) | **2** | **8.36GB** | **9.92GB** | **Yes** |

The plain-`peft` numbers are a worst-case baseline, not what this project
actually trains with -- Unsloth wasn't importable in this environment at
first (missing `torchvision`; fixed by installing a matching build from
`https://download.pytorch.org/whl/cu126`, which also upgraded torch
2.10.0 -> 2.13.0+cu126). Once Unsloth actually loaded, `batch_size: 2`
comfortably fits with ~2GB of headroom to spare. The gap between the two
backends (16.5GB vs 8.4GB) comes from Qwen's 152K-token vocabulary: computing
the loss materializes full-sequence logits in fp32, which plain
transformers/peft doesn't optimize away but Unsloth's fused kernels do --
this is precisely the kind of cost the README's "VRAM-efficient... via
Unsloth" design decision was about, now verified rather than assumed.

**Decision: keep `per_device_train_batch_size: 2`** in `configs/train_config.yaml`
(no change needed) -- confirmed to fit, not just intended to.

One side finding from Unsloth's own startup log: `lora.dropout: 0.05`
disables Unsloth's fastest QKV/O/MLP kernel patches ("Unsloth will patch all
other layers, except LoRA matrices, causing a performance hit"). VRAM savings
still applied regardless (those come from separate optimizations -- offloaded
gradient checkpointing, fused cross-entropy -- that don't depend on dropout),
but full training throughput would need `dropout: 0`.

**Decision: keep `dropout: 0.05`.** Spider's train split is modest (8,575
examples) relative to a 7B model, and `test` is fully cross-domain, so the
regularization is worth more here than the extra training speed -- this
trades wall-clock time, not correctness or VRAM headroom.

While testing this, `check_vram.py` also surfaced a real, unrelated bug:
`configs/train_config.yaml` had `learning_rate: 2e-4`, which PyYAML parses as
the *string* `"2e-4"`, not a float, because its scientific-notation matcher
requires a decimal point in the mantissa. This would have broken the actual
training script the same way. Fixed to `2.0e-4`.
