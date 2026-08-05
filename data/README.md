# Data

This project uses the [Spider](https://yale-lily.github.io/spider) Text-to-SQL dataset. It is not checked into the repo — fetch it into `raw/` before running preprocessing.

## Getting Spider

```bash
python scripts/download_data.py
python scripts/prepare_data.py
```

`download_data.py` pulls the **original distribution** (via Google Drive), not the
Hugging Face `datasets` mirror (`xlangai/spider`) — the HF version only has
question/query/db_id triples, while the original zip also ships `tables.json`
(schema) and the per-database SQLite files needed for execution-accuracy
evaluation. It writes `train_spider.json`, `train_others.json`, `dev.json`,
`tables.json`, and `database/` straight into `raw/`.

If the automated download fails (Google Drive rate-limits this file
periodically), download it manually from the
[Spider project page](https://yale-lily.github.io/spider), save the zip as
`data/raw/spider.zip`, and re-run `download_data.py` — it will pick up the
zip already on disk and just extract it.

`prepare_data.py` then builds instruction-formatted splits in `processed/`:
`train_spider.json` + `train_others.json` -> `train`, `dev.json` -> `dev`
(monitor training / tune hyperparameters), `test.json` -> `test` (never
trained on, used only for the final baseline/fine-tuned comparison). Test is
genuinely held out: 40 of its 206 databases don't appear anywhere in
train/dev, so it also measures cross-domain generalization, not just unseen
questions on familiar schemas.

## Layout

- `raw/` — untouched, as downloaded (questions/SQL pairs + per-database schema + SQLite `.db` files)
- `processed/` — instruction-formatted train/dev/test splits produced by `src/text2sql/data/preprocess.py`, referenced from `configs/data_config.yaml`

Both directories are gitignored (contents only — `.gitkeep` placeholders keep the folders in version control).
