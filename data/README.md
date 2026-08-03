# Data

This project uses the [Spider](https://yale-lily.github.io/spider) Text-to-SQL dataset. It is not checked into the repo — fetch it into `raw/` before running preprocessing.

## Getting Spider

Either:
- Load it via the Hugging Face `datasets` library (e.g. `xlangai/spider`), or
- Download it directly from the [Spider project page](https://yale-lily.github.io/spider) (original distribution, including the per-database SQLite files needed for execution accuracy).

## Layout

- `raw/` — untouched, as downloaded (questions/SQL pairs + per-database schema + SQLite `.db` files)
- `processed/` — instruction-formatted train/dev/test splits produced by `src/text2sql/data/preprocess.py`, referenced from `configs/data_config.yaml`

Both directories are gitignored (contents only — `.gitkeep` placeholders keep the folders in version control).
