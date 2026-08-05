"""Download the Spider dataset (original distribution) into data/raw/.

The original distribution -- not the Hugging Face `datasets` mirror -- is used
because it ships the per-database SQLite files and tables.json schema needed
for execution-accuracy evaluation. The HF mirror (`xlangai/spider`) only has
question/query/db_id triples.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

from text2sql.utils.config import load_config, resolve_path
from text2sql.utils.logging import get_logger

logger = get_logger(__name__)

# Google Drive file ID for the official release, linked from
# https://yale-lily.github.io/spider
SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"

# Entries we expect directly under raw_dir once the zip is extracted and flattened.
# train_spider/train_others/dev share tables.json + database/; test.json is a
# genuinely held-out split (40 of its 206 databases never appear in train/dev)
# with its own test_tables.json + test_database/.
EXPECTED_ENTRIES = [
    "train_spider.json",
    "train_others.json",
    "dev.json",
    "tables.json",
    "database",
    "test.json",
    "test_tables.json",
    "test_database",
]


def already_downloaded(raw_dir: Path) -> bool:
    return all((raw_dir / entry).exists() for entry in EXPECTED_ENTRIES)


def _manual_instructions(zip_path: Path) -> str:
    return (
        "Automated download failed. Google Drive rate-limits/quota-blocks this file "
        "fairly often since it's a large, popular public file.\n"
        "Fetch it manually instead:\n"
        "  1. Open https://yale-lily.github.io/spider and click the dataset download link\n"
        f"  2. Save the zip as {zip_path}\n"
        "  3. Re-run this script (it will extract the zip already sitting there)"
    )


def download_spider(raw_dir: Path, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    if already_downloaded(raw_dir) and not force:
        logger.info("Spider already present in %s (use --force to re-download)", raw_dir)
        return

    zip_path = raw_dir / "spider.zip"

    if not zip_path.exists():
        try:
            import gdown
        except ImportError as e:
            raise SystemExit(
                "gdown is required to download Spider (`pip install gdown`), "
                "or fetch it manually -- see data/README.md."
            ) from e

        logger.info("Downloading Spider from Google Drive (this is a large file, ~1GB)...")
        try:
            result = gdown.download(id=SPIDER_GDRIVE_ID, output=str(zip_path), quiet=False)
        except Exception as e:
            raise SystemExit(f"{_manual_instructions(zip_path)}\n\nOriginal error: {e}") from e

        if result is None or not zip_path.exists() or zip_path.stat().st_size == 0:
            raise SystemExit(_manual_instructions(zip_path))
    else:
        logger.info("Found existing %s, skipping download", zip_path)

    logger.info("Extracting %s", zip_path)
    extract_dir = raw_dir / "_extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    # The archive wraps everything in one top-level folder (observed name:
    # "spider_data", but don't hardcode it -- find the dir that actually
    # contains tables.json, and ignore zip-generated junk like __MACOSX).
    candidates = [d for d in extract_dir.iterdir() if d.is_dir() and d.name != "__MACOSX"]
    inner = next((d for d in candidates if (d / "tables.json").exists()), None)
    source = inner if inner is not None else extract_dir

    for item in source.iterdir():
        if item.name in (".DS_Store",):
            continue
        dest = raw_dir / item.name
        if dest.is_dir():
            shutil.rmtree(dest)
        elif dest.exists():
            dest.unlink()
        shutil.move(str(item), str(dest))

    missing = [e for e in EXPECTED_ENTRIES if not (raw_dir / e).exists()]
    if missing:
        raise SystemExit(
            f"Extraction finished but is missing expected entries: {missing}. "
            f"Left {extract_dir} and {zip_path} in place for inspection."
        )

    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    logger.info("Spider dataset ready in %s", raw_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Spider dataset into data/raw/")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = resolve_path(config["raw_dir"])
    download_spider(raw_dir, force=args.force)


if __name__ == "__main__":
    main()
