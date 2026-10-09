"""Gradio demo for Text2SQL-Tune.

Real model, not a mock: loads `checkpoints/v2/checkpoint-600` (the v2
checkpoint selected in `reports/qlora_training.md`) and generates SQL live,
combined with few-shot demonstrations from the same database -- the
combination that measured best (execution_accuracy 77.4%, vs. 71.7%
fine-tuned alone -- see "v2: composite-primary-key fix and retrain" in that
report). Uses the same generation code as
evaluation (`text2sql.eval.baseline_eval.generate_sql`,
`build_few_shot_messages`), so the demo's behavior matches what was actually
measured rather than a separate reimplementation.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import gradio as gr
import pandas as pd

# Unsloth prints emoji straight to stdout; Windows' default console codepage
# (cp1252) can't encode them and crashes right after the model finishes
# loading -- see scripts/train.py for the same fix.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text2sql.data.schema_utils import get_db_path  # noqa: E402
from text2sql.eval.baseline_eval import generate_sql  # noqa: E402
from text2sql.models.load_model import load_finetuned_model  # noqa: E402
from text2sql.prompts.templates import build_few_shot_messages  # noqa: E402
from text2sql.utils.config import load_config, resolve_path  # noqa: E402
from text2sql.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

DATA_CONFIG = load_config("configs/data_config.yaml")
EVAL_CONFIG = load_config("configs/eval_config.yaml")
RAW_DIR = resolve_path(DATA_CONFIG["raw_dir"])
TEST_FILE = resolve_path(DATA_CONFIG["test_file"])
MAX_SEQ_LENGTH = DATA_CONFIG["max_seq_length"]
NUM_FEW_SHOT = EVAL_CONFIG.get("baselines", {}).get("few_shot", {}).get("num_examples", 5)
CHECKPOINT_DIR = resolve_path("checkpoints/v2/checkpoint-600")

EXAMPLES_PER_DB = 5
QUERY_TIMEOUT = 5.0

# A handful of small, relatable schemas from the real (held-out) test split,
# picked for demo readability -- not representative of the full 40-database
# test set's difficulty. All from data/processed/test.json, so their schema
# DDL and example questions are real pipeline output, not hand-written.
DB_LABELS = {
    "bakery_1": "Bakery -- customers, goods, receipts",
    "book_1": "Bookstore -- clients, authors, orders",
    "vehicle_rent": "Vehicle rental -- customers, vehicles, rental history",
    "headphone_store": "Headphone store -- models, stores, stock",
    "pilot_1": "Aviation -- pilots, planes, hangars",
    "art_1": "Art gallery -- artists, paintings, sculptures",
}
CURATED_DBS = list(DB_LABELS)

EMPTY_QUESTION = "Type a question, or click one of the example questions on the left."


def _load_test_examples() -> dict[str, list[dict]]:
    with open(TEST_FILE, encoding="utf-8") as f:
        data = json.load(f)
    by_db: dict[str, list[dict]] = {}
    for ex in data:
        by_db.setdefault(ex["db_id"], []).append(ex)
    return by_db


_EXAMPLES_BY_DB = _load_test_examples()

logger.info("Loading fine-tuned model from %s ...", CHECKPOINT_DIR)
_MODEL, _TOKENIZER = load_finetuned_model(str(CHECKPOINT_DIR), MAX_SEQ_LENGTH)
logger.info("Model loaded.")


def _dedup_by_query(examples: list[dict]) -> list[dict]:
    """Spider asks most gold queries in 2 paraphrased forms back to back --
    dedup by query so the example list covers distinct intents, not near-duplicates."""
    seen: set[str] = set()
    unique = []
    for ex in examples:
        if ex["query"] in seen:
            continue
        seen.add(ex["query"])
        unique.append(ex)
    return unique


def get_schema(db_id: str) -> str:
    return _EXAMPLES_BY_DB[db_id][0]["schema"]


def get_example_questions(db_id: str) -> list[str]:
    return [ex["question"] for ex in _dedup_by_query(_EXAMPLES_BY_DB[db_id])[:EXAMPLES_PER_DB]]


def _few_shot_demos(db_id: str, question: str) -> list[dict[str, str]]:
    """Worked (question, query) pairs from this database, for use as in-context
    demos -- excludes `question` itself, matching `evaluate.py --mode few_shot`'s
    exclusion logic so a clicked example question can't just show its own answer."""
    q_norm = question.strip().lower()
    candidates = [ex for ex in _dedup_by_query(_EXAMPLES_BY_DB[db_id]) if ex["question"].strip().lower() != q_norm]
    return [{"question": ex["question"], "query": ex["query"]} for ex in candidates[:NUM_FEW_SHOT]]


def predict_sql(db_id: str, question: str) -> tuple[str, str]:
    """Real generation: fine-tuned model + few-shot demos from this database
    (the combination measured at 77.4% execution_accuracy -- see
    reports/qlora_training.md). Returns (sql, status_note); `sql` is empty if
    nothing could be produced."""
    if not question or not question.strip():
        return "", EMPTY_QUESTION
    demos = _few_shot_demos(db_id, question)
    schema = get_schema(db_id)
    messages = build_few_shot_messages(question, schema, demos)
    sql = generate_sql(_MODEL, _TOKENIZER, messages, MAX_SEQ_LENGTH)
    note = f"Generated using {len(demos)} example question(s) from this database as context."
    return sql, note


def run_query(db_id: str, db_dir: str, sql: str) -> tuple[list[str] | None, list[tuple] | None, str | None]:
    """Execute `sql` read-only against the real per-database SQLite file.

    Mirrors the connection/timeout pattern in `text2sql.eval.metrics.execute_query`
    (read-only URI mode, progress-handler timeout) plus column names, which that
    function doesn't need for its own use (accuracy comparison, not display).
    """
    db_path = get_db_path(db_id, RAW_DIR, db_dir)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    start = time.monotonic()
    con.set_progress_handler(lambda *_: 1 if time.monotonic() - start > QUERY_TIMEOUT else 0, 1000)
    try:
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
        return columns, rows, None
    except Exception as e:
        return None, None, str(e)
    finally:
        con.close()


def _load_metrics_table() -> str:
    """Reads real numbers from results/{baselines,finetuned}/*.json rather than
    hardcoding them, so this can't silently drift from reports/qlora_training.md."""
    lines = ["| Model | Exact Match | Execution Accuracy |", "|---|---|---|"]
    baselines_dir = PROJECT_ROOT / "results" / "baselines" / "v2"
    for key, label in (("zero_shot", "Zero-shot"), ("few_shot", "Few-shot (5-shot)")):
        path = baselines_dir / f"{key}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)["metrics"]
        lines.append(f"| {label} | {metrics['exact_match']:.1%} | {metrics['execution_accuracy']:.1%} |")

    finetuned_dir = PROJECT_ROOT / "results" / "finetuned" / "v2"
    for filename, label, bold in (
        ("checkpoint-600_test_zero_shot.json", "Fine-tuned, zero-shot", False),
        ("checkpoint-600_test_few_shot.json", "**Fine-tuned + few-shot (this demo)**", True),
    ):
        path = finetuned_dir / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)["metrics"]
        em, ex = f"{metrics['exact_match']:.1%}", f"{metrics['execution_accuracy']:.1%}"
        if bold:
            em, ex = f"**{em}**", f"**{ex}**"
        lines.append(f"| {label} | {em} | {ex} |")
    lines.append("")
    lines.append(
        "All rows use the corrected schema prompts. The few-shot rows run about 1pt high: for 7.6% of "
        "test questions one example question is a paraphrase with the same answer (see reports/qlora_training.md)."
    )
    return "\n".join(lines)


BANNER = """\
# Text2SQL-Tune -- Demo

Pick a database, ask a question in plain English, and get back SQL -- generated live by a QLoRA fine-tuned \
Qwen2.5-Coder-3B, run against the real SQLite database so you see actual results, not just the query text.
"""

INFO_BANNER = """\
Every query below is generated live by the fine-tuned model (`checkpoints/v2/checkpoint-600`), combined with a \
few worked examples from the selected database as context -- the combination that measured best during \
evaluation (see the table below). SQL execution is real: every query actually runs against the corresponding \
database file. These databases are drawn from the 40-database held-out test split -- none of them were seen \
during training.
"""


def on_db_change(db_id: str):
    return (
        get_schema(db_id),
        gr.Radio(choices=get_example_questions(db_id), value=None),
        "",
        "",
        "",
        None,
    )


def on_generate(db_id: str, question: str):
    sql, note = predict_sql(db_id, question)
    if not sql:
        return "", note, None
    db_dir = _EXAMPLES_BY_DB[db_id][0]["db_dir"]
    columns, rows, error = run_query(db_id, db_dir, sql)
    if error:
        return sql, f"{note}\n\n**Execution error:** {error}", None
    return sql, note, pd.DataFrame(rows, columns=columns)


with gr.Blocks(title="Text2SQL-Tune -- Demo") as demo:
    gr.Markdown(BANNER)
    gr.Markdown(INFO_BANNER)
    with gr.Accordion("Model performance (measured on the held-out test set)", open=False):
        gr.Markdown(_load_metrics_table())

    with gr.Row():
        with gr.Column(scale=1):
            db_dropdown = gr.Dropdown(
                choices=[(DB_LABELS[d], d) for d in CURATED_DBS],
                value=CURATED_DBS[0],
                label="Database",
            )
            schema_box = gr.Code(value=get_schema(CURATED_DBS[0]), language="sql", label="Schema", interactive=False)
            examples_radio = gr.Radio(
                choices=get_example_questions(CURATED_DBS[0]),
                label="Example questions (click to use)",
            )
        with gr.Column(scale=1):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="Type a question about this database, or pick an example on the left...",
            )
            generate_btn = gr.Button("Generate SQL", variant="primary")
            sql_output = gr.Code(language="sql", label="Generated SQL", interactive=False)
            status_note = gr.Markdown()
            result_table = gr.Dataframe(label="Query result (live, from the real SQLite database)", wrap=True)

    db_dropdown.change(
        fn=on_db_change,
        inputs=db_dropdown,
        outputs=[schema_box, examples_radio, question_box, sql_output, status_note, result_table],
    )
    examples_radio.change(fn=lambda q: q, inputs=examples_radio, outputs=question_box)
    generate_btn.click(fn=on_generate, inputs=[db_dropdown, question_box], outputs=[sql_output, status_note, result_table])
    question_box.submit(fn=on_generate, inputs=[db_dropdown, question_box], outputs=[sql_output, status_note, result_table])


if __name__ == "__main__":
    demo.launch()
