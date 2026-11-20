"""SQLocal accuracy on real CSV files: 50 English questions on 5 public datasets.

The datasets are seaborn's example data (tips, titanic, penguins, mpg,
diamonds), downloaded from a pinned commit of github.com/mwaskom/seaborn-data
and checked against SHA-256 hashes (not committed to this repo). Each question
has one or more gold SQL queries; the expected answer is what DuckDB returns
for them on the same file. The questions run through the real app in a browser
(upload, ask, automatic fixes), exactly as a user would.

    python web/e2e/benchmark.py [--model 3b] [--url https://...] [--profile DIR]

A result is correct when it has as many rows as the expected answer and every
expected row's values appear in a different result row (extra columns are
fine, order is ignored; numbers match within rounding). Results go to
web/e2e/results/benchmark_<label>.json.
"""

import argparse
import hashlib
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import run_app

E2E_DIR = Path(__file__).resolve().parent
DATA_DIR = E2E_DIR / "data"
SEABORN_COMMIT = "71e2436a092d714350de0fc409ca8a8714e7e78f"
DATASETS = {
    "tips": "e54cc4d2ce1bff65d32ca60b3e4b802e06bde1d7e7caf6f796f6bf7370e863b0",
    "titanic": "81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2",
    "penguins": "e07636bd8af74260099ea2f8678e2eabbf35def579940cc76f67061ee16c06c1",
    "mpg": "c14b8b855ea7ee86cb9736bf8caaf281c4685ca08826f3eb2acaccaaf40f0d5a",
    "diamonds": "9574730b03aba241d899c4a97511c5061b19358fab89510774fb6c24168345c4",
}

# (question, [gold SQL, ...]) per dataset; the table is named after the file.
# Several gold queries mean several acceptable readings (a rate as 0.74 or 74%).
QUESTIONS = {
    "tips": [
        ("What is the average tip?", ["SELECT avg(tip) FROM tips"]),
        ("How many bills were paid by smokers?", ["SELECT count(*) FROM tips WHERE smoker"]),
        ("What is the total bill amount for each day?", ["SELECT day, sum(total_bill) FROM tips GROUP BY day"]),
        ("Which day has the highest average tip?", ["SELECT day FROM tips GROUP BY day ORDER BY avg(tip) DESC LIMIT 1"]),
        ("What is the largest party size?", ["SELECT max(size) FROM tips"]),
        ("How many bills were paid at dinner?", ["SELECT count(*) FROM tips WHERE time = 'Dinner'"]),
        ("What is the average tip for each sex?", ["SELECT sex, avg(tip) FROM tips GROUP BY sex"]),
        ("What is the biggest tip given at lunch?", ["SELECT max(tip) FROM tips WHERE time = 'Lunch'"]),
        ("How many parties had more than 4 people?", ["SELECT count(*) FROM tips WHERE size > 4"]),
        ("What is the average tip as a percentage of the total bill?",
         ["SELECT avg(tip / total_bill) * 100 FROM tips", "SELECT avg(tip / total_bill) FROM tips",
          "SELECT sum(tip) / sum(total_bill) * 100 FROM tips", "SELECT sum(tip) / sum(total_bill) FROM tips"]),
    ],
    "titanic": [
        ("How many passengers survived?", ["SELECT count(*) FROM titanic WHERE survived = 1"]),
        ("What was the average age of the passengers?", ["SELECT avg(age) FROM titanic"]),
        ("How many passengers were in each class?",
         ["SELECT class, count(*) FROM titanic GROUP BY class", "SELECT pclass, count(*) FROM titanic GROUP BY pclass"]),
        ("What was the survival rate for women?",
         ["SELECT avg(survived) FROM titanic WHERE sex = 'female'", "SELECT avg(survived) * 100 FROM titanic WHERE sex = 'female'"]),
        ("What was the highest fare paid?", ["SELECT max(fare) FROM titanic"]),
        ("How many passengers embarked at Southampton?", ["SELECT count(*) FROM titanic WHERE embark_town = 'Southampton'"]),
        ("Which deck had the most passengers?",
         ["SELECT deck FROM titanic WHERE deck IS NOT NULL GROUP BY deck ORDER BY count(*) DESC LIMIT 1"]),
        ("What was the average fare in first class?", ["SELECT avg(fare) FROM titanic WHERE pclass = 1"]),
        ("How many passengers were younger than 18?", ["SELECT count(*) FROM titanic WHERE age < 18"]),
        ("How many passengers travelled alone?", ["SELECT count(*) FROM titanic WHERE alone"]),
    ],
    "penguins": [
        ("How many penguins are there of each species?", ["SELECT species, count(*) FROM penguins GROUP BY species"]),
        ("What is the average body mass of Gentoo penguins?", ["SELECT avg(body_mass_g) FROM penguins WHERE species = 'Gentoo'"]),
        ("Which island has the most penguins?", ["SELECT island FROM penguins GROUP BY island ORDER BY count(*) DESC LIMIT 1"]),
        ("What is the longest flipper length?", ["SELECT max(flipper_length_mm) FROM penguins"]),
        ("What is the average bill length for each species?", ["SELECT species, avg(bill_length_mm) FROM penguins GROUP BY species"]),
        ("How many female penguins are there?", ["SELECT count(*) FROM penguins WHERE sex = 'FEMALE'"]),
        ("Which species is the heaviest on average?",
         ["SELECT species FROM penguins GROUP BY species ORDER BY avg(body_mass_g) DESC LIMIT 1"]),
        ("How many penguins live on Dream island?", ["SELECT count(*) FROM penguins WHERE island = 'Dream'"]),
        ("What is the smallest body mass recorded?", ["SELECT min(body_mass_g) FROM penguins"]),
        ("How many different islands are there?", ["SELECT count(DISTINCT island) FROM penguins"]),
    ],
    "mpg": [
        ("What is the average mpg of all cars?", ["SELECT avg(mpg) FROM mpg"]),
        ("How many cars have 8 cylinders?", ["SELECT count(*) FROM mpg WHERE cylinders = 8"]),
        ("Which car has the highest mpg?", ["SELECT name FROM mpg ORDER BY mpg DESC LIMIT 1"]),
        ("What is the average horsepower for each origin?", ["SELECT origin, avg(horsepower) FROM mpg GROUP BY origin"]),
        ("How many cars are from Japan?", ["SELECT count(*) FROM mpg WHERE origin = 'japan'"]),
        ("What is the heaviest car?", ["SELECT name FROM mpg ORDER BY weight DESC LIMIT 1"]),
        ("What is the average weight of cars made in 1970?", ["SELECT avg(weight) FROM mpg WHERE model_year = 70"]),
        ("How many cars are there for each number of cylinders?", ["SELECT cylinders, count(*) FROM mpg GROUP BY cylinders"]),
        ("What is the average acceleration of cars with more than 150 horsepower?",
         ["SELECT avg(acceleration) FROM mpg WHERE horsepower > 150"]),
        ("Which origin has the most cars?", ["SELECT origin FROM mpg GROUP BY origin ORDER BY count(*) DESC LIMIT 1"]),
    ],
    "diamonds": [
        ("What is the average price of a diamond?", ["SELECT avg(price) FROM diamonds"]),
        ("How many diamonds have an Ideal cut?", ["SELECT count(*) FROM diamonds WHERE cut = 'Ideal'"]),
        ("What is the carat weight of the most expensive diamond?", ["SELECT carat FROM diamonds ORDER BY price DESC LIMIT 1"]),
        ("What is the average price for each cut?", ["SELECT cut, avg(price) FROM diamonds GROUP BY cut"]),
        ("How many diamonds weigh more than 2 carats?", ["SELECT count(*) FROM diamonds WHERE carat > 2"]),
        ("Which color has the highest average price?", ["SELECT color FROM diamonds GROUP BY color ORDER BY avg(price) DESC LIMIT 1"]),
        ("What is the cheapest price for a diamond with clarity IF?", ["SELECT min(price) FROM diamonds WHERE clarity = 'IF'"]),
        ("What is the average table value?", ['SELECT avg("table") FROM diamonds']),
        ("How many diamonds are there for each clarity?", ["SELECT clarity, count(*) FROM diamonds GROUP BY clarity"]),
        ("What is the average carat of Premium diamonds?", ["SELECT avg(carat) FROM diamonds WHERE cut = 'Premium'"]),
    ],
}


def fetch_datasets() -> dict[str, Path]:
    DATA_DIR.mkdir(exist_ok=True)
    paths = {}
    for name, sha in DATASETS.items():
        path = DATA_DIR / f"{name}.csv"
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            url = f"https://raw.githubusercontent.com/mwaskom/seaborn-data/{SEABORN_COMMIT}/{name}.csv"
            data = urllib.request.urlopen(url, timeout=60).read()
            if hashlib.sha256(data).hexdigest() != sha:
                raise RuntimeError(f"{name}.csv does not match its pinned SHA-256")
            path.write_bytes(data)
        paths[name] = path
    return paths


def to_text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def expected_answers(paths: dict[str, Path]) -> dict[str, list[list[list[str]]]]:
    import duckdb

    con = duckdb.connect()
    for name, path in paths.items():
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv_auto('{path.as_posix()}')")
    return {q: [[[to_text(v) for v in row] for row in con.execute(sql).fetchall()] for sql in golds]
            for qs in QUESTIONS.values() for q, golds in qs}


def same_value(result: str | None, expected: str | None) -> bool:
    if result == expected:
        return True
    try:
        a, b = float(result), float(expected)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(1e-4, 1e-6 * abs(b))  # the app rounds to 4 decimals


def matches(rows, accepted) -> bool:
    if rows is None:
        return False
    for expected in accepted:
        if len(rows) != len(expected):
            continue
        free = [list(r) for r in rows]
        for exp in expected:
            hit = next((r for r in free if all(any(same_value(v, e) for v in r) for e in exp)), None)
            if hit is None:
                break
            free.remove(hit)
        else:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="test a deployed copy instead of serving web/app locally")
    parser.add_argument("--model", choices=list(run_app.MODEL_INDEX), default="1.5b")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--channel", default="msedge")
    parser.add_argument("--profile", help="browser profile dir to reuse (keeps the cached model)")
    parser.add_argument("--only", nargs="+", choices=list(QUESTIONS), help="run only these datasets")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths = fetch_datasets()
    expected = expected_answers(paths)
    url = args.url or f"http://127.0.0.1:{args.port}/"
    server = None if args.url else run_app.serve(args.port)
    label = f"{'pages' if args.url else 'local'}_{args.model}"
    report = {"url": url, "model": args.model, "seaborn_commit": SEABORN_COMMIT, "questions": []}

    tmp = tempfile.TemporaryDirectory(prefix="sqlocal-profile-")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            args.profile or tmp.name, channel=args.channel, args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"])
        page = context.new_page()
        page.goto(url)
        page.wait_for_function("document.getElementById('model-select').options.length > 0")
        page.select_option("#model-select", index=run_app.MODEL_INDEX[args.model])
        time.sleep(1)
        if page.is_visible("#model-load") and page.is_enabled("#model-load"):
            page.click("#model-load")
        run_app.wait_for(page, "window.__sqlocal.modelReady", 900, "model load")

        for name in args.only or QUESTIONS:
            page.set_input_files("#file-input", str(paths[name]))
            run_app.wait_for(page, f"window.__sqlocal.table === {json.dumps(name)}", 120, f"{name} load")
            for question, _ in QUESTIONS[name]:
                r = run_app.ask(page, question)
                r["correct"] = matches(r["rows"] if r["ok"] else None, expected[question])
                r["expected"] = expected[question][0][:5]
                report["questions"].append({"dataset": name, **r})
                verdict = "correct" if r["correct"] else ("WRONG" if r["ok"] else "ERROR")
                print(f"[{verdict}] {name}: {question}\n    {r['attempts'][-1]['sql']}"
                      f"\n    got {r['rows'][:3] if r['rows'] else r['error']!r:.200} | expected {r['expected'][:3]}", flush=True)
        context.close()
    if server:
        server.shutdown()
    tmp.cleanup()

    qs = report["questions"]
    by_dataset = {}
    for q in qs:
        d = by_dataset.setdefault(q["dataset"], {"questions": 0, "correct": 0})
        d["questions"] += 1
        d["correct"] += q["correct"]
    report["summary"] = {
        "questions": len(qs),
        "correct": sum(q["correct"] for q in qs),
        "ran_but_wrong": sum(q["ok"] and not q["correct"] for q in qs),
        "failed_to_run": sum(not q["ok"] for q in qs),
        "first_attempt_failed": sum(bool(q["attempts"][0]["error"]) for q in qs),
        "rescued_by_repair": sum(q["ok"] and len(q["attempts"]) > 1 for q in qs),
        "rescued_and_correct": sum(q["correct"] and len(q["attempts"]) > 1 for q in qs),
        "zero_rows_with_hints": sum(bool(q.get("hints")) for q in qs),
        "median_sql_seconds": sorted(q["attempts"][-1]["totalMs"] for q in qs)[len(qs) // 2] / 1000,
        "by_dataset": by_dataset,
    }
    out = E2E_DIR / "results" / f"benchmark_{label}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
