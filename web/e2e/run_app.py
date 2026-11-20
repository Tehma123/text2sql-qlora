"""End-to-end test of the SQLocal app in a real browser, as a user would use it.

Serves web/app/ (or uses --url, e.g. the GitHub Pages deployment), opens it in
Edge/Chrome via Playwright, loads the model from the HF Hub, then:
  1. loads the sample CSV and asks 10 questions;
  2. checks that a non-SELECT query typed by hand is blocked;
  3. uploads a CSV with spaces and accents in its column names and asks 2 questions.
Each answer is compared with the expected rows, computed independently from
the CSV (see SAMPLE_CHECKS), so the run also measures answer accuracy.

    python web/e2e/run_app.py [--url https://...] [--model 3b] [--profile DIR] [--headless]

--profile reuses a browser profile (and its cached model) between runs; by
default a fresh profile is used, so the run includes the first download.
Results go to web/e2e/results/<label>.json.
"""

import argparse
import csv
import functools
import http.server
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

E2E_DIR = Path(__file__).resolve().parent
APP_DIR = E2E_DIR.parent / "app"
MODEL_INDEX = {"1.5b": 0, "3b": 1}

# (question, accepted answers). An answer is a list of rows; a result matches
# when it has the same number of rows and each expected row's values appear in
# a different result row (extra columns such as the summed value are fine, and
# row order is ignored). Values use the app's formatting (2697.0 -> "2697").
SAMPLE_CHECKS = [
    ("What is the total revenue for each region?",
     [[["East", "33383.5"], ["South", "45050"], ["North", "37503.5"], ["West", "35295.5"]]]),
    ("Which 3 products sold the most units?", [[["Pen Set"], ["Keyboard"], ["Backpack"]]]),
    ("How many orders came from new customers?", [[["120"]]]),
    ("What is the average order revenue per category?",
     [[["Furniture", "462.9787"], ["Electronics", "827.2288"], ["Stationery", "114.767"]]]),
    ("How many orders were placed in March 2024?", [[["46"]]]),
    ("What was the total revenue for each month?",
     [[["2024-01", "20357.5"], ["2024-02", "24672"], ["2024-03", "30114"], ["2024-04", "17088.5"],
       ["2024-05", "33187.5"], ["2024-06", "25813"]]]),
    ("Which region had the highest total revenue from Furniture?", [[["South"]]]),
    ("What is the largest single order revenue?", [[["2697"]]]),
    ("What percentage of orders are from returning customers?", [[["60"]], [["0.6"]]]),
    ("How many different products were sold in the West region?", [[["10"]]]),
]
ACCENT_CHECKS = [
    ("What is the total revenue in each city?", [[["Hà Nội", "1650000"], ["Đà Nẵng", "800000"], ["TP HCM", "3290000"]]]),
    ("Which customer has the highest revenue?", [[["Dũng"]]]),
]


def matches(rows, accepted) -> bool:
    if rows is None:
        return False
    for expected in accepted:
        if len(rows) != len(expected):
            continue
        free = [list(r) for r in rows]
        for exp in expected:
            hit = next((r for r in free if all(v in r for v in exp)), None)
            if hit is None:
                break
            free.remove(hit)
        else:
            return True
    return False


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def end_headers(self):
        # A reused --profile would otherwise keep running the previous app.js.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve(port: int) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(QuietHandler, directory=str(APP_DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def write_accent_csv(path: Path) -> None:
    rows = [
        ["Tên khách hàng", "Thành phố", "Doanh thu (VND)", "Ngày mua"],
        ["An", "Hà Nội", 1200000, "2024-03-01"],
        ["Bình", "Đà Nẵng", 800000, "2024-03-02"],
        ["Chi", "Hà Nội", 450000, "2024-03-05"],
        ["Dũng", "TP HCM", 2300000, "2024-03-07"],
        ["Em", "TP HCM", 990000, "2024-03-09"],
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def wait_for(page, js: str, timeout_s: float, what: str):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        value = page.evaluate(js)
        if value:
            return value
        error = page.evaluate("document.getElementById('banner').classList.contains('err') && document.getElementById('banner').textContent")
        if error:
            raise RuntimeError(f"{what}: page shows an error: {error}")
        time.sleep(0.5)
    raise TimeoutError(f"{what} did not happen within {timeout_s}s")


ASK_TIMEOUT_S = 180


def ask(page, question: str) -> dict:
    page.fill("#question", question)
    page.click("#ask")
    wait_for(page, f"window.__sqlocal.lastAsk?.question === {json.dumps(question)} && window.__sqlocal.lastAsk.done", ASK_TIMEOUT_S, f"answer to {question!r}")
    return page.evaluate(
        """() => {
          const s = window.__sqlocal;
          const r = s.lastResult;
          return {
            question: s.lastAsk.question,
            attempts: s.lastAsk.attempts,
            ok: !!r && document.getElementById('error').hidden,
            columns: r?.columns ?? null,
            rows: r ? r.rows.slice(0, 50) : null,
            total_rows: r?.total ?? null,
            status: document.getElementById('result-status').textContent,
            error: document.getElementById('error').hidden ? null : document.getElementById('error').textContent,
            explain: document.getElementById('explain').hidden ? null : document.getElementById('explain').textContent,
            chart: document.getElementById('chart').hidden ? null : document.querySelector('#chart svg')?.getAttribute('aria-label'),
            hints: document.getElementById('hints').hidden ? null : document.getElementById('hints').textContent,
          };
        }"""
    )


def check(page, report: dict, data: str, question: str, accepted) -> None:
    r = ask(page, question)
    r["correct"] = matches(r["rows"] if r["ok"] else None, accepted)
    report["questions"].append({"data": data, **r})
    verdict = "correct" if r["correct"] else ("WRONG" if r["ok"] else "ERROR")
    shown = r["rows"][:3] if r["rows"] else r["error"]
    print(f"[{verdict}] {question}\n    {r['attempts'][-1]['sql']}\n    rows={r['total_rows']} {shown}", flush=True)
    for key in ("explain", "chart", "hints"):
        if r.get(key):
            print(f"    {key}: {r[key]}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="test a deployed copy instead of serving web/app locally")
    parser.add_argument("--model", choices=list(MODEL_INDEX), default="1.5b")
    parser.add_argument("--device", choices=["webgpu", "wasm"], default="webgpu", help="wasm: force the CPU fallback (?device=wasm)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--channel", default="msedge")
    parser.add_argument("--profile", help="browser profile dir to reuse (keeps the cached model)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--label", default=None, help="results file name (default: local_<model> or pages_<model>)")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console: Vietnamese column names

    url = args.url or f"http://127.0.0.1:{args.port}/"
    if args.device == "wasm":
        url += "?device=wasm"
        global ASK_TIMEOUT_S
        ASK_TIMEOUT_S = 900
    server = None if args.url else serve(args.port)
    label = args.label or f"{'pages' if args.url else 'local'}_{args.model}{'_wasm' if args.device == 'wasm' else ''}"
    report = {"url": url, "model": args.model, "questions": []}

    tmp = tempfile.TemporaryDirectory(prefix="sqlocal-profile-")
    profile = args.profile or tmp.name
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            profile, channel=args.channel, headless=args.headless,
            args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"], accept_downloads=True,
        )
        page = context.new_page()
        page.on("console", lambda m: print(f"[browser] {m.text}", flush=True) if m.type == "error" else None)
        page.on("pageerror", lambda e: print(f"[pageerror] {e}", flush=True))
        page.goto(url)

        # Model: choose, download if not cached, wait until ready.
        page.wait_for_function("document.getElementById('model-select').options.length > 0")
        page.select_option("#model-select", index=MODEL_INDEX[args.model])
        time.sleep(1)
        t0 = time.time()
        if page.is_visible("#model-load") and page.is_enabled("#model-load"):
            report["model_was_cached"] = False
            page.click("#model-load")
        else:
            report["model_was_cached"] = True
        wait_for(page, "window.__sqlocal.modelReady", 900, "model load")
        report["model_load_s"] = round(time.time() - t0, 1)
        report["model_status"] = page.text_content("#model-status")
        report["cross_origin_isolated"] = page.evaluate("window.crossOriginIsolated")
        print(f"cross-origin isolated: {report['cross_origin_isolated']}", flush=True)
        print(f"model ready in {report['model_load_s']} s (cached={report['model_was_cached']})", flush=True)

        # 1. Sample data and its questions.
        # the app loads the sample data by itself on first open; wait for that instead of racing it
        page.wait_for_function("window.__sqlocal.table === 'sales'", timeout=60_000)
        if page.evaluate("window.__sqlocal.table") != "sales":
            page.click("#sample-load")
        wait_for(page, "window.__sqlocal.table === 'sales'", 60, "sample load")
        report["sample_status"] = page.text_content("#data-status")
        for q, accepted in SAMPLE_CHECKS:
            check(page, report, "sample", q, accepted)

        # 2. Hand-typed non-SELECT is blocked.
        page.fill("#sql", "DELETE FROM sales")
        page.click("#run")
        page.wait_for_selector("#error:not([hidden])")
        report["delete_blocked"] = page.text_content("#error")
        count = page.evaluate("window.__sqlocal.conn.query('SELECT count(*) FROM sales').then(t => Number(t.getChildAt(0).get(0)))")
        report["rows_after_delete_attempt"] = count
        print(f"DELETE blocked: {report['delete_blocked']!r}; rows still {count}", flush=True)

        # 3. CSV with accents/spaces in column names, uploaded through the file input.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "khách hàng.csv"
            write_accent_csv(path)
            page.set_input_files("#file-input", str(path))
            wait_for(page, "window.__sqlocal.table && window.__sqlocal.table !== 'sales'", 60, "accent CSV load")
        report["accent_status"] = page.text_content("#data-status")
        report["accent_schema"] = page.evaluate("window.__sqlocal.schema")
        for q, accepted in ACCENT_CHECKS:
            check(page, report, "accent", q, accepted)

        (E2E_DIR / "results").mkdir(exist_ok=True)
        page.screenshot(path=str(E2E_DIR / "results" / f"{label}.png"), full_page=True)
        context.close()
    if server:
        server.shutdown()
    tmp.cleanup()

    qs = report["questions"]
    report["summary"] = {
        "questions": len(qs),
        "ran": sum(q["ok"] for q in qs),
        "correct": sum(q["correct"] for q in qs),
        "fixed_by_repair": sum(q["correct"] and len(q["attempts"]) > 1 for q in qs),
    }
    out = E2E_DIR / "results" / f"{label}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"summary: {report['summary']}; wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
