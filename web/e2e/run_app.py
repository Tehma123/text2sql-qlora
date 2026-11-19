"""End-to-end test of the SQLocal app in a real browser, as a user would use it.

Serves web/app/ (or uses --url, e.g. the GitHub Pages deployment), opens it in
Edge/Chrome via Playwright, loads the model from the HF Hub, then:
  1. loads the sample CSV and asks the sample questions;
  2. checks that a non-SELECT query typed by hand is blocked;
  3. uploads a CSV with spaces and accents in its column names and asks about it.

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

ACCENT_QUESTIONS = [
    "What is the total revenue in each city?",
    "Which customer has the highest revenue?",
]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


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


def ask(page, question: str) -> dict:
    page.fill("#question", question)
    page.click("#ask")
    wait_for(page, f"window.__sqlocal.lastAsk?.question === {json.dumps(question)} && window.__sqlocal.lastAsk.done", 180, f"answer to {question!r}")
    return page.evaluate(
        """() => {
          const s = window.__sqlocal;
          const r = s.lastResult;
          return {
            question: s.lastAsk.question,
            attempts: s.lastAsk.attempts,
            ok: !!r && document.getElementById('error').hidden,
            columns: r?.columns ?? null,
            rows: r ? r.rows.slice(0, 5) : null,
            total_rows: r?.total ?? null,
            status: document.getElementById('result-status').textContent,
            error: document.getElementById('error').hidden ? null : document.getElementById('error').textContent,
          };
        }"""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="test a deployed copy instead of serving web/app locally")
    parser.add_argument("--model", choices=list(MODEL_INDEX), default="1.5b")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--channel", default="msedge")
    parser.add_argument("--profile", help="browser profile dir to reuse (keeps the cached model)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--label", default=None, help="results file name (default: local_<model> or pages_<model>)")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console: Vietnamese column names

    url = args.url or f"http://127.0.0.1:{args.port}/"
    server = None if args.url else serve(args.port)
    label = args.label or f"{'pages' if args.url else 'local'}_{args.model}"
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
        print(f"model ready in {report['model_load_s']} s (cached={report['model_was_cached']})", flush=True)

        # 1. Sample data and its questions.
        page.click("#sample-load")
        wait_for(page, "window.__sqlocal.table === 'sales'", 60, "sample load")
        report["sample_status"] = page.text_content("#data-status")
        questions = page.eval_on_selector_all("#examples button", "els => els.map(e => e.textContent)")
        for q in questions:
            r = ask(page, q)
            report["questions"].append({"data": "sample", **r})
            print(f"[{'ok' if r['ok'] else 'FAIL'}] {q} -> {r['attempts'][-1]['sql']} | rows={r['total_rows']} {r['rows'][:3] if r['rows'] else r['error']}", flush=True)

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
        for q in ACCENT_QUESTIONS:
            r = ask(page, q)
            report["questions"].append({"data": "accent", **r})
            print(f"[{'ok' if r['ok'] else 'FAIL'}] {q} -> {r['attempts'][-1]['sql']} | rows={r['total_rows']} {r['rows'][:3] if r['rows'] else r['error']}", flush=True)

        (E2E_DIR / "results").mkdir(exist_ok=True)
        page.screenshot(path=str(E2E_DIR / "results" / f"{label}.png"), full_page=True)
        context.close()
    if server:
        server.shutdown()
    tmp.cleanup()

    out = E2E_DIR / "results" / f"{label}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    ok = sum(q["ok"] for q in report["questions"])
    print(f"{ok}/{len(report['questions'])} questions returned a result; wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
