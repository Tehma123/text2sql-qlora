"""Checks every row of the "Xử lý lỗi & UX" table in SQLocal.md in a real browser.

Uses hand-typed SQL ("Run SQL"), so no model is loaded, except where a row is
about the model itself. Rows covered here:

  - no WebGPU         -> ?device=wasm shows the CPU-mode notice and offers the q4 download
  - first download    -> nothing is downloaded before the button is pressed; the size is shown
  - SQL error         -> the DuckDB error is shown and the SQL can be fixed and re-run
  - not SELECT/WITH   -> blocked before running
  - 0 rows            -> says "0 rows" and lists real values of the filtered column
  - odd column names  -> renamed for SQL, original names in the table and the CSV export
  - large file        -> 1,000,000-row CSV: loads, shows 100 rows, the CSV export has every row
  - ambiguous question-> a plain-English line describes what the SQL does

The automatic fixes (model rewrites a failing query) and the CPU model itself
are exercised by run_app.py --device wasm and benchmark.py.

    python web/e2e/error_cases.py [--url https://...]
"""

import argparse
import csv
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import run_app

E2E_DIR = Path(__file__).resolve().parent
BIG_ROWS = 1_000_000


def write_big_csv(path: Path) -> None:
    rng = random.Random(1)
    regions = ["North", "South", "East", "West"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "region", "amount"])
        for i in range(BIG_ROWS):
            w.writerow([i, regions[i % 4], round(rng.random() * 100, 2)])


def run_sql(page, sql: str) -> dict:
    page.fill("#sql", sql)
    page.evaluate("window.__sqlocal.lastResult = null")
    page.click("#run")
    page.wait_for_function("!window.__sqlocal.busy")
    time.sleep(0.3)  # 0-row value hints are filled in after the table
    return page.evaluate("""() => ({
        sql: document.getElementById('sql').value,
        status: document.getElementById('result-status').textContent,
        error: document.getElementById('error').hidden ? null : document.getElementById('error').textContent,
        hints: document.getElementById('hints').hidden ? null : document.getElementById('hints').textContent,
        explain: document.getElementById('explain').hidden ? null : document.getElementById('explain').textContent,
        chart: document.getElementById('chart').hidden ? null : document.querySelector('#chart svg')?.getAttribute('aria-label'),
        headers: [...document.querySelectorAll('#result th')].map(th => th.textContent),
        shown_rows: document.querySelectorAll('#result tbody tr').length,
        total: window.__sqlocal.lastResult?.total ?? null,
    })""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    url = args.url or f"http://127.0.0.1:{args.port}/"
    server = None if args.url else run_app.serve(args.port)
    checks = []

    def check(name, ok, detail):
        checks.append({"case": name, "ok": bool(ok), "detail": detail})
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    with sync_playwright() as p, tempfile.TemporaryDirectory(prefix="sqlocal-profile-") as profile, \
            tempfile.TemporaryDirectory() as d:
        context = p.chromium.launch_persistent_context(
            profile, channel=args.channel, args=["--enable-unsafe-webgpu"], accept_downloads=True)
        page = context.new_page()
        model_requests = []
        page.on("request", lambda r: model_requests.append(r.url) if "onnx_data" in r.url else None)

        # No WebGPU -> CPU mode (forced with ?device=wasm; the real check is navigator.gpu / shader-f16).
        page.goto(url + "?device=wasm")
        page.wait_for_function("document.getElementById('model-select').options.length > 0")
        banner = page.text_content("#banner")
        options = page.eval_on_selector_all("#model-select option", "els => els.map(e => [e.textContent, e.disabled])")
        check("no WebGPU: CPU fallback offered", "CPU" in banner and "half a minute" in banner and "982 MB" in page.text_content("#model-load")
              and options[1][1], f"banner={banner[:90]!r}… options={options}")

        # First download: nothing fetched before the click, size on the button.
        page.goto(url)
        page.wait_for_function("document.getElementById('model-select').options.length > 0")
        time.sleep(2)
        check("first download asks first", page.is_visible("#model-load") and "877 MB" in page.text_content("#model-load")
              and not model_requests, f"button={page.text_content('#model-load')!r}, model requests={len(model_requests)}")

        # the app loads the sample data by itself on first open; wait for that instead of racing it
        page.wait_for_function("window.__sqlocal.table === 'sales'", timeout=60_000)
        if page.evaluate("window.__sqlocal.table") != "sales":
            page.click("#sample-load")
        run_app.wait_for(page, "window.__sqlocal.table === 'sales'", 60, "sample")
        page.evaluate("document.getElementById('result-card').hidden = false")

        r = run_sql(page, "SELECT regoin, sum(revenue) FROM sales GROUP BY regoin")
        check("SQL error shown", r["error"] and "regoin" in r["error"], (r["error"] or "")[:120])
        r = run_sql(page, "SELECT region, sum(revenue) FROM sales GROUP BY region ORDER BY 2 DESC")
        check("fixed SQL re-runs, with chart and explanation", r["total"] == 4 and not r["error"] and r["chart"] and r["explain"],
              f"rows={r['total']} chart={r['chart']!r} explain={r['explain']!r}")

        r = run_sql(page, "DROP TABLE sales")
        check("non-SELECT blocked", r["error"] and "SELECT" in r["error"], r["error"])
        r = run_sql(page, "WITH x AS (SELECT 1) INSERT INTO sales SELECT * FROM sales")
        check("write hidden in WITH blocked", r["error"] and "INSERT" in r["error"], r["error"])

        r = run_sql(page, "SELECT * FROM sales WHERE region = 'Northeast'")
        check("0 rows: says so and lists real values", r["total"] == 0 and "0 rows" in r["status"] and r["hints"] and "North" in r["hints"],
              f"status={r['status']!r} hints={r['hints']!r}")
        r = run_sql(page, "SELECT count(*) FROM sales WHERE quantity = 100")
        check("aggregate over nothing: number range shown", r["hints"] and "numbers from 1 to 12" in r["hints"], r["hints"])
        r = run_sql(page, "SELECT count(*) FROM sales WHERE region = 'north'")
        check("value case fixed to match the data", "region = 'North'" in r["sql"] and r["explain"] and "Changed" in r["explain"],
              f"sql={r['sql']!r} note={r['explain']!r}")

        r = run_sql(page, "SELECT strftime('%Y-%m', order_date) AS month, sum(revenue) FROM sales GROUP BY 1 ORDER BY 1")
        check("time result drawn as a line", r["chart"] and r["chart"].startswith("Line"), r["chart"])

        # Odd column names: renamed for SQL, original names shown and exported.
        accent = Path(d) / "khách hàng.csv"
        run_app.write_accent_csv(accent)
        page.set_input_files("#file-input", str(accent))
        run_app.wait_for(page, "window.__sqlocal.table === 'khach_hang'", 60, "accent csv")
        r = run_sql(page, "SELECT thanh_pho, sum(doanh_thu_vnd) FROM khach_hang GROUP BY thanh_pho")
        with page.expect_download() as dl:
            page.click("#download")
        exported = dl.value.path().read_text(encoding="utf-8").splitlines()
        check("odd column names: original headers in table and CSV", "Thành phố" in r["headers"] and exported[0].startswith("Thành phố"),
              f"headers={r['headers']} csv header={exported[0]!r}")

        # Large file.
        big = Path(d) / "big.csv"
        write_big_csv(big)
        t0 = time.time()
        page.set_input_files("#file-input", str(big))
        run_app.wait_for(page, "window.__sqlocal.table === 'big'", 300, "big csv")
        load_s = time.time() - t0
        r = run_sql(page, "SELECT * FROM big")
        with page.expect_download(timeout=300_000) as dl:
            page.click("#download")
        lines = sum(1 for _ in open(dl.value.path(), encoding="utf-8"))
        check("large file: 100 rows shown, full CSV export",
              r["total"] == BIG_ROWS and r["shown_rows"] == 100 and lines == BIG_ROWS + 1,
              f"{big.stat().st_size / 1e6:.0f} MB loaded in {load_s:.1f} s; status={r['status']!r}; exported lines={lines:,}")
        r = run_sql(page, "SELECT region, avg(amount) FROM big GROUP BY region")
        check("large file: aggregate query", r["total"] == 4, r["status"])
        context.close()
    if server:
        server.shutdown()

    out = E2E_DIR / "results" / f"error_cases_{'pages' if args.url else 'local'}.json"
    out.write_text(json.dumps(checks, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{sum(c['ok'] for c in checks)}/{len(checks)} checks passed; wrote {out}")


if __name__ == "__main__":
    main()
