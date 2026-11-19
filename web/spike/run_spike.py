"""Drive the M0 browser spike: write prompts.json, open the page in Edge/Chrome, collect timings.

Needs `pip install playwright` (no browser download: it drives an installed
Edge/Chrome via --channel) and serve.py running in another process.

    python web/spike/run_spike.py --n 5 [--channel msedge] [--headless]

The page is opened twice in one browser profile so the second load shows the
cached (Cache API) load time. Results go to web/spike/results/<model>_<dtype>.json.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SPIKE_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text2sql.prompts.templates import build_messages  # noqa: E402


def write_prompts(n: int, reference: Path) -> None:
    dev = json.loads((PROJECT_ROOT / "data/processed/dev.json").read_text(encoding="utf-8"))[:n]
    ref = json.loads(reference.read_text(encoding="utf-8"))["results"][:n]
    prompts = [
        {"question": ex["question"], "messages": build_messages(ex["question"], ex["schema"]), "reference_pred": r["pred"]}
        for ex, r in zip(dev, ref)
    ]
    (SPIKE_DIR / "prompts.json").write_text(json.dumps(prompts, indent=1), encoding="utf-8")


def run_page(page, url: str, timeout_s: int) -> dict:
    page.goto(url)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        report = page.evaluate("window.__spike ?? null")
        if report and report.get("done"):
            return report
        time.sleep(2)
    raise TimeoutError(f"page did not finish within {timeout_s}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--model", default="onnx-600-genai", help="dir served by serve.py, or an owner/name Hub repo id")
    parser.add_argument("--dtype", default="q4f16")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--channel", default="msedge")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--reference", default="checkpoints/v2/onnx-600-genai/verify_dev_model_q4f16.json",
        help="onnxruntime results for the same model, to compare generated SQL",
    )
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    write_prompts(args.n, PROJECT_ROOT / args.reference)
    url = f"http://127.0.0.1:{args.port}/?model={args.model}&dtype={args.dtype}&n={args.n}"
    runs = {}
    # A fresh on-disk profile rather than browser.new_context(): new_context is
    # incognito-like, and its storage quota is too small to cache a 1.7 GB model.
    with sync_playwright() as p, tempfile.TemporaryDirectory(prefix="spike-profile-") as profile:
        context = p.chromium.launch_persistent_context(
            profile, channel=args.channel, headless=args.headless,
            args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
        )
        page = context.new_page()
        page.on("console", lambda m: print(f"[browser] {m.text}", flush=True) if m.type in ("log", "error") else None)
        runs["first_load"] = run_page(page, url, args.timeout)
        if not runs["first_load"].get("error"):
            runs["cached_load"] = run_page(page, url, args.timeout)
        context.close()

    out_dir = SPIKE_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.model.replace('/', '__')}_{args.dtype}.json"
    out.write_text(json.dumps(runs, indent=1), encoding="utf-8")
    for name, r in runs.items():
        print(f"{name}: load={r.get('load_ms', 0) / 1000:.1f}s error={r.get('error')}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
