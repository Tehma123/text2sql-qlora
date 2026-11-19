"""Local static server for the M0 browser spike.

Serves web/spike/ at / and an exported ONNX model directory at /models/<name>/,
bound to 127.0.0.1 only. Kept separate from the repo root so nothing else in
the project (e.g. .env) is reachable over HTTP.

    python web/spike/serve.py --model checkpoints/v2/onnx-600-genai [more model dirs] --port 8765
"""

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

SPIKE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SPIKE_DIR.parents[1]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, model_dirs: dict[str, Path], **kwargs):
        self.model_dirs = model_dirs
        super().__init__(*args, directory=str(SPIKE_DIR), **kwargs)

    def translate_path(self, path):
        url_path = unquote(urlparse(path).path)
        for name, model_dir in self.model_dirs.items():
            prefix = f"/models/{name}/"
            if url_path.startswith(prefix):
                target = (model_dir / url_path[len(prefix) :]).resolve()
                if model_dir in target.parents:
                    return str(target)
                return str(model_dir / "__forbidden__")
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # keep the console quiet except for errors
        if args and str(args[1])[:1] in "45":
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", nargs="+", default=["checkpoints/v2/onnx-600-genai"],
        help="model dirs, served as /models/<dir name>/; NAME=DIR serves DIR as /models/NAME/",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    model_dirs = {}
    for m in args.model:
        name, sep, path = m.partition("=")
        d = (PROJECT_ROOT / (path if sep else m)).resolve()
        model_dirs[name if sep else d.name] = d
    handler = functools.partial(Handler, model_dirs=model_dirs)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Serving http://127.0.0.1:{args.port}/", flush=True)
    for name, d in model_dirs.items():
        print(f"  /models/{name}/ -> {d}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
