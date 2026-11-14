#!/usr/bin/env python3
"""Run the tutor's web page on a laptop, without App Lab or the LED matrix.

    python3 app/math-matrix/python/dev_server.py            # http://127.0.0.1:7000

It serves assets/ and the same /api/ calls as main.py, through the same
tutor.py. The page draws the matrix frame it gets back, so you see what the
board would show. Standard library only.
"""
import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import tutor

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7000")))
    p.add_argument("--llm", default="http://127.0.0.1:8081")
    args = p.parse_args()
    copilot = tutor.Tutor(llm=args.llm)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path.startswith("/api/"):
                body, kind = json.dumps(copilot.handle(path[5:].strip("/").split("/"))).encode(), "application/json"
            else:
                name = os.path.normpath(path.lstrip("/") or "index.html")
                full = os.path.join(ASSETS, name)
                if name.startswith("..") or not os.path.isfile(full):
                    self.send_error(404)
                    return
                with open(full, "rb") as f:
                    body = f.read()
                kind = mimetypes.guess_type(full)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    print(f"{len(copilot.bank.problems)} problems in the bank. Open http://127.0.0.1:{args.port} (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
