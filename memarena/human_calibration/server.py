#!/usr/bin/env python3
"""LAN-accessible human-calibration server with multi-reviewer progress sync.

Serves the static index.html + sample.json from this directory and adds a
tiny JSON API for persisting labels across reviewers and devices.

Endpoints:
  GET  /                     -> index.html
  GET  /sample.json          -> the sample data
  GET  /api/state            -> {reviewers: [...], counts: {name: n}, total_items: N}
  GET  /api/labels/<name>    -> {reviewer: name, labels: {id: value}}
  POST /api/label            -> body: {reviewer, id, value}. value=0/1/"skip"/null(erase)

Labels are stored in docs/human_calibration/labels.json as:
  {"reviewers": {"alice": {"id1": 1, ...}, "bob": {...}}}

Each reviewer's labels are kept separate so inter-rater kappa can be computed
later — we do not aggregate or merge across reviewers.

Usage:
  python3 docs/human_calibration/server.py --port 8080
  # then open http://<your-lan-ip>:8080/ on any device on the same network
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
LABELS_FILE = ROOT / "labels.json"
SAMPLE_FILE = ROOT / "sample.json"
_lock = threading.Lock()


def _load_labels() -> dict:
    if LABELS_FILE.exists():
        try:
            return json.loads(LABELS_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"reviewers": {}}


def _save_labels(data: dict) -> None:
    tmp = LABELS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(LABELS_FILE)


def _count_sample() -> int:
    try:
        return len(json.loads(SAMPLE_FILE.read_text()).get("sample", []))
    except Exception:
        return 0


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/api/state":
            return self._api_state()
        if p.path.startswith("/api/labels/"):
            reviewer = p.path[len("/api/labels/"):]
            return self._api_labels(reviewer)
        return super().do_GET()

    def do_POST(self):
        p = urlparse(self.path)
        if p.path != "/api/label":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            payload = json.loads(body) if body else {}
            reviewer = str(payload.get("reviewer", "")).strip()[:64]
            qid = str(payload.get("id", "")).strip()[:128]
            if not reviewer or not qid:
                raise ValueError("reviewer and id are required")
            value = payload.get("value", None)
            if value not in (None, 0, 1, "skip"):
                raise ValueError(f"invalid value: {value!r}")
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=400)
            return

        with _lock:
            data = _load_labels()
            data.setdefault("reviewers", {})
            data["reviewers"].setdefault(reviewer, {})
            if value is None:
                data["reviewers"][reviewer].pop(qid, None)
            else:
                data["reviewers"][reviewer][qid] = value
            _save_labels(data)
            count = len(data["reviewers"][reviewer])
        self._send_json({"ok": True, "count": count})

    def _api_state(self):
        with _lock:
            data = _load_labels()
            rs = data.get("reviewers", {})
        # Count only completed (0 or 1); skips/null don't count as progress.
        def _completed(v):
            return sum(1 for x in v.values() if x in (0, 1))
        summary = {
            "reviewers": sorted(rs.keys()),
            "counts":   {name: _completed(v) for name, v in rs.items()},
            "counts_with_skips": {name: len(v) for name, v in rs.items()},
            "total_items": _count_sample(),
        }
        self._send_json(summary)

    def _api_labels(self, reviewer: str):
        reviewer = reviewer.strip()[:64]
        if not reviewer:
            self._send_json({"ok": False, "error": "reviewer required"}, status=400)
            return
        with _lock:
            data = _load_labels()
            labels = data.get("reviewers", {}).get(reviewer, {})
        self._send_json({"reviewer": reviewer, "labels": labels})

    def _send_json(self, obj: dict, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Silence noisy GET /api/state polls; keep everything else.
        line = (fmt % args) if args else str(fmt)
        if "/api/state" in line:
            return
        super().log_message(fmt, *args)


def _local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    os.chdir(ROOT)
    if not SAMPLE_FILE.exists():
        raise SystemExit(f"missing {SAMPLE_FILE}")
    if not LABELS_FILE.exists():
        _save_labels({"reviewers": {}})

    ip = _local_ip()
    print("Human-calibration server")
    print(f"  Root:   {ROOT}")
    print(f"  Labels: {LABELS_FILE}")
    print(f"  Local:  http://localhost:{args.port}/")
    print(f"  LAN:    http://{ip}:{args.port}/")
    print("Press Ctrl+C to stop.\n")

    httpd = HTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.server_close()


if __name__ == "__main__":
    main()
