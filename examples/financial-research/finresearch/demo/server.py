"""GEPA demo server. Standard library only.

  python -m finresearch.demo.server [--port 8770] [--snapshot PATH]
      (default: the packaged snapshot, finresearch.config.SNAPSHOT)

  GET  /                      static/demo.html
  GET  /static/<file>         other files next to it
  GET  /api/snapshot          the exported snapshot (demo mode: no network, no model calls)
  POST /api/live/start        {"objective": str} -> {"run_id": pair_id, "agents": [{agent_id, label}]}; starts a
                              PAIR of real swarms (costs money)
  GET  /api/live/<pair_id>    build_pair_view of that pair as it is written, plus "error" if its thread crashed

Live mode (demo/pair.py): scenarios/nvda_amd.json, or a team planned for the typed objective (demo/team.py), run twice at once, one swarm run by
Nano + the original prompt (a), one by Nano + the GEPA prompt (b); each corrects its own drifting agents and
writes its own team report. 409 while a pair is going, 400 without API keys.
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import unquote, urlparse

from ..config import SNAPSHOT, load_settings
from ..timeline import agent_label
from .pair import LiveError, Pair, build_pair
from .views import PAIR_FILE, build_pair_view

STATIC = Path(__file__).parent / "static"
DEFAULT_SNAPSHOT = SNAPSHOT
MAX_BODY_BYTES = 64 * 1024
_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_FILE = re.compile(r"^[A-Za-z0-9._-]+$")


class LiveBusy(LiveError):
    status = 409


class LiveRuns:
    """At most one live pair at a time, each run with asyncio.run in its own daemon thread."""

    def __init__(self, runs_dir: Path, build: Callable[[str, Path], Pair]):
        self.runs_dir = Path(runs_dir)
        self.build = build
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._pairs: dict[str, Pair] = {}
        self.errors: dict[str, Optional[str]] = {}

    def running(self) -> Optional[str]:
        return next((rid for rid, t in self._threads.items() if t.is_alive()), None)

    def start(self, objective: str) -> str:
        with self._lock:
            if (rid := self.running()):
                raise LiveBusy(f"live pair {rid} is still going; one at a time")
            pair = self.build(objective, self.runs_dir)  # synchronous, so pair.json exists before we answer
            rid = pair.pair_id
            self._pairs[rid], self.errors[rid] = pair, None
            self._threads[rid] = threading.Thread(target=self._run, args=(rid, pair), daemon=True,
                                                  name=f"live-{rid}")
            self._threads[rid].start()
            return rid

    def _run(self, rid: str, pair: Pair) -> None:
        try:
            asyncio.run(pair.run_all())
        except Exception as e:  # surfaced to the page via /api/live/<pair_id>
            self.errors[rid] = f"{type(e).__name__}: {e}"

    def agents(self, run_id: str) -> list[dict]:
        pair = self._pairs.get(run_id)
        return [{"agent_id": a.agent_id, "label": a.label or agent_label(a.agent_id)}
                for a in (pair.left.scenario.agents if pair else [])]

    def view(self, run_id: str) -> Optional[dict]:
        if not _RUN_ID.match(run_id) or run_id in (".", ".."):
            return None
        d = self.runs_dir / run_id
        if not (d / PAIR_FILE).is_file():
            return None
        # each swarm's meta.json is written last; the live objects fill in its scenario and judge until then
        return {**build_pair_view(d, live=self._pairs.get(run_id)), "source": "live", "error": self.errors.get(run_id)}


class _Handler(BaseHTTPRequestHandler):
    snapshot: Path
    live: LiveRuns
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, body: Any, status: int = 200) -> None:
        self._send(status, json.dumps(body).encode(), "application/json")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parts = [unquote(p) for p in urlparse(self.path).path.split("/") if p]
        if not parts or parts == ["index.html"]:
            page = STATIC / "demo.html"
            if not page.is_file():
                return self._send(404, b"demo.html not found (finresearch/demo/static/demo.html)\n", "text/plain")
            return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        if len(parts) == 2 and parts[0] == "static" and _SAFE_FILE.match(parts[1]) and (STATIC / parts[1]).is_file():
            ctype = mimetypes.guess_type(parts[1])[0] or "application/octet-stream"
            return self._send(200, (STATIC / parts[1]).read_bytes(), ctype)
        if parts == ["api", "health"]:
            return self._json({"ok": True, "live_running": self.live.running()})
        if parts == ["api", "snapshot"]:
            if not self.snapshot.is_file():
                return self._json({"error": f"no snapshot at {self.snapshot}; "
                                            "run python -m finresearch.demo.export"}, 404)
            return self._send(200, self.snapshot.read_bytes(), "application/json")
        if len(parts) == 3 and parts[:2] == ["api", "live"]:
            view = self.live.view(parts[2])
            return self._json(view) if view else self._json({"error": f"unknown run {parts[2]}"}, 404)
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            return self._json({"error": "body too large"}, 413)
        raw = self.rfile.read(length) if length else b""
        if parts != ["api", "live", "start"]:
            return self._json({"error": "not found"}, 404)
        try:
            body = json.loads(raw or b"{}")
            objective = body.get("objective") or ""
            if not isinstance(objective, str):
                raise ValueError
        except (ValueError, AttributeError):
            return self._json({"error": 'body must be JSON like {"objective": "..."}'}, 400)
        try:
            rid = self.live.start(objective)
            return self._json({"run_id": rid, "agents": self.live.agents(rid)})
        except LiveError as e:
            return self._json({"error": str(e)}, e.status)
        except Exception as e:
            return self._json({"error": f"could not start: {type(e).__name__}: {e}"}, 500)


def make_server(snapshot: Path = DEFAULT_SNAPSHOT, *, runs_dir: Optional[Path] = None,
                host: str = "127.0.0.1", port: int = 8770,
                build: Optional[Callable[[str, Path], Pair]] = None) -> ThreadingHTTPServer:
    """`build(objective, runs_dir) -> Pair` is injectable for tests; the default reads settings per start."""
    build = build or (lambda objective, d: build_pair(objective, d, load_settings()))
    live = LiveRuns(Path(runs_dir) if runs_dir else load_settings().runs_dir, build)
    handler = type("Handler", (_Handler,), {"snapshot": Path(snapshot), "live": live})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="finresearch.demo.server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--runs-dir", type=Path, help="where live runs are written (default PITSTOP_RUNS_DIR / runs)")
    a = ap.parse_args()
    srv = make_server(a.snapshot, runs_dir=a.runs_dir, host=a.host, port=a.port)
    print(f"GEPA demo: http://{a.host}:{srv.server_port}/  (snapshot {a.snapshot}; Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
