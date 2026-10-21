"""Timeline API for the UI: JSON endpoints plus a live Server-Sent Events stream.

Stdlib only, so it adds no dependencies. Run it next to (or after) a swarm run:

  python -m pitstop.cli serve --port 8765

  GET /api/health
  GET /api/runs                              newest first, with drift summary
  GET /api/runs/<run_id>/timeline            timeline.json, or the partial timeline of a live run
  GET /api/runs/<run_id>/timeline/stream     SSE: one `entry` event per timeline entry, live;
                                             resume with ?after=<seq> or the Last-Event-ID header
  GET /api/runs/<run_id>/decisions/<index>   full decision record (an entry's refs.decision_index)
  GET /api/runs/<run_id>/events/<event_id>   one trajectory event (an entry's evidence / refs.event_ids)
"""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import parse_qs, urlparse

from .timeline import TIMELINE_VERSION, build_from_run_dir

_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")
POLL_S = 0.25
HEARTBEAT_S = 15.0


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # ignore a trailing partial line the writer has not finished yet
    return [json.loads(l) for l in path.read_text().split("\n")[:-1] if l.strip()]


class RunStore:
    """Read-only view of runs/<run_id>/ directories."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self._rebuild_lock = threading.Lock()

    def run_dir(self, run_id: str) -> Optional[Path]:
        if not _RUN_ID.match(run_id) or run_id in (".", ".."):
            return None
        d = self.runs_dir / run_id
        return d if d.is_dir() else None

    def _ensure_timeline(self, d: Path) -> None:
        # Older runs predate the timeline (or its current version); rebuild once,
        # but never while a run may still be writing.
        if (d / "meta.json").exists() and self._stale(d):
            with self._rebuild_lock:
                if self._stale(d):
                    build_from_run_dir(d)

    @staticmethod
    def _stale(d: Path) -> bool:
        p = d / "timeline.json"
        return not p.exists() or json.loads(p.read_text()).get("timeline_version") != TIMELINE_VERSION

    def list_runs(self) -> list[dict]:
        if not self.runs_dir.exists():
            return []
        out = []
        for d in sorted((p for p in self.runs_dir.iterdir() if p.is_dir() and self.run_dir(p.name)),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            self._ensure_timeline(d)
            tl = self.timeline(d.name) or {}
            entries = tl.get("entries", [])
            out.append({
                "run_id": d.name, "task_id": tl.get("run", {}).get("task_id"),
                "agents": tl.get("run", {}).get("agents", []),
                "in_progress": tl.get("in_progress", True),
                "started_at": entries[0]["timestamp"] if entries else None,
                "finished_at": entries[-1]["timestamp"] if entries and not tl.get("in_progress") else None,
                "flagged_agents": [a for a, s in tl.get("agents", {}).items() if s.get("flags")]
                                  or sorted({e["agent_id"] for e in entries if e["kind"] == "drift_flagged"}),
                "counts": tl.get("counts") or _counts(entries),
            })
        return out

    def timeline(self, run_id: str) -> Optional[dict]:
        d = self.run_dir(run_id)
        if d is None:
            return None
        self._ensure_timeline(d)
        if (d / "timeline.json").exists():
            return {**json.loads((d / "timeline.json").read_text()), "in_progress": False}
        entries = _read_jsonl(d / "timeline.jsonl")
        started = next((e for e in entries if e["kind"] == "run_started"), None)
        run = {"run_id": run_id, **(started or {}).get("data", {})} if started else {"run_id": run_id}
        return {"run": run, "agents": {}, "counts": _counts(entries), "entries": entries, "in_progress": True}

    def decision(self, run_id: str, index: int) -> Optional[dict]:
        d = self.run_dir(run_id)
        recs = _read_jsonl(d / "decisions.jsonl") if d else []
        return recs[index] if 0 <= index < len(recs) else None

    def event(self, run_id: str, event_id: str) -> Optional[dict]:
        d = self.run_dir(run_id)
        return next((e for e in _read_jsonl(d / "events.jsonl") if e["event_id"] == event_id), None) if d else None

    def follow(self, run_id: str, after: int = -1) -> Iterator[Optional[dict]]:
        """Yield timeline entries with seq > after as they are written; None while idle (for heartbeats).
        Ends after the run_finished entry."""
        d = self.run_dir(run_id)
        if d is None:
            return
        self._ensure_timeline(d)
        path, offset, buf = d / "timeline.jsonl", 0, ""
        while True:
            chunk = ""
            if path.exists():
                with path.open() as f:
                    f.seek(offset)
                    chunk = f.read()
                    offset = f.tell()
            buf += chunk
            *lines, buf = buf.split("\n")
            for line in lines:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry["seq"] > after:
                    yield entry
                if entry["kind"] == "run_finished":
                    return
            if not lines:
                yield None
                time.sleep(POLL_S)


def _counts(entries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    return counts


class _Handler(BaseHTTPRequestHandler):
    store: RunStore
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # keep the console for the swarm output
        pass

    def _send_json(self, body: Any, status: int = 200) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _not_found(self, what: str = "not found") -> None:
        self._send_json({"error": what}, 404)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Last-Event-ID")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if parts == ["api", "health"]:
            return self._send_json({"ok": True})
        if parts == ["api", "runs"]:
            return self._send_json(self.store.list_runs())
        if len(parts) < 4 or parts[:2] != ["api", "runs"]:
            return self._not_found()
        run_id, rest = parts[2], parts[3:]
        if self.store.run_dir(run_id) is None:
            return self._not_found(f"unknown run {run_id}")
        if rest == ["timeline"]:
            return self._send_json(self.store.timeline(run_id))
        if rest == ["timeline", "stream"]:
            return self._stream(run_id, parse_qs(url.query))
        if len(rest) == 2 and rest[0] == "decisions" and rest[1].isdigit():
            rec = self.store.decision(run_id, int(rest[1]))
            return self._send_json(rec) if rec else self._not_found("unknown decision index")
        if len(rest) == 2 and rest[0] == "events":
            ev = self.store.event(run_id, rest[1])
            return self._send_json(ev) if ev else self._not_found("unknown event id")
        return self._not_found()

    def _stream(self, run_id: str, query: dict) -> None:
        after_raw = self.headers.get("Last-Event-ID") or (query.get("after") or ["-1"])[0]
        after = int(after_raw) if after_raw.lstrip("-").isdigit() else -1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        last_write = time.monotonic()
        try:
            for entry in self.store.follow(run_id, after):
                if entry is None:
                    if time.monotonic() - last_write < HEARTBEAT_S:
                        continue
                    self.wfile.write(b": keepalive\n\n")
                else:
                    self.wfile.write(f"id: {entry['seq']}\nevent: entry\ndata: {json.dumps(entry)}\n\n".encode())
                self.wfile.flush()
                last_write = time.monotonic()
            self.wfile.write(b"event: end\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away


def make_server(runs_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"store": RunStore(runs_dir)})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
