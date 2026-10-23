"""Local pit-wall dashboard. Standard library only; reads runs/ directly.

  python -m finresearch.ui.server         # then open http://localhost:8765

Logs are append-only JSONL, so the UI can tail a run while it is in progress.
"""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

STATIC = Path(__file__).parent / "static"
_SAFE = re.compile(r"^[A-Za-z0-9._@-]+$")


def read_jsonl_complete(path: Path) -> list[dict]:
    """Parse only complete lines; a line still being written is skipped until next poll."""
    if not path.exists():
        return []
    lines = path.read_text().split("\n")[:-1]  # last element is '' or a partial line
    out = []
    for line in lines:
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


def slim_decision(rec: dict) -> dict:
    inp, res = rec.get("input", {}), rec.get("result", {})
    return {
        "agent_id": rec["agent_id"], "step": rec["step"], "phase": rec["phase"],
        "alert_level": rec.get("alert_level"), "alert_note": rec.get("alert_note"),
        "drift_injection_active": rec.get("drift_injection_active", False),
        "cutoff_seq": inp.get("observation_cutoff_seq"),
        "proposed": inp.get("proposed_next_action"),
        "decision": res.get("decision", {}), "issues": res.get("issues", []),
        "dropped_evidence_ids": res.get("dropped_evidence_ids", []),
        "valid_json": res.get("valid_json", True), "latency_s": res.get("latency_s", 0),
        "prompt_version": res.get("prompt_version", ""),
    }


class Handler(BaseHTTPRequestHandler):
    runs_dir: Path = Path("runs")

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _run_dir(self, run_id: str) -> Path | None:
        if not _SAFE.match(run_id):
            return None
        d = (self.runs_dir / run_id).resolve()
        return d if d.parent == self.runs_dir.resolve() and d.is_dir() else None

    def do_GET(self) -> None:
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]
        q = parse_qs(url.query)
        if not parts:
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if parts == ["api", "runs"]:
            return self._json(self.list_runs())
        if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
            d = self._run_dir(parts[2])
            if d is None:
                return self._json({"error": f"No run named {parts[2]!r} in {self.runs_dir}"}, 404)
            if len(parts) == 3:
                ev, dec = int(q.get("ev", ["0"])[0]), int(q.get("dec", ["0"])[0])
                events = read_jsonl_complete(d / "events.jsonl")
                decisions = read_jsonl_complete(d / "decisions.jsonl")
                meta_p = d / "meta.json"
                meta = json.loads(meta_p.read_text()) if meta_p.exists() else None
                compares = sorted(p.name for p in (d / "compare").glob("*.jsonl")) if (d / "compare").is_dir() else []
                return self._json({
                    "meta": meta, "events": events[ev:], "decisions": [slim_decision(r) for r in decisions[dec:]],
                    "ev_total": len(events), "dec_total": len(decisions), "compares": compares,
                })
            if len(parts) == 5 and parts[3] == "compare" and _SAFE.match(parts[4]):
                return self._json(read_jsonl_complete(d / "compare" / parts[4]))
        self._json({"error": f"Nothing at {url.path}"}, 404)

    def list_runs(self) -> list[dict]:
        if not self.runs_dir.is_dir():
            return []
        out = []
        for d in self.runs_dir.iterdir():
            if not d.is_dir():
                continue
            meta_p = d / "meta.json"
            meta = {}
            if meta_p.exists():
                try:
                    meta = json.loads(meta_p.read_text())
                except json.JSONDecodeError:
                    pass
            ev = d / "events.jsonl"
            out.append({"run_id": d.name, "status": meta.get("status", "unknown"),
                        "supervisor_model": (meta.get("supervisor") or {}).get("model", ""),
                        "updated": max((ev.stat().st_mtime if ev.exists() else 0), d.stat().st_mtime)})
        return sorted(out, key=lambda r: -r["updated"])


def serve(runs_dir: Path, port: int = 8765) -> None:
    Handler.runs_dir = Path(runs_dir)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"PitStop pit wall: http://localhost:{port}  (reading {runs_dir}; Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":  # python -m finresearch.ui.server [--port 8765] [--runs-dir runs]
    import argparse

    from ..config import load_settings

    ap = argparse.ArgumentParser(prog="finresearch.ui.server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--runs-dir")
    a = ap.parse_args()
    serve(Path(a.runs_dir) if a.runs_dir else load_settings().runs_dir, a.port)
