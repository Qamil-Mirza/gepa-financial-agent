"""Managed prompts: versioned in the store, developer edits to the code default win."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Literal

from .context import _safe, current_run
from .runtime import get_runtime

log = logging.getLogger("pitstop")


@dataclass(frozen=True)
class ManagedPrompt:
    name: str
    text: str
    version: int
    source: Literal["default", "dev_edit", "gepa"]
    sha: str


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve(name: str, default: str) -> ManagedPrompt:
    sha = _sha(default)
    with get_runtime().store.transaction(immediate=True) as c:
        rows = c.execute("SELECT version, text, sha, source, active FROM prompts WHERE name=? ORDER BY version",
                         (name,)).fetchall()
        if not rows:
            c.execute("INSERT INTO prompts(name, version, text, sha, source, created_at, active)"
                      " VALUES (?,?,?,?,?,?,1)", (name, 1, default, sha, "default", time.time()))
            c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (f"prompt_default_sha:{name}", sha))
            return ManagedPrompt(name, default, 1, "default", sha)
        key = f"prompt_default_sha:{name}"
        seen = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        changed = (seen[0] != sha) if seen is not None else not any(r["sha"] == sha for r in rows)
        c.execute("INSERT INTO meta(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, sha))
        if changed:
            match = [r for r in rows if r["sha"] == sha]
            if match:
                version = max(r["version"] for r in match)
                c.execute("UPDATE prompts SET active=(version=?) WHERE name=?", (version, name))
                r = next(r for r in match if r["version"] == version)
                return ManagedPrompt(name, r["text"], version, r["source"], sha)
            version = max(r["version"] for r in rows) + 1
            c.execute("UPDATE prompts SET active=0 WHERE name=?", (name,))
            c.execute("INSERT INTO prompts(name, version, text, sha, source, created_at, active)"
                      " VALUES (?,?,?,?,?,?,1)", (name, version, default, sha, "dev_edit", time.time()))
            return ManagedPrompt(name, default, version, "dev_edit", sha)
        active = [r for r in rows if r["active"]]
        row = max(active, key=lambda r: r["version"]) if active else rows[-1]
        if not active:
            c.execute("UPDATE prompts SET active=1 WHERE name=? AND version=?", (name, row["version"]))
        return ManagedPrompt(name, row["text"], row["version"], row["source"], row["sha"])


def _record(run_id: str, versions: dict[str, int]) -> None:
    with get_runtime().store.transaction() as c:
        c.execute("UPDATE runs SET prompt_versions=? WHERE id=?", (json.dumps(versions), run_id))


def prompt(name: str, default: str) -> ManagedPrompt:
    try:
        p = _resolve(name, default)
    except Exception:
        log.warning("pitstop: prompt store unavailable; using code default", exc_info=True)
        return ManagedPrompt(name, default, 0, "default", _sha(default))
    ctx = current_run()
    if ctx is not None:
        ctx.prompt_versions[name] = p.version
        snapshot = dict(ctx.prompt_versions)
        _safe(lambda: _record(ctx.run_id, snapshot))
    return p


def set_active_version(name: str, version: int) -> None:
    """Make `version` the single active version of `name` (promotion and rollback)."""
    with get_runtime().store.transaction(immediate=True) as c:
        if c.execute("SELECT 1 FROM prompts WHERE name=? AND version=?", (name, version)).fetchone() is None:
            raise ValueError(f"no version {version} of prompt {name!r}")
        c.execute("UPDATE prompts SET active=(version=?) WHERE name=?", (version, name))
