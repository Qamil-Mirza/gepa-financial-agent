from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Policy:
    """Immutable supervisor system prompt. version = sha256(text)[:12]."""
    text: str
    version: str
    name: str

    @classmethod
    def from_text(cls, text: str, name: str = "candidate") -> "Policy":
        return cls(text=text, version=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], name=name)

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None) -> "Policy":
        p = Path(path)
        return cls.from_text(p.read_text(encoding="utf-8"), name or p.stem)


class PolicyStore:
    def __init__(self, root: str | Path = "policies") -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.manifest_path = self.versions_dir / "manifest.json"

    def _manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"current": None, "versions": {}}

    def _write_manifest(self, m: dict[str, Any]) -> None:
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _resolve(self, name_or_path: str) -> Path:
        p = Path(name_or_path)
        if p.exists():
            return p
        for cand in (self.root / f"{name_or_path}.txt", self.versions_dir / f"{name_or_path}.txt"):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"policy not found: {name_or_path}")

    def load(self, name_or_path: str) -> Policy:
        p = self._resolve(name_or_path)
        return Policy.from_file(p, name=p.stem)

    def baseline(self) -> Policy:
        return self.load("baseline")

    def current(self) -> Policy:
        return self.load("current")

    def save_version(self, text: str, name: str, metadata: dict[str, Any] | None = None) -> Policy:
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        (self.versions_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        pol = Policy.from_text(text, name=name)
        m = self._manifest()
        m["versions"][name] = {"version": pol.version, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "metadata": metadata or {}}
        self._write_manifest(m)
        return pol

    def promote(self, name: str) -> Policy:
        pol = self.load(name)
        (self.root / "current.txt").write_text(pol.text, encoding="utf-8")
        m = self._manifest()
        m["current"] = name
        self._write_manifest(m)
        return Policy.from_text(pol.text, name="current")

    def list_versions(self) -> dict[str, Any]:
        return self._manifest()["versions"]
