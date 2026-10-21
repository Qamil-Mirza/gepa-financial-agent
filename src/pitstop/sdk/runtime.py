"""Process-wide PitStop runtime: store, user config and budget ledger."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from ..budget.ledger import Ledger
from ..config import pitstop_home
from ..store import Store
from ..userconfig import UserConfig, load_user_config


@dataclass
class Runtime:
    store: Store
    config: UserConfig
    ledger: Ledger
    home: Path


_lock = threading.Lock()
_runtime: Runtime | None = None


def init(config_path: Path | None = None, home: Path | None = None) -> Runtime:
    """Create (or return the existing) runtime. Idempotent per home."""
    global _runtime
    home = Path(home).expanduser() if home is not None else pitstop_home()
    with _lock:
        if _runtime is not None and _runtime.home == home:
            return _runtime
        store = Store(home / "pitstop.db")
        config = load_user_config(Path(config_path) if config_path is not None else home / "config.toml")
        _runtime = Runtime(store, config, Ledger(store, config), home)
        return _runtime


def get_runtime() -> Runtime:
    rt = _runtime
    return rt if rt is not None else init()


def reset_runtime() -> None:
    global _runtime
    with _lock:
        _runtime = None
