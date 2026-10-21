"""Process-wide PitStop runtime: store, user config and budget ledger."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from ..budget.ledger import Ledger
from ..config import pitstop_home
from ..store import Store, StoreVersionError
from ..userconfig import ConfigInvalid, UserConfig, load_user_config
from .errors import Halted


@dataclass
class Runtime:
    store: Store
    config: UserConfig
    ledger: Ledger
    home: Path


_lock = threading.Lock()
_runtime: Runtime | None = None


def init(config_path: Path | None = None, home: Path | None = None) -> Runtime:
    """Create (or return the existing) runtime. Idempotent per home.

    Raises ConfigInvalid (a ValueError) when config.toml is unusable or pitstop.db was written by a newer
    PitStop; no runtime is cached then, so every later call re-checks and keeps failing closed until it is fixed."""
    global _runtime
    home = Path(home).expanduser() if home is not None else pitstop_home()
    with _lock:
        if _runtime is not None and _runtime.home == home:
            return _runtime
        config = load_user_config(Path(config_path) if config_path is not None else home / "config.toml")
        try:
            store = Store(home / "pitstop.db")
        except StoreVersionError as e:  # a too-new store fails closed like a bad config: budgets must not vanish
            raise ConfigInvalid(str(e)) from e
        try:
            ledger = Ledger(store, config)
        except Exception as e:  # e.g. a timezone the zoneinfo database cannot load
            raise ConfigInvalid(f"budget ledger rejected the config: {type(e).__name__}: {e}") from e
        _runtime = Runtime(store, config, ledger, home)
        return _runtime


def get_runtime() -> Runtime:
    rt = _runtime
    return rt if rt is not None else init()


def halt_if_config_invalid() -> None:
    """Call-boundary guard: an invalid config fails closed with Halted (the hard budget must never silently
    vanish). Any other runtime failure (sqlite busy/locked, unwritable home) is left to the caller's
    fail-open handling, so this never raises anything but Halted."""
    try:
        get_runtime()
    except ConfigInvalid as e:
        raise config_halt(e) from e
    except Exception:
        pass


def config_halt(e: ConfigInvalid) -> Halted:
    return Halted(f"pitstop config invalid: {e}")


def reset_runtime() -> None:
    global _runtime
    with _lock:
        _runtime = None
