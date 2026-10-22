"""Process-wide PitStop runtime: store, user config, budget ledger and the L1 gate's judge."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..budget.ledger import Ledger
from ..config import pitstop_home
from ..store import Store, StoreVersionError
from ..userconfig import ConfigInvalid, UserConfig, load_user_config
from .errors import Halted

if TYPE_CHECKING:
    from ..detect.l1 import Gate

log = logging.getLogger("pitstop")

_UNBUILT = object()


@dataclass
class Runtime:
    store: Store
    config: UserConfig
    ledger: Ledger
    home: Path
    judge_client: Any = None  # from init(judge_client=...); wrapped when the gate is built
    judge_model: str | None = None
    _gate: Any = field(default=_UNBUILT, repr=False, compare=False)
    _no_gate: str | None = field(default=None, repr=False, compare=False)  # why gate() is None
    _gate_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def gate(self) -> Gate | None:
        """The L1 pre-action gate, resolved once per runtime and cached; None when no judge is available
        (`no_gate_reason()` says why).

        An `init(judge_client=...)` client is wrapped and used with `judge_model` or `[judge] model`; an async
        client is refused (the gate's call is sync). Otherwise, if the `[judge] api_key_env` variable is set, an
        OpenAI client for `[judge] base_url` is built (no retries, `timeout_s`). A key exported later takes effect
        only after `reset_runtime()`, a restart or `init(judge_client=...)`. A failure to build the judge is logged
        once (type name only) and leaves no gate."""
        g = self._gate
        if g is not _UNBUILT:
            return g
        with self._gate_lock:
            if self._gate is _UNBUILT:
                self._gate, self._no_gate = self._build_gate()
            return self._gate

    def no_gate_reason(self) -> str:
        """Why `gate()` returned None, for the gate_error event and the block reason. Names the env var, never
        its value."""
        self.gate()
        return self._no_gate or _no_judge_text(self.config.judge.api_key_env)

    def _build_gate(self) -> tuple[Gate | None, str | None]:
        from ..detect.l1 import Gate
        from .wrap import wrap
        from .wrap.proxy import _is_async

        spec = self.config.judge
        try:
            if self.judge_client is not None:
                create = getattr(getattr(getattr(self.judge_client, "chat", None), "completions", None),
                                 "create", None)
                if _is_async(create):
                    log.warning("pitstop: %s; gated tools follow on_gate_error", ASYNC_JUDGE)
                    return None, ASYNC_JUDGE
                return Gate(wrap(self.judge_client), self.judge_model or spec.model), None
            key = os.environ.get(spec.api_key_env)
            if not key:
                return None, _no_judge_text(spec.api_key_env)
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=spec.base_url, timeout=spec.timeout_s, max_retries=0)
            return Gate(wrap(client), self.judge_model or spec.model), None
        except Exception as e:
            log.warning("pitstop: could not build the L1 gate judge (%s); gated tools follow on_gate_error",
                        type(e).__name__)
            return None, f"judge setup failed ({type(e).__name__})"


ASYNC_JUDGE = "async judge_client unsupported; pass a sync client"


def _no_judge_text(api_key_env: str) -> str:
    return f"no judge configured: set {api_key_env} or [judge] in config.toml"


_lock = threading.Lock()
_runtime: Runtime | None = None


def init(config_path: Path | None = None, home: Path | None = None, *, judge_client: Any = None,
         judge_model: str | None = None) -> Runtime:
    """Create (or return the existing) runtime. Idempotent per home.

    `judge_client` (a sync OpenAI-compatible client) and `judge_model` choose the L1 gate's judge; passing either
    to an existing runtime replaces its judge and rebuilds the gate on next use. Otherwise the judge is resolved
    once per runtime: a key exported later takes effect after `reset_runtime()`, a restart or
    `init(judge_client=...)`.

    Raises ConfigInvalid (a ValueError) when config.toml is unusable or pitstop.db was written by a newer
    PitStop; no runtime is cached then, so every later call re-checks and keeps failing closed until it is fixed."""
    global _runtime
    home = Path(home).expanduser() if home is not None else pitstop_home()
    with _lock:
        if _runtime is not None and _runtime.home == home:
            if judge_client is not None or judge_model is not None:
                with _runtime._gate_lock:
                    if judge_client is not None:
                        _runtime.judge_client = judge_client
                    if judge_model is not None:
                        _runtime.judge_model = judge_model
                    _runtime._gate, _runtime._no_gate = _UNBUILT, None
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
        _runtime = Runtime(store, config, ledger, home, judge_client, judge_model)
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
