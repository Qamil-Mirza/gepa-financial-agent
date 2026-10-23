"""The dashboard's background worker (decision 4): one daemon thread that runs `scan` every `scan_every_s` (60 s) and
the digest check every `digest_every_s` (an hour), so a running dashboard replaces a cron `pitstop scan`. A cron
scan next to it is still safe: `scan` claims each run once.

Each pass first checks that the store's write lock can be taken. `scan` itself never raises (a locked store ends its
pass with a warning per pass), so without that check a long-held lock would log a warning every pass and leave
`last_error` empty. A failing step (a held lock, any other error) is logged once per exception type, by type name
only, and the loop carries on; `status()["last_error"]` holds the type until the next completed scan. The digest
check is `pitstop.alerts.dispatch.run_digest()`, a no-op without digest channels and at most one send per 24 h.
The thread's store connection is closed after each pass."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from ..alerts import dispatch
from ..detect.scan import scan
from ..store import Store

log = logging.getLogger("pitstop.server")


class Worker:
    def __init__(self, store: Store, *, scan_every_s: float = 60.0, digest_every_s: float = 3600.0):
        self.store = store
        self.scan_every_s = float(scan_every_s)
        self.digest_every_s = float(digest_every_s)
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._next_digest = 0.0  # time.monotonic(); the first pass runs the digest check
        self._logged: set[str] = set()  # exception type names already logged
        self._last_scan_at: float | None = None
        self._last_scan_runs: int | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        """Start the thread (a no-op while it is running); the first pass runs at once."""
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._wake.clear()
            self._thread = threading.Thread(target=self._loop, name="pitstop-worker", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Wake the thread out of its sleep and wait up to `timeout` s for it to finish its pass."""
        self._wake.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout)

    def status(self) -> dict[str, Any]:
        t = self._thread
        return {"enabled": True, "running": t is not None and t.is_alive() and not self._wake.is_set(),
                "last_scan_at": self._last_scan_at, "last_scan_runs": self._last_scan_runs,
                "last_error": self._last_error, "scan_every_s": self.scan_every_s}

    def _loop(self) -> None:
        while not self._wake.is_set():
            self.run_once()
            self._wake.wait(self.scan_every_s)

    def run_once(self) -> None:
        """One pass: the scan, then the digest check when it is due."""
        try:
            self._step(self._scan)
            if not self._wake.is_set() and time.monotonic() >= self._next_digest:
                self._next_digest = time.monotonic() + self.digest_every_s
                self._step(dispatch.run_digest)
        finally:
            self.store.close_local()

    def _scan(self) -> None:
        with self.store.transaction(immediate=True):  # raises OperationalError while another writer holds the lock
            pass
        report = scan(self.store)
        self._last_scan_at, self._last_scan_runs, self._last_error = time.time(), report.runs, None

    def _step(self, fn) -> None:
        try:
            fn()
        except Exception as e:
            kind = type(e).__name__
            self._last_error = kind
            if kind not in self._logged:
                self._logged.add(kind)
                log.warning("pitstop: dashboard worker pass failed (%s); retrying every %gs", kind,
                            self.scan_every_s)
