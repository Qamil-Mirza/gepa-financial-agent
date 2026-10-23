"""The dashboard's background worker: a scan pass every `scan_every_s`, the digest check, a clean stop."""
import logging
import sqlite3
import time

from pitstop.server.worker import Worker
from pitstop.store import Store

from tests.detect.run_factory import make_run

T0 = 1_790_000_000.0


class _QuickBusyStore(Store):
    """A store whose connections give up on a held write lock after 100 ms instead of 5 s."""

    def connect(self):
        conn = super().connect()
        conn.execute("PRAGMA busy_timeout=100")
        return conn


def _wait(pred, timeout=2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_worker_scans_and_stops(tmp_path):
    store = Store(tmp_path / "p.db")
    make_run(store, "wf", started_at=T0, ended_at=T0 + 5)
    worker = Worker(store, scan_every_s=60)
    assert worker.status() == {"enabled": True, "running": False, "last_scan_at": None, "last_scan_runs": None,
                               "last_error": None, "scan_every_s": 60.0}
    worker.start()
    try:
        assert _wait(lambda: worker.status()["last_scan_runs"] == 1)
        assert worker.status()["running"] is True
        assert worker.status()["last_error"] is None
    finally:
        t = time.monotonic()
        worker.stop()
    assert time.monotonic() - t < 1.0  # stop wakes the 60 s sleep
    assert worker.status()["running"] is False
    row = store.connect().execute("SELECT scanned_at FROM runs").fetchone()
    assert row["scanned_at"] is not None


def test_worker_calls_the_digest_check(tmp_path, monkeypatch):
    from pitstop.alerts import dispatch

    calls = []
    monkeypatch.setattr(dispatch, "run_digest", lambda: calls.append(time.monotonic()) or (0, {}))
    worker = Worker(Store(tmp_path / "p.db"), scan_every_s=0.05, digest_every_s=3600)
    worker.start()
    try:
        assert _wait(lambda: worker.status()["last_scan_at"] is not None)
        assert _wait(lambda: len(calls) == 1)
        time.sleep(0.3)  # several scan passes, but the digest check waits its hour
        assert len(calls) == 1
    finally:
        worker.stop()


def test_worker_survives_locked_store(tmp_path, caplog):
    path = tmp_path / "p.db"
    store = _QuickBusyStore(path)
    make_run(store, "wf", started_at=T0, ended_at=T0 + 5)
    store.close_local()
    locker = sqlite3.connect(path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    worker = Worker(store, scan_every_s=0.1)
    with caplog.at_level(logging.WARNING, logger="pitstop"):
        worker.start()
        try:
            assert _wait(lambda: worker.status()["last_error"] == "OperationalError")
            time.sleep(0.6)  # several more passes against the held lock
            st = worker.status()
            assert st["running"] is True and st["last_error"] == "OperationalError"
            assert st["last_scan_runs"] in (None, 0)
            warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith("pitstop")]
            assert len(warnings) == 1, [r.getMessage() for r in warnings]
            assert "OperationalError" in warnings[0].getMessage()
            locker.execute("ROLLBACK")
            assert _wait(lambda: worker.status()["last_scan_runs"] == 1)
            assert worker.status()["last_error"] is None
        finally:
            worker.stop()
            locker.close()
    assert store.connect().execute("SELECT scanned_at FROM runs").fetchone()["scanned_at"] is not None


def test_worker_logs_each_error_type_once_and_continues(tmp_path, monkeypatch, caplog):
    from pitstop.server import worker as worker_mod

    n = {"calls": 0}

    def flaky(store):
        n["calls"] += 1
        raise (ValueError if n["calls"] % 2 else KeyError)("boom")

    monkeypatch.setattr(worker_mod, "scan", flaky)
    worker = Worker(Store(tmp_path / "p.db"), scan_every_s=0.02)
    with caplog.at_level(logging.WARNING, logger="pitstop"):
        worker.start()
        try:
            assert _wait(lambda: n["calls"] >= 8)
        finally:
            worker.stop()
    msgs = [r.getMessage() for r in caplog.records if r.name.startswith("pitstop")]
    assert sum("ValueError" in m for m in msgs) == 1 and sum("KeyError" in m for m in msgs) == 1
    assert "boom" not in " ".join(msgs)  # the type name only, never the message


def test_worker_closes_its_connection_after_each_pass(tmp_path):
    store = Store(tmp_path / "p.db")
    store.close_local()
    worker = Worker(store, scan_every_s=60)
    worker.start()
    try:
        assert _wait(lambda: worker.status()["last_scan_at"] is not None)
        assert _wait(lambda: store.open_connections() == 0)
    finally:
        worker.stop()


def test_worker_clamps_a_huge_interval(tmp_path):
    """`Event.wait` overflows above threading.TIMEOUT_MAX and would kill the thread; the CLI rejects such a value
    and the worker clamps it as a backstop."""
    import threading

    from pitstop.server.worker import MAX_SCAN_EVERY_S

    worker = Worker(Store(tmp_path / "p.db"), scan_every_s=1e300)
    assert worker.status()["scan_every_s"] == MAX_SCAN_EVERY_S <= threading.TIMEOUT_MAX
    worker.start()
    try:
        assert _wait(lambda: worker.status()["last_scan_at"] is not None)
        time.sleep(0.05)
        assert worker.status()["running"] is True  # still alive in its (clamped) sleep
    finally:
        worker.stop()
    assert worker.status()["running"] is False
