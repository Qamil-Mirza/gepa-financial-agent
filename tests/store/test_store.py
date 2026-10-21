from __future__ import annotations

import sqlite3

import pytest

from pitstop.store import SCHEMA_VERSION, Store

TABLES = {"meta", "runs", "events", "drifts", "ledger", "prompts"}


def test_store_creates_db_with_wal_and_schema(tmp_path):
    path = tmp_path / "sub" / "pitstop.db"
    store = Store(path)
    assert path.exists()
    conn = store.connect()
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store.schema_version() == SCHEMA_VERSION
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert TABLES <= names


def test_transaction_rolls_back_on_error(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    try:
        with store.transaction(immediate=True) as conn:
            conn.execute("INSERT INTO meta(key, value) VALUES ('x', '1')")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert store.connect().execute("SELECT COUNT(*) FROM meta WHERE key='x'").fetchone()[0] == 0


def test_store_is_reopenable_and_idempotent(tmp_path):
    path = tmp_path / "pitstop.db"
    first = Store(path)
    with first.transaction() as conn:
        conn.execute("INSERT INTO meta(key, value) VALUES ('x', '1')")
    second = Store(path)
    assert second.schema_version() == SCHEMA_VERSION
    assert second.connect().execute("SELECT value FROM meta WHERE key='x'").fetchone()[0] == "1"


class _FailingConn:
    """Wraps a real sqlite3.Connection and raises once on a chosen statement (sqlite3 objects can't be patched)."""

    def __init__(self, conn, fail_on: str):
        self._conn = conn
        self._fail_on = fail_on
        self.failed = False

    def execute(self, sql, *args):
        if not self.failed and sql.strip().upper() == self._fail_on:
            self.failed = True
            raise sqlite3.OperationalError("database is locked")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_failed_commit_rolls_back_and_connection_stays_usable(tmp_path, monkeypatch):
    store = Store(tmp_path / "pitstop.db")
    real = store.connect()
    proxy = _FailingConn(real, "COMMIT")
    monkeypatch.setattr(store, "connect", lambda: proxy)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        with store.transaction(immediate=True) as conn:
            conn.execute("INSERT INTO meta(key, value) VALUES ('x', '1')")
    monkeypatch.undo()
    assert proxy.failed
    assert not real.in_transaction
    assert real.execute("SELECT COUNT(*) FROM meta WHERE key='x'").fetchone()[0] == 0
    with store.transaction(immediate=True) as conn:  # would raise "cannot start a transaction within a transaction"
        conn.execute("INSERT INTO meta(key, value) VALUES ('y', '1')")
    assert real.execute("SELECT COUNT(*) FROM meta WHERE key='y'").fetchone()[0] == 1


def test_failing_rollback_does_not_mask_original_error(tmp_path, monkeypatch):
    store = Store(tmp_path / "pitstop.db")
    real = store.connect()
    proxy = _FailingConn(real, "ROLLBACK")
    monkeypatch.setattr(store, "connect", lambda: proxy)
    with pytest.raises(RuntimeError, match="boom"):
        with store.transaction() as conn:
            conn.execute("INSERT INTO meta(key, value) VALUES ('x', '1')")
            raise RuntimeError("boom")
    monkeypatch.undo()
    real.execute("ROLLBACK")  # the fake failure left the real transaction open; clean up


def test_v1_db_is_migrated_in_place(tmp_path):
    path = tmp_path / "pitstop.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE runs (id TEXT PRIMARY KEY, workflow TEXT, goal TEXT, criticality TEXT, sensitivity TEXT,
            fix_mode TEXT, started_at REAL, ended_at REAL, status TEXT, model TEXT, base_url TEXT,
            prompt_versions TEXT, pitstop_version TEXT, adapter TEXT, price_known INTEGER DEFAULT 1);
        INSERT INTO meta VALUES ('schema_version', '1');
        INSERT INTO runs(id, workflow) VALUES ('old', 'wf');
    """)
    conn.commit()
    conn.close()
    for _ in range(2):  # idempotent
        store = Store(path)
        assert store.schema_version() == 2
        cols = {r["name"] for r in store.connect().execute("PRAGMA table_info(runs)")}
        assert "budget_usd" in cols
    assert store.connect().execute("SELECT budget_usd FROM runs WHERE id='old'").fetchone()[0] is None
