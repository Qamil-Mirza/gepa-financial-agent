from __future__ import annotations

import sqlite3

import pytest

from pitstop.store import MIGRATIONS, SCHEMA_VERSION, Store, StoreVersionError

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
        assert store.schema_version() == SCHEMA_VERSION
        cols = {r["name"] for r in store.connect().execute("PRAGMA table_info(runs)")}
        assert "budget_usd" in cols
    assert store.connect().execute("SELECT budget_usd FROM runs WHERE id='old'").fetchone()[0] is None


# -- v3: stepwise migrations, version check, nested transactions --------------------------------------------

V1_SCHEMA = """
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE runs (
        id TEXT PRIMARY KEY, workflow TEXT, goal TEXT, criticality TEXT, sensitivity TEXT,
        fix_mode TEXT, started_at REAL, ended_at REAL, status TEXT, model TEXT, base_url TEXT,
        prompt_versions TEXT, pitstop_version TEXT, adapter TEXT, price_known INTEGER DEFAULT 1);
    CREATE TABLE events (
        id INTEGER PRIMARY KEY, run_id TEXT, ts REAL, agent TEXT, kind TEXT, name TEXT,
        payload TEXT, cost_usd REAL, parent_id INTEGER);
    CREATE TABLE drifts (
        id INTEGER PRIMARY KEY, run_id TEXT, workflow TEXT, ts REAL, level TEXT, kind TEXT,
        drift_type TEXT, evidence TEXT, action TEXT, status TEXT DEFAULT 'open');
    CREATE TABLE ledger (
        id INTEGER PRIMARY KEY, ts REAL, run_id TEXT, workflow TEXT, meter TEXT, model TEXT,
        base_url TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, usd REAL, state TEXT);
    CREATE TABLE prompts (
        name TEXT, version INTEGER, text TEXT, sha TEXT, source TEXT, created_at REAL,
        active INTEGER, PRIMARY KEY(name, version));
    INSERT INTO meta VALUES ('schema_version', '1');
    INSERT INTO drifts(run_id, workflow, ts, level, kind, drift_type, evidence, action)
        VALUES ('r1', 'wf', 123.0, 'P1', 'off_task', 'unproductive_loop', '[{"n": 1}, {"n": 2}]', 'steer');
"""


def _make_v1_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(V1_SCHEMA)
    conn.commit()
    conn.close()
    return path


def _columns(store) -> dict:
    conn = store.connect()
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return {t: [tuple(r) for r in conn.execute(f"PRAGMA table_info({t})")] for t in tables}


def _set_version(path, version: int) -> None:
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(version),))
    conn.commit()
    conn.close()


def _meta(store, key):
    row = store.connect().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def test_migrations_cover_every_version_step():
    assert SCHEMA_VERSION == 3
    assert sorted(MIGRATIONS) == list(range(2, SCHEMA_VERSION + 1))


def test_fresh_db_matches_migrated_db(tmp_path):
    old = _make_v1_db(tmp_path / "old.db")
    assert _columns(Store(old)) == _columns(Store(tmp_path / "new.db"))
    assert Store(old).schema_version() == SCHEMA_VERSION == 3


def test_v3_migration_backfills_drifts(tmp_path):
    store = Store(_make_v1_db(tmp_path / "old.db"))
    row = store.connect().execute("SELECT ts, last_seen, occurrences FROM drifts").fetchone()
    assert (row["ts"], row["last_seen"], row["occurrences"]) == (123.0, 123.0, 2)  # occurrences = old evidence count


def test_v2_db_migrates_to_v3(tmp_path):
    path = tmp_path / "v2.db"
    conn = sqlite3.connect(path)
    conn.executescript(V1_SCHEMA.replace("price_known INTEGER DEFAULT 1)", "price_known INTEGER DEFAULT 1, budget_usd REAL)")
                       .replace("('schema_version', '1')", "('schema_version', '2')"))
    conn.commit()
    conn.close()
    store = Store(path)
    assert store.schema_version() == 3
    assert _columns(store) == _columns(Store(tmp_path / "new.db"))


def test_failing_migration_step_rolls_back_every_step(tmp_path, monkeypatch):
    path = _make_v1_db(tmp_path / "old.db")

    def boom(conn):
        raise sqlite3.OperationalError("step 3 failed")

    monkeypatch.setitem(MIGRATIONS, 3, boom)
    with pytest.raises(sqlite3.OperationalError, match="step 3 failed"):
        Store(path)
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "1"
    assert "budget_usd" not in {r[1] for r in conn.execute("PRAGMA table_info(runs)")}  # step 2 rolled back too
    conn.close()


def test_newer_db_raises_store_version_error(tmp_path):
    path = tmp_path / "pitstop.db"
    Store(path)
    _set_version(path, 99)
    with pytest.raises(StoreVersionError, match="99"):
        Store(path)
    conn = sqlite3.connect(path)  # untouched
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "99"
    conn.close()


def test_runtime_fails_closed_on_newer_db(tmp_path):
    import pitstop
    from pitstop.config import pitstop_home
    from pitstop.sdk.errors import Halted
    from pitstop.sdk.runtime import init
    from pitstop.userconfig import ConfigInvalid
    from tests.sdk.fakes import FakeOpenAI, make_response

    path = pitstop_home() / "pitstop.db"
    Store(path)
    _set_version(path, 99)
    with pytest.raises(ConfigInvalid, match="schema version 99"):
        init()
    fake = FakeOpenAI([make_response("hi")])
    with pytest.raises(Halted, match="pitstop config invalid: .*schema version 99"):
        pitstop.wrap(fake).chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert fake.calls == []


def test_nested_transaction_inner_rollback_keeps_outer(tmp_path):
    s = Store(tmp_path / "pitstop.db")
    with s.transaction() as c:
        c.execute("INSERT INTO meta VALUES ('a','1')")
        with pytest.raises(RuntimeError):
            with s.transaction() as c2:
                c2.execute("INSERT INTO meta VALUES ('b','1')")
                raise RuntimeError
        c.execute("INSERT INTO meta VALUES ('c','1')")
    assert _meta(s, "a") == "1" and _meta(s, "b") is None and _meta(s, "c") == "1"
    assert not s.connect().in_transaction


def test_nested_transaction_inner_commit_is_undone_by_outer_rollback(tmp_path):
    s = Store(tmp_path / "pitstop.db")
    with pytest.raises(RuntimeError):
        with s.transaction(immediate=True) as c:
            c.execute("INSERT INTO meta VALUES ('a','1')")
            with s.transaction(immediate=True) as c2:  # immediate inside an outer transaction: a savepoint
                with s.transaction() as c3:
                    c3.execute("INSERT INTO meta VALUES ('b','1')")
                c2.execute("INSERT INTO meta VALUES ('c','1')")
            raise RuntimeError
    assert (_meta(s, "a"), _meta(s, "b"), _meta(s, "c")) == (None, None, None)
    assert not s.connect().in_transaction
    with s.transaction() as c:  # the connection is usable again, at depth 0
        c.execute("INSERT INTO meta VALUES ('d','1')")
    assert _meta(s, "d") == "1"


def test_nested_transactions_commit_together(tmp_path):
    s = Store(tmp_path / "pitstop.db")
    with s.transaction() as c:
        with s.transaction(immediate=True) as c2:
            c2.execute("INSERT INTO meta VALUES ('a','1')")
        assert c.in_transaction
    assert _meta(s, "a") == "1"
    assert not s.connect().in_transaction
