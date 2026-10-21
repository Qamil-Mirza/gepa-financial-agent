from __future__ import annotations

from pitstop.store import Store

TABLES = {"meta", "runs", "events", "drifts", "ledger", "prompts"}


def test_store_creates_db_with_wal_and_schema(tmp_path):
    path = tmp_path / "sub" / "pitstop.db"
    store = Store(path)
    assert path.exists()
    conn = store.connect()
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store.schema_version() == 1
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
    assert second.schema_version() == 1
    assert second.connect().execute("SELECT value FROM meta WHERE key='x'").fetchone()[0] == "1"
