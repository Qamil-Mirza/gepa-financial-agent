"""Local SQLite store (WAL) shared by the SDK, supervisor and UI."""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 2
_SCHEMA = Path(__file__).with_name("schema.sql")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        conn = self.connect()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA.read_text())
        self._migrate()

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    @contextmanager
    def transaction(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Run the block in one transaction. The thread-local connection is never left inside an open
        transaction: a failing COMMIT (e.g. SQLITE_BUSY) is rolled back and re-raised, and a failing
        ROLLBACK never masks the original error."""
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield conn
        except BaseException:
            _rollback_quietly(conn)
            raise
        try:
            conn.execute("COMMIT")
        except BaseException:
            _rollback_quietly(conn)
            raise

    def _migrate(self) -> None:
        """Upgrade an older DB in place (idempotent, safe across concurrent openers)."""
        if self.schema_version() >= SCHEMA_VERSION:
            return
        with self.transaction(immediate=True) as c:
            cols = {r["name"] for r in c.execute("PRAGMA table_info(runs)")}
            if "budget_usd" not in cols:
                c.execute("ALTER TABLE runs ADD COLUMN budget_usd REAL")
            c.execute("UPDATE meta SET value=? WHERE key='schema_version' AND CAST(value AS INTEGER) < ?",
                      (str(SCHEMA_VERSION), SCHEMA_VERSION))

    def schema_version(self) -> int:
        row = self.connect().execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row[0]) if row else 0


def _rollback_quietly(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ROLLBACK")
    except Exception:
        pass
