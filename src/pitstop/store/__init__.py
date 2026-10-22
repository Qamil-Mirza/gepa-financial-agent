"""Local SQLite store (WAL) shared by the SDK, supervisor and UI.

A fresh DB is created at the latest schema by `schema.sql`. An older DB is upgraded in place by the steps in
`MIGRATIONS` (keyed by the version each step produces), all applied in one `BEGIN IMMEDIATE`. To add a schema
version: append a step, bump `SCHEMA_VERSION` and update `schema.sql` (a fresh DB and a migrated one must
match). A DB written by a newer PitStop is refused with `StoreVersionError` before anything touches it.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 3
_SCHEMA = Path(__file__).with_name("schema.sql")


class StoreVersionError(RuntimeError):
    """The DB's schema_version is newer than this PitStop understands."""


def _add_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    # Idempotent: schema.sql may already have created the table at the latest shape on an old DB.
    if column not in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _v2_runs_budget(conn: sqlite3.Connection) -> None:
    _add_column(conn, "runs", "budget_usd", "REAL")


def _v3_drift_occurrences(conn: sqlite3.Connection) -> None:
    _add_column(conn, "drifts", "last_seen", "REAL")
    _add_column(conn, "drifts", "occurrences", "INTEGER NOT NULL DEFAULT 1")
    conn.execute("UPDATE drifts SET last_seen = ts WHERE last_seen IS NULL")
    # Before v3 every occurrence appended its raw evidence dict. Wrap each one in the v3 entry shape
    # {"run_id", "ts", "detail"} (the row's run and first-seen time are the best we know), and count them:
    # the old list length is the occurrence count.
    for row in conn.execute("SELECT id, run_id, ts, evidence FROM drifts").fetchall():
        try:
            ev = json.loads(row["evidence"]) if row["evidence"] else []
        except Exception:
            continue
        ev = ev if isinstance(ev, list) else [ev]
        wrapped = [e if _is_v3_entry(e) else {"run_id": row["run_id"], "ts": row["ts"], "detail": e} for e in ev]
        conn.execute("UPDATE drifts SET evidence=?, occurrences=? WHERE id=?",
                     (json.dumps(wrapped, default=str), max(len(wrapped), 1), row["id"]))


def _is_v3_entry(e) -> bool:
    return isinstance(e, dict) and set(e) == {"run_id", "ts", "detail"}


MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    2: _v2_runs_budget,
    3: _v3_drift_occurrences,
}


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        conn = self.connect()
        try:
            self._check_version(self._read_version(conn))  # before schema.sql or a migration touches a newer DB
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA.read_text())
            self._migrate()
        except StoreVersionError:
            self._close_local()  # a refused store keeps no open handle on the DB file
            raise

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _close_local(self) -> None:
        conn = getattr(self._local, "conn", None)
        self._local.conn = None
        if conn is not None:
            conn.close()

    @contextmanager
    def transaction(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Run the block in one transaction. The thread-local connection is never left inside an open
        transaction: a failing COMMIT (e.g. SQLITE_BUSY) is rolled back and re-raised, and a failing
        ROLLBACK never masks the original error.

        Transactions nest per thread. An inner block runs in `SAVEPOINT pitstop_<n>`: an error inside it
        rolls back only the inner block (`ROLLBACK TO`), and its writes commit or roll back with the
        outermost transaction. `immediate` only matters for the outermost block: an inner `immediate=True`
        inside a deferred outer transaction is allowed but does not upgrade the lock (the write lock is
        taken at the first write, as for any deferred transaction)."""
        conn = self.connect()
        depth = getattr(self._local, "depth", 0)
        if depth:
            yield from _savepoint(self._local, conn, depth)
            return
        conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        self._local.depth = 1
        try:
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
        finally:
            self._local.depth = 0

    def _migrate(self) -> None:
        """Upgrade an older DB in place: every step above its version runs in one transaction, so a failing
        step leaves the DB at its old version. Safe across concurrent openers (the version is re-read under
        the write lock)."""
        current = self.schema_version()
        self._check_version(current)  # a newer PitStop may have migrated it since the constructor's check
        if current >= SCHEMA_VERSION:
            return
        with self.transaction(immediate=True) as c:
            current = self._read_version(c)
            self._check_version(current)
            for version in range(current + 1, SCHEMA_VERSION + 1):
                MIGRATIONS[version](c)
            c.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),))

    def schema_version(self) -> int:
        return self._read_version(self.connect())

    @staticmethod
    def _read_version(conn: sqlite3.Connection) -> int:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone() is None:
            return 0
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row[0]) if row else 0

    def _check_version(self, version: int) -> None:
        if version > SCHEMA_VERSION:
            raise StoreVersionError(
                f"{self.path} has schema version {version}, newer than this PitStop supports ({SCHEMA_VERSION});"
                " upgrade pitstop or point PITSTOP_HOME elsewhere")


def _savepoint(local: threading.local, conn: sqlite3.Connection, depth: int) -> Iterator[sqlite3.Connection]:
    name = f"pitstop_{depth}"
    conn.execute(f"SAVEPOINT {name}")
    local.depth = depth + 1
    try:
        try:
            yield conn
        except BaseException:
            _rollback_to_quietly(conn, name)
            raise
        try:
            conn.execute(f"RELEASE {name}")
        except BaseException:
            _rollback_to_quietly(conn, name)
            raise
    finally:
        local.depth = depth


def _rollback_quietly(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ROLLBACK")
    except Exception:
        pass


def _rollback_to_quietly(conn: sqlite3.Connection, name: str) -> None:
    # ROLLBACK TO keeps the savepoint on the stack; RELEASE pops it so the outer transaction continues cleanly.
    try:
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
    except Exception:
        pass
