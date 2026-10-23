"""Local SQLite store (WAL) shared by the SDK, supervisor and UI.

A fresh DB is created at the latest schema by `schema.sql`. An older DB is upgraded in place by the steps in
`MIGRATIONS` (keyed by the version each step produces), all applied in one `BEGIN IMMEDIATE`. Only a DB that had
no tables when opened is stamped with the latest version outright; one with tables but no schema_version row is
treated as version 1 and migrated. To add a schema version: append a step, bump `SCHEMA_VERSION` and update
`schema.sql` (a fresh DB and a migrated one must match). `schema.sql` runs in two parts: everything above its
`-- @after-migrations` line before the migration steps, the rest after them, so an index on a column that a step
adds goes below that line. A DB written by a newer PitStop is refused with `StoreVersionError` before anything
touches it.

Each thread uses its own connection (`connect()`), opened on first use. A long-lived worker thread (a pool's)
calls `close_local()` when its job ends, so it never holds a connection between jobs.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 5
_SCHEMA = Path(__file__).with_name("schema.sql")
_AFTER_MIGRATIONS = "-- @after-migrations"


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


def _v4_alerts(conn: sqlite3.Connection) -> None:
    # Same statements as schema.sql (which already created them on an old DB): one row per alert send attempt.
    conn.execute("CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY, drift_id INTEGER, level TEXT,"
                 " channel TEXT, kind TEXT, ts REAL, status TEXT, error TEXT)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_drift ON alerts(drift_id, channel)")


def _v5_observe(conn: sqlite3.Connection) -> None:
    # Scan state for L2/L5 (runs.scanned_at, l2_state), the fallback tag, OTel span links, and the indexes the
    # scan and per-run queries need. Same statements as schema.sql.
    _add_column(conn, "runs", "scanned_at", "REAL")
    _add_column(conn, "runs", "fallback_from", "TEXT")
    _add_column(conn, "events", "span_id", "TEXT")
    _add_column(conn, "events", "parent_span_id", "TEXT")
    conn.execute("CREATE TABLE IF NOT EXISTS l2_state (workflow TEXT, metric TEXT, state TEXT, updated_at REAL,"
                 " PRIMARY KEY(workflow, metric))")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_wf_started ON runs(workflow, started_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_scan ON runs(scanned_at, ended_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_run ON ledger(run_id)")


MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    2: _v2_runs_budget,
    3: _v3_drift_occurrences,
    4: _v4_alerts,
    5: _v5_observe,
}


class _Connection(sqlite3.Connection):
    """A plain sqlite3 connection that can be weakly referenced, so a Store can count the ones it opened."""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._opened: weakref.WeakSet[sqlite3.Connection] = weakref.WeakSet()  # every thread's, until closed
        self._opened_lock = threading.Lock()
        conn = self.connect()
        try:
            version = self._read_version(conn)
            self._check_version(version)  # before schema.sql or a migration touches a newer DB
            fresh = version == 0 and conn.execute("SELECT 1 FROM sqlite_master WHERE type='table'").fetchone() is None
            conn.execute("PRAGMA journal_mode=WAL")
            before, marker, after = _SCHEMA.read_text().partition(_AFTER_MIGRATIONS)
            conn.executescript(before)
            self._migrate(fresh=fresh)
            conn.executescript(marker + after)  # the marker line is a comment
        except StoreVersionError:
            self.close_local()  # a refused store keeps no open handle on the DB file
            raise

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=5.0, factory=_Connection)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
            with self._opened_lock:
                self._opened.add(conn)
        return conn

    def close_local(self) -> None:
        """Close the calling thread's connection, if it has one; the next `connect()` opens a new one."""
        conn = getattr(self._local, "conn", None)
        self._local.conn = None
        if conn is not None:
            with self._opened_lock:
                self._opened.discard(conn)
            conn.close()

    def open_connections(self) -> int:
        """How many connections this store opened are still open, across all threads (for tests). A connection
        of a thread that ended without `close_local()` counts until it is garbage collected."""
        with self._opened_lock:
            return len(self._opened)

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

    def _migrate(self, fresh: bool = False) -> None:
        """Upgrade an older DB in place: every step above its version runs in one transaction, so a failing
        step leaves the DB at its old version. Safe across concurrent openers (the version is re-read under
        the write lock). No schema_version row means version 1 (tables from before versioning), unless the DB
        had no tables when this store opened it (`fresh`): schema.sql then created it at the latest shape."""
        current = self.schema_version()
        self._check_version(current)  # a newer PitStop may have migrated it since the constructor's check
        if current >= SCHEMA_VERSION:
            return
        with self.transaction(immediate=True) as c:
            current = self._read_version(c)
            self._check_version(current)
            if current == 0:
                current = SCHEMA_VERSION if fresh else 1
            for version in range(current + 1, SCHEMA_VERSION + 1):
                MIGRATIONS[version](c)
            c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

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
