CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, workflow TEXT, goal TEXT, criticality TEXT, sensitivity TEXT,
    fix_mode TEXT, started_at REAL, ended_at REAL, status TEXT, model TEXT, base_url TEXT,
    prompt_versions TEXT, pitstop_version TEXT, adapter TEXT, price_known INTEGER DEFAULT 1, budget_usd REAL,
    scanned_at REAL, fallback_from TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, run_id TEXT, ts REAL, agent TEXT, kind TEXT, name TEXT,
    payload TEXT, cost_usd REAL, parent_id INTEGER, span_id TEXT, parent_span_id TEXT);
CREATE TABLE IF NOT EXISTS drifts (
    id INTEGER PRIMARY KEY, run_id TEXT, workflow TEXT, ts REAL, level TEXT, kind TEXT,
    drift_type TEXT, evidence TEXT, action TEXT, status TEXT DEFAULT 'open', last_seen REAL,
    occurrences INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY, ts REAL, run_id TEXT, workflow TEXT, meter TEXT, model TEXT,
    base_url TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, usd REAL, state TEXT);
CREATE TABLE IF NOT EXISTS prompts (
    name TEXT, version INTEGER, text TEXT, sha TEXT, source TEXT, created_at REAL,
    active INTEGER, PRIMARY KEY(name, version));
-- alerts: kind immediate | digest | test; status pending | sent | failed; error is an exception type name or
-- "HTTP <status>", never a message, URL or body (they can hold secrets).
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY, drift_id INTEGER, level TEXT, channel TEXT, kind TEXT,
    ts REAL, status TEXT, error TEXT);
-- l2_state: the L2 change detector's saved state, one row per (workflow, metric).
CREATE TABLE IF NOT EXISTS l2_state (workflow TEXT, metric TEXT, state TEXT, updated_at REAL,
    PRIMARY KEY(workflow, metric));
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_drifts_wf_status ON drifts(workflow, status);
CREATE INDEX IF NOT EXISTS idx_ledger_ts ON ledger(ts);
CREATE INDEX IF NOT EXISTS idx_alerts_drift ON alerts(drift_id, channel);
-- @after-migrations: the statements below run after the migration steps, so they may name columns a step adds.
CREATE INDEX IF NOT EXISTS idx_runs_wf_started ON runs(workflow, started_at);
CREATE INDEX IF NOT EXISTS idx_runs_scan ON runs(scanned_at, ended_at);
CREATE INDEX IF NOT EXISTS idx_ledger_run ON ledger(run_id);
-- schema_version is written by Store._migrate: the latest version on a fresh DB, else after the migration steps.
