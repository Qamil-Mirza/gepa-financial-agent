CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, workflow TEXT, goal TEXT, criticality TEXT, sensitivity TEXT,
    fix_mode TEXT, started_at REAL, ended_at REAL, status TEXT, model TEXT, base_url TEXT,
    prompt_versions TEXT, pitstop_version TEXT, adapter TEXT, price_known INTEGER DEFAULT 1, budget_usd REAL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, run_id TEXT, ts REAL, agent TEXT, kind TEXT, name TEXT,
    payload TEXT, cost_usd REAL, parent_id INTEGER);
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
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_drifts_wf_status ON drifts(workflow, status);
CREATE INDEX IF NOT EXISTS idx_ledger_ts ON ledger(ts);
CREATE INDEX IF NOT EXISTS idx_alerts_drift ON alerts(drift_id, channel);
INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '4');
