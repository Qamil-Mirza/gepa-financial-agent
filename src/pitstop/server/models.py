"""The dashboard API's response models (Pydantic). Each endpoint declares one as its `response_model`, so the OpenAPI
schema (`export_openapi`, committed as `ui/openapi.json`) types the UI's client."""
from __future__ import annotations

from pydantic import BaseModel


class WorkerStatus(BaseModel):
    """The background worker: `enabled` is false under `--no-worker`. `last_scan_at` (epoch seconds) and
    `last_scan_runs` are from the last completed scan pass; `last_error` is the exception type name of the last
    failed step, cleared by the next completed scan."""
    enabled: bool
    running: bool
    last_scan_at: float | None = None
    last_scan_runs: int | None = None
    last_error: str | None = None
    scan_every_s: float | None = None


class Health(BaseModel):
    ok: bool
    version: str  # the pitstop package version
    schema_version: int  # the store's schema version
    home: str  # PITSTOP_HOME
    worker: WorkerStatus
