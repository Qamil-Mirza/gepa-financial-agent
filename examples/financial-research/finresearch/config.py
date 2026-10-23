"""Settings loaded from environment variables, and the example's packaged assets. No credentials live in code.

Importing `pitstop.config` loads a `.env` from the current directory (or a parent)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pitstop.config import pitstop_home

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS = PACKAGE_DIR / "assets"
PROMPTS = ASSETS / "prompts"
SCENARIOS = ASSETS / "scenarios"
SNAPSHOT = PACKAGE_DIR / "demo" / "snapshot.json"


@dataclass(frozen=True)
class Settings:
    crusoe_api_key: str | None
    crusoe_base_url: str
    supervisor_model: str
    worker_model: str
    report_model: str
    ydc_api_key: str | None
    youcom_effort: str
    max_research_calls: int
    cache_dir: Path
    runs_dir: Path


def load_settings() -> Settings:
    return Settings(
        crusoe_api_key=os.getenv("CRUSOE_API_KEY"),
        crusoe_base_url=os.getenv("CRUSOE_BASE_URL", "https://api.inference.crusoecloud.com/v1"),
        supervisor_model=os.getenv("PITSTOP_SUPERVISOR_MODEL", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"),
        worker_model=os.getenv("PITSTOP_WORKER_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"),
        report_model=os.getenv("PITSTOP_REPORT_MODEL", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"),
        ydc_api_key=os.getenv("YDC_API_KEY"),
        youcom_effort=os.getenv("PITSTOP_YOUCOM_EFFORT", "deep"),
        max_research_calls=int(os.getenv("PITSTOP_MAX_RESEARCH_CALLS", "12")),
        cache_dir=Path(os.getenv("PITSTOP_CACHE_DIR") or pitstop_home() / "cache" / "youcom").expanduser(),
        runs_dir=Path(os.getenv("PITSTOP_RUNS_DIR") or pitstop_home() / "runs").expanduser(),
    )
