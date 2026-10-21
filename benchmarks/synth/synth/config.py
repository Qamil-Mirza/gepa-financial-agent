"""Paths and settings. Loads benchmarks/synth/.env so scripts work from any cwd."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

EVIDENCE_DIR = ROOT / "evidence"
YOUCOM_CACHE_DIR = EVIDENCE_DIR / "youcom_cache"
TRACES_DIR = ROOT / "traces"
DATA_DIR = ROOT / "data"

CRUSOE_BASE_URL = os.getenv("CRUSOE_BASE_URL", "https://api.inference.crusoecloud.com/v1")
WORKER_MODEL = os.getenv("WORKER_MODEL", "nvidia/Nemotron-3.5-Lightning-30B-A3B")
YOUCOM_URL = "https://api.you.com/v1/finance_research"


def crusoe_key() -> str:
    return os.environ["CRUSOE_API_KEY"]


def youcom_key() -> str | None:
    return os.getenv("YDC_API_KEY") or None
