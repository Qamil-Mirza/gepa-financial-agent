"""Package paths and the user state directory. No credentials live in code.

Importing it loads a `.env` from the current directory (or a parent), if python-dotenv is installed."""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional convenience
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:  # pragma: no cover
    pass

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS = PACKAGE_DIR / "assets"
PROMPTS = ASSETS / "prompts"


def pitstop_home() -> Path:
    """User-writable state (runs, research cache, pitstop.db). Override with PITSTOP_HOME.

    An empty PITSTOP_HOME (e.g. `PITSTOP_HOME=` in a .env) means the default, never the cwd; a relative one is
    resolved to an absolute path once, so later chdirs cannot move the state."""
    return Path(os.getenv("PITSTOP_HOME") or "~/.pitstop").expanduser().resolve()

