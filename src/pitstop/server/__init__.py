"""The local dashboard server (`pitstop ui` / `pitstop serve`): a FastAPI JSON API under `/api` over the SQLite store
the SDK writes, the built React app (`static/`, from `npm --prefix ui run build`) and a background scan worker.

It needs the optional `ui` extra (`uv sync` in the repo; `pip install "pitstop[ui]"` once published), so this
package imports nothing at import time: `app` and the CLI import FastAPI and uvicorn only when the dashboard is
started."""
