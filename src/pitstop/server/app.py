"""The dashboard's FastAPI app: the JSON API under `/api`, the built React app for every other GET, and the guards.

Local only, no login (decision 2). Every HTTP request passes `_Guard` first:
- a Host check against DNS rebinding: the Host header's name, without its port (a bracketed IPv6 address keeps its
  brackets), must be one of `allowed_hosts`, else 400;
- a write guard: POST, PUT, PATCH and DELETE need `X-PitStop: 1`, else 403. No CORS headers are ever sent, so a
  cross-site page cannot send that header;
- errors: a locked or busy store (`sqlite3.OperationalError`) is 503 `{"error": "store busy, retry"}`; any other
  exception is 500 `{"error": "<ExceptionType>"}`, logged with its traceback here and never sent to the client.

Static serving (decision 12): `static_dir` (default `pitstop/server/static`, written by `npm --prefix ui run build`)
serves its files, e.g. `/assets/*`; any other non-API GET gets `index.html` (the SPA's own router reads the path).
A missing `/assets/` file is 404, as is every unknown `/api` path (JSON, never `index.html`). Without an
`index.html`, every non-API GET is a "Dashboard not built" page naming the build commands, and the API still works.

`create_app` only builds the app; `pitstop ui` binds the socket and runs uvicorn. A `worker` is started and stopped
with the app's lifespan and reported by `/api/health`."""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..budget.ledger import Ledger
from ..sdk.runtime import Runtime
from ..store import Store
from ..userconfig import load_user_config
from .models import Health, WorkerStatus
from .worker import Worker

log = logging.getLogger("pitstop.server")

STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_HOSTS = ("localhost", "127.0.0.1", "[::1]")
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

NOT_BUILT_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PitStop: dashboard not built</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 40rem; margin: 4rem auto; padding: 0 1rem;
line-height: 1.5">
<h1>Dashboard not built</h1>
<p>The PitStop server is running, but the dashboard's web files are missing. From the repository root, run:</p>
<pre>npm --prefix ui ci
npm --prefix ui run build</pre>
<p>then reload this page. The JSON API still answers, e.g. <a href="/api/health">/api/health</a>.</p>
</body>
</html>
"""


def normalize_host(host: str) -> str:
    """The form a Host header carries `host` in: lower case, an IPv6 address in brackets."""
    host = host.strip().lower()
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _host_name(header: str) -> str:
    """The Host header's name without its port ("" when malformed); a bracketed IPv6 address keeps its brackets."""
    header = header.strip().lower()
    if header.startswith("["):
        end = header.find("]")
        name, rest = header[:end + 1], header[end + 1:]
        if end == -1 or (rest and not (rest.startswith(":") and rest[1:].isdigit())):
            return ""
        return name
    name, sep, port = header.partition(":")
    return "" if sep and not port.isdigit() else name


def _is_busy(e: BaseException) -> bool:
    msg = str(e).lower()
    return isinstance(e, sqlite3.OperationalError) and ("locked" in msg or "busy" in msg)


class _Guard:
    """Pure ASGI middleware: the Host check, the write guard and JSON errors (see the module docstring)."""

    def __init__(self, app, allowed_hosts: frozenset[str]):
        self.app = app
        self.allowed = allowed_hosts

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):  # lifespan
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if _host_name(headers.get("host", "")) not in self.allowed:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await JSONResponse({"error": "invalid Host header"}, 400)(scope, receive, send)
            return
        if scope["type"] == "http" and scope["method"] in _WRITE_METHODS and \
                headers.get("x-pitstop", "").strip() != "1":
            await JSONResponse({"error": "missing X-PitStop header"}, 403)(scope, receive, send)
            return
        started = False

        async def send_tracked(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, send_tracked)
        except Exception as e:
            if scope["type"] != "http" or started:  # too late for an error response
                raise
            if _is_busy(e):
                response = JSONResponse({"error": "store busy, retry"}, 503)
            else:
                log.error("pitstop: %s %s failed (%s)", scope["method"], scope["path"], type(e).__name__,
                          exc_info=e)
                response = JSONResponse({"error": type(e).__name__}, 500)
            await response(scope, receive, send)


def _static_response(static_dir: Path, path: str) -> Response | None:
    """The response for a non-API GET of `path`, or None for a 404."""
    index = static_dir / "index.html"
    if not index.is_file():
        return HTMLResponse(NOT_BUILT_HTML)
    rel = path.lstrip("/")
    if rel:
        root = static_dir.resolve()
        target = (root / rel).resolve()
        if not target.is_relative_to(root):
            return None
        if target.is_file():
            return FileResponse(target)
        if rel.startswith("assets/"):
            return None
    return FileResponse(index, headers={"Cache-Control": "no-cache"})


def _is_api(path: str) -> bool:
    return path == "/api" or path.startswith("/api/")


def worker_status(worker: Worker | None) -> WorkerStatus:
    return WorkerStatus(enabled=False, running=False) if worker is None else WorkerStatus(**worker.status())


def create_app(rt: Runtime, *, static_dir: Path | None = None, allowed_hosts: tuple[str, ...] = DEFAULT_HOSTS,
               worker: Worker | None = None) -> FastAPI:
    static_dir = Path(static_dir) if static_dir is not None else STATIC_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if worker is not None:
            worker.start()
        try:
            yield
        finally:
            if worker is not None:
                await asyncio.to_thread(worker.stop)

    async def not_found(request: Request, exc: StarletteHTTPException) -> Response:
        """Unmatched non-API GETs are the SPA's (or the not-built page); everything else stays a JSON 404."""
        if request.method in ("GET", "HEAD") and not _is_api(request.url.path):
            response = _static_response(static_dir, request.url.path)
            if response is not None:
                return response
        return await http_exception_handler(request, exc)

    # no /docs or /redoc: their pages load assets from a CDN, and the dashboard makes no request to other hosts
    app = FastAPI(title="PitStop", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url="/api/openapi.json", exception_handlers={404: not_found})
    app.state.rt = rt
    app.state.worker = worker
    app.state.static_dir = static_dir
    app.add_middleware(_Guard, allowed_hosts=frozenset(normalize_host(h) for h in allowed_hosts if h.strip()))

    api = APIRouter(prefix="/api")

    @api.get("/health", response_model=Health)
    def health() -> Health:
        return Health(ok=True, version=__version__, schema_version=rt.store.schema_version(), home=str(rt.home),
                      worker=worker_status(worker))

    app.include_router(api)
    return app


def export_openapi() -> dict:
    """The API's OpenAPI schema, from an app on a throwaway runtime (Task 10 commits it as `ui/openapi.json`)."""
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        store = Store(home / "pitstop.db")
        try:
            config = load_user_config(home / "config.toml")  # absent: the defaults
            schema = create_app(Runtime(store, config, Ledger(store, config), home)).openapi()
        finally:
            store.close_local()
    return schema
