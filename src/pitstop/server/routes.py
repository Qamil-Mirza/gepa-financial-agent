"""The dashboard's JSON endpoints: thin wrappers over `views.py`, registered on the app's `/api` router.

Workflow names travel as query parameters, never as path segments (decision 5): a name may hold `/`, `?`, `#`,
spaces or emoji.

Store connections: every endpoint is sync, so it runs in a threadpool thread, and the store opens one connection
per thread. Each endpoint is wrapped by `closes_connection`, which closes the serving thread's connection when the
endpoint returns or raises (in that same thread), so no pool thread keeps an idle connection between requests. A
FastAPI dependency could not do this: a sync dependency's setup and teardown may run in other pool threads than
the endpoint."""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Annotated, TypeVar

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from ..sdk.runtime import Runtime
from . import changes, graph, views
from .models import (Board, Changes, DismissResult, Explain, FeedItem, Graph, RunDetail, RunPage, Settings,
                     WorkflowDetail)
from .worker import Worker

F = TypeVar("F", bound=Callable)
_DISMISS_STATUS = {"dismissed": 200, "not_found": 404, "not_open": 409}
_ROW_ID = Path(ge=0, le=2**63 - 1)  # an SQLite integer


def closes_connection(store) -> Callable[[F], F]:
    """Decorate a sync endpoint so it closes the calling thread's store connection when it finishes."""
    def decorate(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            finally:
                store.close_local()
        return wrapper  # type: ignore[return-value]
    return decorate


def register(api: APIRouter, rt: Runtime, worker: Worker | None) -> None:
    """Add the read endpoints and Dismiss to `api` (the `/api` router)."""
    closing = closes_connection(rt.store)

    @api.get("/board", response_model=Board)
    @closing
    def get_board() -> Board:
        return views.board(rt)

    @api.get("/feed", response_model=list[FeedItem])
    @closing
    def get_feed(limit: int = 50) -> list[FeedItem]:
        return views.feed(rt, limit=limit)

    @api.get("/workflow", response_model=WorkflowDetail)
    @closing
    def get_workflow(name: Annotated[str, Query()]) -> WorkflowDetail:
        try:
            return views.workflow_detail(rt, name)
        except views.NotFound as e:
            raise HTTPException(404, str(e)) from None

    @api.post("/drifts/{drift_id}/dismiss", response_model=DismissResult,
              responses={404: {"model": DismissResult}, 409: {"model": DismissResult}})
    @closing
    def post_dismiss(drift_id: Annotated[int, _ROW_ID]) -> JSONResponse:
        status = views.dismiss(rt, drift_id)
        return JSONResponse(DismissResult(status=status).model_dump(), _DISMISS_STATUS[status])

    @api.get("/drifts/{drift_id}/explain", response_model=Explain)
    @closing
    def get_explain(drift_id: Annotated[int, _ROW_ID]) -> Explain:
        try:
            return views.explain(rt, drift_id)
        except views.NotFound as e:
            raise HTTPException(404, str(e)) from None
        except views.Unprocessable as e:
            raise HTTPException(422, str(e)) from None

    @api.get("/runs", response_model=RunPage)
    @closing
    def get_runs(workflow: str | None = None, limit: int = 50, before: float | None = None) -> RunPage:
        return views.runs_page(rt, workflow, limit=limit, before=before)

    @api.get("/runs/{run_id}", response_model=RunDetail)
    @closing
    def get_run(run_id: str) -> RunDetail:
        try:
            return views.run_detail(rt, run_id)
        except views.NotFound as e:
            raise HTTPException(404, str(e)) from None

    @api.get("/graph", response_model=Graph)
    @closing
    def get_graph(workflow: Annotated[str, Query()], run_id: str | None = None,
                  window: graph.Window = "24h") -> Graph:
        try:
            return graph.build_graph(rt.store, workflow, run_id=run_id or None, window=window)
        except views.NotFound as e:
            raise HTTPException(404, str(e)) from None

    @api.get("/changes", response_model=Changes)
    @closing
    def get_changes(workflow: Annotated[str, Query()], limit: int = 200) -> Changes:
        return changes.change_series(rt.store, workflow, limit=limit)

    @api.get("/settings", response_model=Settings)
    @closing
    def get_settings() -> Settings:
        return views.settings(rt, worker)
