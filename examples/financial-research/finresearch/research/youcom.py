"""Financial Research Adapter for the You.com Finance Research API.

Key facts from the docs (https://you.com/docs/guides/finance-research):
  * POST https://api.you.com/v1/finance_research, key in the X-API-Key header.
    The same path on ydc-index.io returns 403 even with a valid key.
  * Each call is itself a multi-step research agent: slow (deep < 120s,
    exhaustive < 300s), no background mode, and priced per request.
  * Response: output.content (Markdown with [[1, 2]] citation markers),
    output.content_type, output.sources[{url, title, ...excerpts}].

Every response is cached on disk keyed by (endpoint, query, effort, schema),
so GEPA evaluations and demos replay without spending money. Identical requests
made while one is still in flight share that one paid call (`inflight_hits`).
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import httpx

from ..schemas import utc_now

FINANCE_RESEARCH_URL = "https://api.you.com/v1/finance_research"
EST_COST_USD = {"deep": 0.11, "exhaustive": 0.50}  # $110 / $500 per 1k requests
TIMEOUT_S = {"deep": 180.0, "exhaustive": 360.0}    # above documented latency ceilings
MAX_INPUT_CHARS = 40_000

_CITE_RE = re.compile(r"\[\[([\d,\s]+)\]\]")


class YouComError(Exception):
    pass


class CacheMiss(YouComError):
    pass


class ResearchBudgetExceeded(YouComError):
    pass


@dataclass
class ResearchSource:
    index: int  # 1-based, matches [[n]] markers
    url: str
    title: str
    excerpts: list[str] = field(default_factory=list)


@dataclass
class ResearchResult:
    query: str
    effort: str
    content: Any
    content_type: str
    sources: list[ResearchSource]
    cited_indices: list[int]
    retrieved_at: str
    cache_key: str
    from_cache: bool
    latency_s: float
    fixture: bool = False

    @property
    def text(self) -> str:
        return self.content if isinstance(self.content, str) else json.dumps(self.content)


def parse_citation_indices(text: str) -> list[int]:
    out: list[int] = []
    for m in _CITE_RE.finditer(text or ""):
        for part in m.group(1).split(","):
            part = part.strip()
            if part.isdigit() and int(part) not in out:
                out.append(int(part))
    return sorted(out)


def parse_response(raw: dict) -> tuple[Any, str, list[ResearchSource]]:
    out = raw.get("output", raw)
    content = out.get("content", "")
    content_type = out.get("content_type", "text")
    sources = []
    for i, s in enumerate(out.get("sources") or [], start=1):
        excerpts = s.get("excerpts") or s.get("snippets") or ([s["snippet"]] if s.get("snippet") else [])
        sources.append(
            ResearchSource(index=i, url=s.get("url", ""), title=s.get("title", ""), excerpts=list(excerpts)[:3])
        )
    return content, content_type, sources


class FinanceResearchClient:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        cache_dir: Path,
        mode: Literal["live", "replay"] = "live",
        effort: str = "deep",
        max_live_calls: int = 12,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        if effort not in EST_COST_USD:
            raise ValueError(f"unknown research_effort {effort!r}")
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.effort = effort
        self.max_live_calls = max_live_calls
        self.live_calls = 0
        self.cache_hits = 0
        self.inflight_hits = 0  # requests served by an identical call already in flight (no second payment)
        self.est_cost_usd = 0.0
        self._http = http_client
        self._lock = asyncio.Lock()
        self._inflight: dict[str, asyncio.Task] = {}

    def cache_key(self, query: str, output_schema: Optional[dict] = None) -> str:
        blob = json.dumps(
            {"url": FINANCE_RESEARCH_URL, "input": query.strip(), "effort": self.effort, "schema": output_schema},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _build(self, query, key, record, from_cache, latency) -> ResearchResult:
        content, ctype, sources = parse_response(record["response"])
        return ResearchResult(
            query=query,
            effort=self.effort,
            content=content,
            content_type=ctype,
            sources=sources,
            cited_indices=parse_citation_indices(content if isinstance(content, str) else ""),
            retrieved_at=record["retrieved_at"],
            cache_key=key,
            from_cache=from_cache,
            latency_s=latency,
            fixture=bool(record.get("fixture")),
        )

    async def research(self, query: str, output_schema: Optional[dict] = None) -> ResearchResult:
        query = (query or "").strip()
        if not query:
            raise YouComError("empty research query")
        if len(query) > MAX_INPUT_CHARS:
            raise YouComError("query exceeds 40,000 characters")
        key = self.cache_key(query, output_schema)
        path = self._cache_path(key)
        if path.exists():
            self.cache_hits += 1
            return self._build(query, key, json.loads(path.read_text()), True, 0.0)
        if (task := self._inflight.get(key)) is not None:  # same request already being paid for: share it
            self.inflight_hits += 1
            t0 = time.perf_counter()
            res = await asyncio.shield(task)  # errors reach every waiter; a cancelled waiter leaves the call running
            return dataclasses.replace(res, from_cache=True, latency_s=time.perf_counter() - t0)
        if self.mode == "replay":
            raise CacheMiss(f"no cached response for query (key {key}); run in live mode to record it")
        if not self.api_key:
            raise YouComError("YDC_API_KEY is not set")
        task = asyncio.ensure_future(self._live_call(query, key, path, output_schema))
        self._inflight[key] = task
        return await asyncio.shield(task)

    async def _live_call(self, query: str, key: str, path: Path, output_schema: Optional[dict]) -> ResearchResult:
        try:
            return await self._post(query, key, path, output_schema)
        finally:  # before the task completes, so a later request finds the cache file, not a finished task
            self._inflight.pop(key, None)

    async def _post(self, query: str, key: str, path: Path, output_schema: Optional[dict]) -> ResearchResult:
        async with self._lock:  # reserve budget atomically across concurrent agents
            if self.live_calls >= self.max_live_calls:
                raise ResearchBudgetExceeded(f"live You.com call budget ({self.max_live_calls}) exhausted")
            self.live_calls += 1
            self.est_cost_usd += EST_COST_USD[self.effort]

        body: dict[str, Any] = {"input": query, "research_effort": self.effort}
        if output_schema:
            body["output_schema"] = output_schema
        client = self._http or httpx.AsyncClient(timeout=TIMEOUT_S[self.effort])
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                FINANCE_RESEARCH_URL,
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                json=body,
                timeout=TIMEOUT_S[self.effort],
            )
        except httpx.TimeoutException as e:
            raise YouComError(f"You.com request timed out after {TIMEOUT_S[self.effort]}s") from e
        except httpx.HTTPError as e:
            raise YouComError(f"You.com request failed: {e}") from e
        finally:
            if self._http is None:
                await client.aclose()
        latency = time.perf_counter() - t0
        if resp.status_code != 200:
            raise YouComError(f"You.com HTTP {resp.status_code}: {resp.text[:300]}")
        record = {"request": body, "response": resp.json(), "retrieved_at": utc_now(), "latency_s": latency}
        path.write_text(json.dumps(record, indent=2))
        return self._build(query, key, record, False, latency)

    def stats(self) -> dict:
        return {
            "mode": self.mode,
            "effort": self.effort,
            "live_calls": self.live_calls,
            "cache_hits": self.cache_hits,
            "inflight_hits": self.inflight_hits,
            "est_cost_usd": round(self.est_cost_usd, 2),
            "max_live_calls": self.max_live_calls,
        }
