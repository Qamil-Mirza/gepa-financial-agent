"""Record-and-replay wrapper around the You.com Finance Research API.

Every real response is cached on disk (evidence/youcom_cache/<hash>.json). A lookup first tries an
exact normalized-query hit, then the most similar cached query, and only calls the live API when
nothing similar is cached and the live-call budget allows it. Replays never hit the network.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from . import config
from .textmatch import jaccard, normalize_query, same_question

log = logging.getLogger(__name__)

REUSE_THRESHOLD = 0.5   # similarity at which a cached answer is reused instead of a live call
MIN_RELEVANCE = 0.2     # below this, with no live budget, return "no results"
MAX_CONTENT_CHARS = 2500


@dataclass
class ResearchResult:
    query: str
    matched_query: Optional[str]
    similarity: float
    content: str
    sources: List[Dict[str, str]]
    origin: str  # "cache" | "live" | "none"


class BudgetExceeded(RuntimeError):
    pass


class ResearchBank:
    def __init__(self, cache_dir: Path = config.YOUCOM_CACHE_DIR, live_budget: int = 0,
                 max_concurrent: int = 2, timeout_s: float = 180.0):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.live_budget = live_budget
        self.live_calls = 0
        self._max_concurrent = max_concurrent
        self._sem: Optional[asyncio.Semaphore] = None  # created lazily: py3.9 binds it to the current loop
        self._inflight: Dict[str, asyncio.Future] = {}
        self._timeout = timeout_s
        self._entries: Dict[str, dict] = {}
        for p in sorted(self.cache_dir.glob("*.json")):
            e = json.loads(p.read_text())
            self._entries[e["normalized"]] = e

    # ---- public ----

    def __len__(self) -> int:
        return len(self._entries)

    async def research(self, query: str, allow_live: bool = True) -> ResearchResult:
        norm = normalize_query(query)
        if norm in self._entries:
            return self._result(query, self._entries[norm], 1.0, "cache")
        best, sim = self._best_match(query)
        if best is not None and same_question(query, best["query"], REUSE_THRESHOLD):
            return self._result(query, best, sim, "cache")
        if allow_live and self.live_calls < self.live_budget and config.youcom_key():
            entry = await self._fetch(query)
            return self._result(query, entry, 1.0, "live")
        if best is not None and sim >= MIN_RELEVANCE:
            return self._result(query, best, sim, "cache")
        return ResearchResult(query, None, sim, "No relevant results found for this query.", [], "none")

    async def prefetch(self, query: str) -> bool:
        """Fetch and cache a query unless an exact match exists. Returns True if a live call was made."""
        if normalize_query(query) in self._entries:
            return False
        await self._fetch(query)
        return True

    # ---- internals ----

    def _best_match(self, query: str):
        best, best_sim = None, 0.0
        for e in self._entries.values():
            s = jaccard(query, e["query"])
            if s > best_sim:
                best, best_sim = e, s
        return best, best_sim

    @staticmethod
    def _result(query: str, entry: dict, sim: float, origin: str) -> ResearchResult:
        out = entry["response"]["output"]
        content = out.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS] + " …[truncated]"
        sources = [{"url": s.get("url"), "title": s.get("title")} for s in out.get("sources", [])]
        return ResearchResult(query, entry["query"], round(sim, 3), content, sources, origin)

    async def _fetch(self, query: str) -> dict:
        norm = normalize_query(query)
        if norm in self._inflight:
            return await self._inflight[norm]
        if self.live_calls >= self.live_budget:
            raise BudgetExceeded(f"You.com live budget of {self.live_budget} calls exhausted")
        self.live_calls += 1
        fut = asyncio.get_running_loop().create_future()
        self._inflight[norm] = fut
        try:
            if self._sem is None:
                self._sem = asyncio.Semaphore(self._max_concurrent)
            async with self._sem:
                entry = await self._post(query)
            self._entries[norm] = entry
            path = self.cache_dir / (hashlib.sha1(norm.encode()).hexdigest()[:16] + ".json")
            path.write_text(json.dumps(entry, indent=2))
            fut.set_result(entry)
            return entry
        except BaseException as exc:
            fut.set_exception(exc)
            fut.exception()  # mark retrieved so waiters-less failures don't warn
            raise
        finally:
            del self._inflight[norm]

    async def _post(self, query: str, max_tries: int = 3) -> dict:
        key = config.youcom_key()
        if not key:
            raise RuntimeError("YDC_API_KEY not set")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(max_tries):
                t0 = time.time()
                try:
                    resp = await client.post(config.YOUCOM_URL, headers={"X-API-Key": key},
                                             json={"input": query, "research_effort": "deep"})
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_tries - 1:
                        await asyncio.sleep(2 ** attempt * 2)
                        continue
                    resp.raise_for_status()
                except httpx.TransportError:
                    if attempt == max_tries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt * 2)
                    continue
                elapsed = round(time.time() - t0, 1)
                log.info("you.com live call %.1fs: %s", elapsed, query)
                return {
                    "query": query,
                    "normalized": normalize_query(query),
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "elapsed_s": elapsed,
                    "research_effort": "deep",
                    "response": resp.json(),
                }
        raise RuntimeError("unreachable")
