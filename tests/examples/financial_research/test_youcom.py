import asyncio
import json

import httpx
import pytest

from finresearch.research.youcom import (
    FINANCE_RESEARCH_URL, CacheMiss, FinanceResearchClient, ResearchBudgetExceeded, YouComError,
    parse_citation_indices,
)


def test_parse_citations():
    assert parse_citation_indices("a [[1]] b [[1, 2]] c [[3,4]]") == [1, 2, 3, 4]


def mock_client(fixture_response, counter):
    def handler(req: httpx.Request):
        counter.append(req)
        assert str(req.url) == FINANCE_RESEARCH_URL
        assert req.headers["X-API-Key"] == "k"
        body = json.loads(req.content)
        assert body["research_effort"] == "deep"
        return httpx.Response(200, json=fixture_response)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_live_then_cached(tmp_path, fixture_response):
    calls = []
    rc = FinanceResearchClient(api_key="k", cache_dir=tmp_path, http_client=mock_client(fixture_response, calls))
    r1 = await rc.research("q1")
    r2 = await rc.research("q1")
    assert len(calls) == 1 and not r1.from_cache and r2.from_cache
    assert r1.cited_indices == [1, 2] and r1.sources[1].excerpts == ["x"]
    assert rc.stats()["est_cost_usd"] == 0.11


@pytest.mark.asyncio
async def test_replay_mode_never_calls(tmp_path, fixture_response):
    calls = []
    rc = FinanceResearchClient(api_key="k", cache_dir=tmp_path, mode="replay",
                               http_client=mock_client(fixture_response, calls))
    with pytest.raises(CacheMiss):
        await rc.research("q1")
    assert not calls


@pytest.mark.asyncio
async def test_budget_cap(tmp_path, fixture_response):
    calls = []
    rc = FinanceResearchClient(api_key="k", cache_dir=tmp_path, max_live_calls=1,
                               http_client=mock_client(fixture_response, calls))
    await rc.research("q1")
    with pytest.raises(ResearchBudgetExceeded):
        await rc.research("q2")
    assert len(calls) == 1


def slow_client(fixture_response, counter, *, status=200):
    """Each call takes a moment, so concurrent requests overlap; records every query that reached the API."""
    async def handler(req: httpx.Request):
        counter.append(json.loads(req.content)["input"])
        await asyncio.sleep(0.05)
        return httpx.Response(status, json=fixture_response if status == 200 else {"error": "boom"})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_identical_requests_in_flight_share_one_call(tmp_path, fixture_response):
    calls = []
    rc = FinanceResearchClient(api_key="k", cache_dir=tmp_path, http_client=slow_client(fixture_response, calls))
    r1, r2, r3 = await asyncio.gather(rc.research("q1"), rc.research(" q1 "), rc.research("q1"))
    assert calls == ["q1"] and r1.text == r2.text == r3.text and r1.cache_key == r2.cache_key
    assert [r.from_cache for r in (r1, r2, r3)] == [False, True, True]  # only the first one paid
    st = rc.stats()
    assert (st["live_calls"], st["inflight_hits"], st["cache_hits"], st["est_cost_usd"]) == (1, 2, 0, 0.11)
    assert not rc._inflight
    await rc.research("q1")  # afterwards: the disk cache
    assert rc.stats()["cache_hits"] == 1 and len(calls) == 1


@pytest.mark.asyncio
async def test_different_requests_in_flight_are_not_merged(tmp_path, fixture_response):
    calls = []
    rc = FinanceResearchClient(api_key="k", cache_dir=tmp_path, http_client=slow_client(fixture_response, calls))
    await asyncio.gather(rc.research("q1"), rc.research("q2"))
    assert sorted(calls) == ["q1", "q2"] and rc.stats()["inflight_hits"] == 0 and rc.stats()["live_calls"] == 2


@pytest.mark.asyncio
async def test_in_flight_error_reaches_every_waiter_then_clears(tmp_path, fixture_response):
    calls = []
    rc = FinanceResearchClient(api_key="k", cache_dir=tmp_path,
                               http_client=slow_client(fixture_response, calls, status=500))
    res = await asyncio.gather(rc.research("q1"), rc.research("q1"), return_exceptions=True)
    assert len(calls) == 1 and all(isinstance(r, YouComError) and "HTTP 500" in str(r) for r in res)
    assert not rc._inflight and not list(tmp_path.glob("*.json"))
    with pytest.raises(YouComError):  # a later request tries again instead of reusing the failure
        await rc.research("q1")
    assert len(calls) == 2
