import json

import httpx
import pytest

from pitstop.research.youcom import (
    FINANCE_RESEARCH_URL, CacheMiss, FinanceResearchClient, ResearchBudgetExceeded, parse_citation_indices,
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
