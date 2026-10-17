import asyncio
from types import SimpleNamespace

import pytest

from gepa_financial_agent import crusoe_client as cc
from gepa_financial_agent.crusoe_client import (
    ConfigError, CrusoeClient, MockModelClient, ModelUnavailableError, estimate_cost_usd,
)


class Fake429(Exception):
    status_code = 429


def fake_response(text='{"ok": true}', finish="stop", prompt=10, completion=5, cached=3):
    msg = SimpleNamespace(content=text, reasoning_content=None, reasoning=None)
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)], usage=usage, model="m")


class FakeSync:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(cc.time, "sleep", lambda s: None)


def test_requires_key(monkeypatch):
    monkeypatch.delenv("CRUSOE_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        CrusoeClient(model="x")


def test_retries_then_succeeds():
    fake = FakeSync([Fake429(), Fake429(), fake_response()])
    c = CrusoeClient(model="deepseek-ai/Deepseek-V4-Flash", sync_client=fake, max_retries=3, thinking=False)
    r = c.complete([{"role": "user", "content": "hi"}], json_mode=True, seed=7)
    assert r.text == '{"ok": true}' and r.prompt_tokens == 10 and r.cached_tokens == 3 and not r.mocked
    assert len(fake.calls) == 3
    kw = fake.calls[-1]
    assert kw["response_format"] == {"type": "json_object"} and kw["seed"] == 7
    assert kw["extra_body"] == {"chat_template_kwargs": {"thinking": False}}
    snap = c.tracker.snapshot()
    assert snap["requests"] == 1 and snap["retries"] == 2 and snap["errors"] == 0
    assert snap["est_cost_usd"] is not None and snap["est_cost_usd"] > 0


def test_gives_up_after_max_retries():
    fake = FakeSync([Fake429()] * 3)
    c = CrusoeClient(model="x", sync_client=fake, max_retries=2)
    with pytest.raises(ModelUnavailableError):
        c.complete([{"role": "user", "content": "hi"}])
    assert c.tracker.snapshot()["errors"] == 1


def test_non_retryable_error_propagates():
    fake = FakeSync([ValueError("bad request")])
    c = CrusoeClient(model="x", sync_client=fake, max_retries=5)
    with pytest.raises(ValueError):
        c.complete([{"role": "user", "content": "hi"}])
    assert len(fake.calls) == 1


def test_truncated_flag():
    fake = FakeSync([fake_response(text="", finish="length")])
    c = CrusoeClient(model="x", sync_client=fake)
    assert c.complete([{"role": "user", "content": "hi"}]).truncated


def test_mock_client_is_marked_and_tracks_in_flight():
    m = MockModelClient(latency_s=0.02)

    async def go():
        return await asyncio.gather(*[m.acomplete([{"role": "user", "content": "x"}]) for _ in range(5)])

    results = asyncio.run(go())
    assert all(r.mocked for r in results)
    assert m.tracker.snapshot()["max_in_flight"] == 5


def test_estimate_cost():
    assert estimate_cost_usd("deepseek-ai/Deepseek-V4-Flash", 1_000_000, 0) == pytest.approx(0.14)
    assert estimate_cost_usd("unknown/model", 10, 10) is None
