import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitstop.llm import LLMClient, LLMResult  # noqa: E402


class FakeLLM(LLMClient):
    """Scripted stand-in for Crusoe. handler(system, user) -> str."""

    def __init__(self, handler, model="fake-model"):
        super().__init__(model, api_key=None)
        self.handler = handler
        self.calls = []

    async def complete(self, system, user, *, temperature=None):
        self.calls.append((system, user))
        r = LLMResult(text=self.handler(system, user), model=self.model, latency_s=0.01,
                      prompt_tokens=len(user) // 4, completion_tokens=50)
        self.usage.add(r)
        return r


# SYNTHETIC FIXTURE — not real financial data. Used only to test parsing and plumbing.
FIXTURE_RESPONSE = {
    "output": {
        "content": "FIXTURE: Example Corp data center revenue was $10.0B in FY2025, up 50%.[[1]] "
                   "Drivers included accelerator demand.[[1, 2]] Gaming revenue rose 5%.[[2]]",
        "content_type": "text",
        "sources": [
            {"url": "https://example.com/fixture-10k", "title": "FIXTURE 10-K"},
            {"url": "https://example.com/fixture-news", "title": "FIXTURE earnings news", "snippets": ["x"]},
        ],
    }
}


@pytest.fixture
def fixture_response():
    return json.loads(json.dumps(FIXTURE_RESPONSE))


@pytest.fixture
def FakeLLMClass():
    return FakeLLM
