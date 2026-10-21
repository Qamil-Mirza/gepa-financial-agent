"""Shared async Crusoe client: bounded concurrency, 429/503 backoff, hard call budget, usage accounting."""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import APIConnectionError, AsyncOpenAI, InternalServerError, RateLimitError

from . import config


class CallBudgetExceeded(RuntimeError):
    pass


@dataclass
class Completion:
    message: Any            # openai ChatCompletionMessage
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class CrusoeClient:
    def __init__(self, model: str = config.WORKER_MODEL, max_concurrent: int = 4, max_total_calls: int = 1000):
        self.model = model
        self._client = AsyncOpenAI(base_url=config.CRUSOE_BASE_URL, api_key=config.crusoe_key())
        self._sem = asyncio.Semaphore(max_concurrent)
        self.max_total_calls = max_total_calls
        self.total_calls = 0

    async def complete(self, messages: List[Dict[str, Any]], tools: Optional[list] = None, *,
                       temperature: float = 1.0, seed: Optional[int] = None, max_tokens: int = 1024,
                       max_tries: int = 6) -> Completion:
        if self.total_calls >= self.max_total_calls:
            raise CallBudgetExceeded(f"Crusoe call budget of {self.max_total_calls} exhausted")
        self.total_calls += 1
        kwargs: Dict[str, Any] = dict(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=0.95,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if tools:
            kwargs["tools"] = tools
        if seed is not None:
            kwargs["seed"] = seed
        async with self._sem:
            for attempt in range(max_tries):
                t0 = time.time()
                try:
                    resp = await self._client.chat.completions.create(**kwargs)
                    choice = resp.choices[0]
                    return Completion(
                        message=choice.message,
                        finish_reason=choice.finish_reason,
                        prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                        completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
                        latency_ms=int((time.time() - t0) * 1000),
                    )
                except (RateLimitError, InternalServerError, APIConnectionError):
                    if attempt == max_tries - 1:
                        raise
                    await asyncio.sleep(min(60.0, 2 ** attempt) + random.uniform(0, 0.5))
        raise RuntimeError("unreachable")
