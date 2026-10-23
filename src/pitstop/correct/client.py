from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..budget.prices import PRICES_PER_M, estimate_cost_usd  # noqa: F401  (re-exported for existing callers)
from ..sdk.wrap import meter, wrap
from .settings import CRUSOE_BASE_URL, Settings

log = logging.getLogger("pitstop.correct.client")


def _thinking_key(model: str) -> str:
    """chat_template_kwargs key that toggles reasoning: DeepSeek uses `thinking`, Nemotron and Qwen
    `enable_thinking`."""
    m = model.lower()
    return "enable_thinking" if ("nemotron" in m or "qwen" in m) else "thinking"


class ConfigError(RuntimeError):
    pass


class ModelUnavailableError(RuntimeError):
    pass


@dataclass
class CompletionResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    latency_ms: float
    finish_reason: str | None
    mocked: bool = False
    reasoning: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class UsageTracker:
    def __init__(self, model: str = "") -> None:
        self.model = model
        self._lock = threading.Lock()
        self.requests = self.errors = self.retries = 0
        self.prompt_tokens = self.completion_tokens = self.cached_tokens = 0
        self.total_latency_ms = 0.0
        self._in_flight = 0
        self.max_in_flight = 0

    @contextmanager
    def in_flight(self):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            yield
        finally:
            with self._lock:
                self._in_flight -= 1

    def record(self, r: CompletionResult) -> None:
        with self._lock:
            self.requests += 1
            self.prompt_tokens += r.prompt_tokens
            self.completion_tokens += r.completion_tokens
            self.cached_tokens += r.cached_tokens
            self.total_latency_ms += r.latency_ms

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "model": self.model,
                "requests": self.requests,
                "errors": self.errors,
                "retries": self.retries,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cached_tokens": self.cached_tokens,
                "mean_latency_ms": (self.total_latency_ms / self.requests) if self.requests else 0.0,
                "max_in_flight": self.max_in_flight,
                "est_cost_usd": estimate_cost_usd(
                    self.model, self.prompt_tokens, self.completion_tokens, self.cached_tokens
                ),
            }


class ModelClient(Protocol):
    model: str
    tracker: UsageTracker

    def complete(self, messages: list[dict[str, Any]], *, max_tokens: int = 1024, temperature: float = 0.0,
                 json_mode: bool = False, seed: int | None = None) -> CompletionResult: ...

    async def acomplete(self, messages: list[dict[str, Any]], *, max_tokens: int = 1024, temperature: float = 0.0,
                        json_mode: bool = False, seed: int | None = None) -> CompletionResult: ...


def _is_retryable(exc: BaseException) -> bool:
    from ..sdk.errors import Halted

    if isinstance(exc, Halted):  # a budget/drift halt is final: never retried
        return False
    status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    try:
        import openai
        return isinstance(exc, (openai.RateLimitError, openai.InternalServerError,
                                openai.APIConnectionError, openai.APITimeoutError))
    except ImportError:  # pragma: no cover
        return False


def _backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    return min(cap, base * (2 ** attempt)) + random.uniform(0, 0.5)


class CrusoeClient:
    """Shared OpenAI-compatible client for Crusoe Managed Inference with bounded concurrency and backoff."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str = CRUSOE_BASE_URL,
                 max_concurrent: int = 4, max_retries: int = 5, timeout: float = 120.0,
                 thinking: bool | None = None, tracker: UsageTracker | None = None,
                 sync_client: Any = None, async_client: Any = None) -> None:
        key = api_key or os.environ.get("CRUSOE_API_KEY")
        if not key and sync_client is None and async_client is None:
            raise ConfigError(
                "CRUSOE_API_KEY is not set. export CRUSOE_API_KEY='...' (single quotes) or use MockModelClient."
            )
        self.model = model
        self.base_url = base_url
        self.thinking = thinking
        self.max_retries = max_retries
        self.tracker = tracker or UsageTracker(model)
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._asem: asyncio.Semaphore | None = None
        self._asem_loop: asyncio.AbstractEventLoop | None = None
        # Only build real SDK clients when a key exists; an injected fake for one mode must not
        # force construction of the other (openai v3 raises on a missing key at construction).
        self._async_factory: Any = None
        if key and (sync_client is None or async_client is None):
            import openai
            sync_client = sync_client or openai.OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
            if async_client is None:
                # AsyncOpenAI's connection pool is bound to the event loop that first used it, so an owned
                # async client is rebuilt whenever the running loop changes (e.g. repeated asyncio.run).
                self._async_factory = lambda: openai.AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout,
                                                                 max_retries=0)
        # Injected clients are stored as given (callers may hold identity) and metered at each call.
        self._sync, self._async = sync_client, async_client
        self._async_client_loop: asyncio.AbstractEventLoop | None = None
        log.info("CrusoeClient model=%s base_url=%s max_concurrent=%d", model, base_url, max_concurrent)

    @classmethod
    def from_settings(cls, settings: Settings, role: str = "supervisor", **overrides: Any) -> "CrusoeClient":
        model = settings.supervisor_model if role == "supervisor" else settings.reflection_model
        kwargs: dict[str, Any] = dict(model=model, api_key=settings.crusoe_api_key, base_url=settings.base_url,
                                      max_concurrent=settings.max_concurrent_inferences,
                                      thinking=settings.thinking if role == "supervisor" else None)
        kwargs.update(overrides)
        return cls(**kwargs)

    def _kwargs(self, messages, max_tokens, temperature, json_mode, seed) -> dict[str, Any]:
        kw: dict[str, Any] = dict(model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if seed is not None:
            kw["seed"] = seed
        if self.thinking is not None:
            kw["extra_body"] = {"chat_template_kwargs": {_thinking_key(self.model): self.thinking}}
        return kw

    def _to_result(self, resp: Any, t0: float) -> CompletionResult:
        choice = resp.choices[0]
        msg = choice.message
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        r = CompletionResult(
            text=getattr(msg, "content", None) or "",
            model=getattr(resp, "model", None) or self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            finish_reason=getattr(choice, "finish_reason", None),
            reasoning=getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
        )
        self.tracker.record(r)
        return r

    def complete(self, messages, *, max_tokens=1024, temperature=0.0, json_mode=False, seed=None) -> CompletionResult:
        kw = self._kwargs(messages, max_tokens, temperature, json_mode, seed)
        with self._sem, self.tracker.in_flight():
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    with meter("correction"):
                        resp = wrap(self._sync).chat.completions.create(**kw)
                    return self._to_result(resp, t0)
                except Exception as e:  # noqa: BLE001
                    if not _is_retryable(e):
                        self.tracker.record_error()
                        raise
                    if attempt == self.max_retries:
                        self.tracker.record_error()
                        raise ModelUnavailableError(f"{self.model}: gave up after {attempt + 1} attempts: {e}") from e
                    self.tracker.record_retry()
                    delay = _backoff(attempt)
                    log.warning(
                        "retryable error from %s (attempt %d): %s; sleeping %.1fs", self.model, attempt + 1, e, delay
                    )
                    time.sleep(delay)
        raise AssertionError("unreachable")

    def _async_sem(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._asem is None or self._asem_loop is not loop:
            self._asem, self._asem_loop = asyncio.Semaphore(self._max_concurrent), loop
        return self._asem

    def _async_client(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._async_factory is not None and self._async_client_loop is not loop:
            self._async, self._async_client_loop = self._async_factory(), loop
        return self._async

    async def acomplete(
        self, messages, *, max_tokens=1024, temperature=0.0, json_mode=False, seed=None
    ) -> CompletionResult:
        kw = self._kwargs(messages, max_tokens, temperature, json_mode, seed)
        async with self._async_sem():
            with self.tracker.in_flight():
                for attempt in range(self.max_retries + 1):
                    t0 = time.perf_counter()
                    try:
                        with meter("correction"):
                            resp = await wrap(self._async_client()).chat.completions.create(**kw)
                        return self._to_result(resp, t0)
                    except Exception as e:  # noqa: BLE001
                        if not _is_retryable(e):
                            self.tracker.record_error()
                            raise
                        if attempt == self.max_retries:
                            self.tracker.record_error()
                            raise ModelUnavailableError(
                                f"{self.model}: gave up after {attempt + 1} attempts: {e}"
                            ) from e
                        self.tracker.record_retry()
                        delay = _backoff(attempt)
                        log.warning(
                            "retryable error from %s (attempt %d): %s; sleeping %.1fs",
                            self.model,
                            attempt + 1,
                            e,
                            delay,
                        )
                        await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    def list_models(self) -> list[dict[str, Any]]:
        page = self._sync.models.list()
        return [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in page.data]

    def model_info(self, model_id: str | None = None) -> dict[str, Any] | None:
        target = model_id or self.model
        for m in self.list_models():
            if m.get("id") == target:
                return m
        return None


_DEFAULT_MOCK_OUTPUT = json.dumps({"status": "ON_TRACK", "confidence": 0.5, "drift_type": None,
                                   "evidence_event_ids": [], "explanation": "mock default response",
                                   "recommended_action": "NONE"})


class MockModelClient:
    """Deterministic stand-in for tests and offline pipeline runs. Results are always marked mocked=True."""

    def __init__(self, responder: Callable[[list[dict[str, Any]]], str] | None = None, latency_s: float = 0.0,
                 model: str = "mock-model") -> None:
        self.model = model
        self.responder = responder or (lambda messages: _DEFAULT_MOCK_OUTPUT)
        self.latency_s = latency_s
        self.tracker = UsageTracker(model)
        self.calls: list[list[dict[str, Any]]] = []
        self._lock = threading.Lock()

    def _result(self, messages: list[dict[str, Any]], t0: float) -> CompletionResult:
        text = self.responder(messages)
        with self._lock:
            self.calls.append(messages)
        r = CompletionResult(text=text, model=self.model, prompt_tokens=len(json.dumps(messages)) // 4,
                             completion_tokens=len(text) // 4, cached_tokens=0,
                             latency_ms=(time.perf_counter() - t0) * 1000.0, finish_reason="stop", mocked=True)
        self.tracker.record(r)
        return r

    def complete(self, messages, *, max_tokens=1024, temperature=0.0, json_mode=False, seed=None) -> CompletionResult:
        t0 = time.perf_counter()
        with self.tracker.in_flight():
            if self.latency_s:
                time.sleep(self.latency_s)
            return self._result(messages, t0)

    async def acomplete(
        self, messages, *, max_tokens=1024, temperature=0.0, json_mode=False, seed=None
    ) -> CompletionResult:
        t0 = time.perf_counter()
        with self.tracker.in_flight():
            if self.latency_s:
                await asyncio.sleep(self.latency_s)
            return self._result(messages, t0)
