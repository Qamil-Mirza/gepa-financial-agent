"""Scripted fake OpenAI-compatible clients for SDK tests (no network).

Each `create(**kw)` call records its kwargs in `.calls` and pops the next scripted item:
  - a response object (see make_response) is returned as is;
  - an Exception instance is raised;
  - a list of chunks is returned as a stream. An Exception inside the list is raised mid-iteration.
"""
from __future__ import annotations

from types import SimpleNamespace

BASE_URL = "https://fake.example/v1/"


def make_tool_call(name: str, arguments: str, id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(id=id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def make_usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int | None = None) -> SimpleNamespace:
    details = SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens is not None else None
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                           total_tokens=prompt_tokens + completion_tokens, prompt_tokens_details=details)


def make_response(content: str | None = None, tool_calls: list | None = None, prompt_tokens: int = 10,
                  completion_tokens: int = 5, cached_tokens: int | None = None,
                  finish_reason: str | None = None) -> SimpleNamespace:
    finish = finish_reason or ("tool_calls" if tool_calls else "stop")
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(index=0, message=msg, finish_reason=finish)],
                           usage=make_usage(prompt_tokens, completion_tokens, cached_tokens), model="fake")


def make_chunk(content: str | None = None, tool_calls: list | None = None,
               finish_reason: str | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)], usage=None)


def make_tool_delta(index: int, name: str | None = None, arguments: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(index=index, id=f"call_{index}" if name else None, type="function" if name else None,
                           function=SimpleNamespace(name=name, arguments=arguments))


def make_usage_chunk(prompt_tokens: int, completion_tokens: int, cached_tokens: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[], usage=make_usage(prompt_tokens, completion_tokens, cached_tokens))


class FakeStream:
    def __init__(self, chunks: list):
        self._chunks = list(chunks)
        self.closed = False
        self.response = SimpleNamespace(status_code=200)

    def __iter__(self):
        for c in self._chunks:
            if isinstance(c, BaseException):
                raise c
            yield c

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeAsyncStream:
    def __init__(self, chunks: list):
        self._chunks = list(chunks)
        self.closed = False
        self.response = SimpleNamespace(status_code=200)

    async def __aiter__(self):
        for c in self._chunks:
            if isinstance(c, BaseException):
                raise c
            yield c

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


class _Base:
    def __init__(self, responses: list, base_url: str | None = BASE_URL):
        self._responses = list(responses)
        self.calls: list[dict] = []
        if base_url is not None:
            self.base_url = base_url
        self.models = SimpleNamespace(list=lambda: ["fake-model"])

    def _next(self, kw: dict):
        self.calls.append(kw)
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeOpenAI(_Base):
    def __init__(self, responses: list, base_url: str | None = BASE_URL):
        super().__init__(responses, base_url)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        item = self._next(kw)
        return FakeStream(item) if isinstance(item, list) else item


class FakeAsyncOpenAI(_Base):
    def __init__(self, responses: list, base_url: str | None = BASE_URL):
        super().__init__(responses, base_url)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        item = self._next(kw)
        return FakeAsyncStream(item) if isinstance(item, list) else item
