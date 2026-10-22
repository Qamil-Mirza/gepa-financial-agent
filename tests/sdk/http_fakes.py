"""Real SDK clients over an in-process HTTP transport (`httpx.MockTransport`; no sockets, no network).

`handler(request) -> httpx.Response` scripts the provider. The builders return plain dicts in the provider's
wire format, so the SDK parses them exactly as it would a real response.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

BASE_URL = "http://pitstop.test/v1"
MODEL = "openai/gpt-oss-120b"


def openai_client(handler, *, async_: bool = False, base_url: str = BASE_URL):
    import openai

    if async_:
        return openai.AsyncOpenAI(base_url=base_url, api_key="test", max_retries=0,
                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return openai.OpenAI(base_url=base_url, api_key="test", max_retries=0,
                         http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def json_response(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=body)


def sse_response(events: list[dict], *, named: bool = False) -> httpx.Response:
    """OpenAI style (`data: …` per event, then `data: [DONE]`), or Anthropic style (`event: <type>` + `data: …`)."""
    if named:
        body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    else:
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int | None = None) -> dict:
    u: dict[str, Any] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                         "total_tokens": prompt_tokens + completion_tokens}
    if cached_tokens is not None:
        u["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return u


def chat_completion(content: str | None = "hi", *, tool_calls: list[dict] | None = None, prompt_tokens: int = 11,
                    completion_tokens: int = 3, cached_tokens: int | None = None, finish_reason: str | None = None,
                    usage: bool = True, model: str = MODEL) -> dict:
    """A `chat.completion` body. `tool_calls` items are `{"name", "arguments"}`."""
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [{"id": f"call_{i}", "type": "function", "function": tc} for i, tc in enumerate(tool_calls)]
    body: dict[str, Any] = {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": msg,
                     "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop")}],
    }
    if usage:
        body["usage"] = _usage(prompt_tokens, completion_tokens, cached_tokens)
    return body


def chat_chunk(content: str | None = None, *, finish_reason: str | None = None, usage: tuple[int, int] | None = None,
               choices: bool = True, model: str = MODEL) -> dict:
    """A `chat.completion.chunk`. `choices=False` with `usage=(prompt, completion)` is the include_usage chunk."""
    delta = {"content": content} if content is not None else {}
    return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}] if choices else [],
            "usage": _usage(*usage) if usage is not None else None}
