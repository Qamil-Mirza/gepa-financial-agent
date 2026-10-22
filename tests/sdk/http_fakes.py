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


# -- Responses API ------------------------------------------------------------------------------------------------


def _response_usage(input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> dict:
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens},
            "output_tokens_details": {"reasoning_tokens": 0}}


def _message_item(text: str) -> dict:
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def _function_item(name: str, arguments: str, i: int = 0) -> dict:
    return {"type": "function_call", "id": f"fc_{i}", "call_id": f"call_{i}", "name": name, "arguments": arguments,
            "status": "completed"}


def response_body(text: str | None = "hi", *, tool_calls: list[dict] | None = None, input_tokens: int = 13,
                  output_tokens: int = 5, cached_tokens: int = 0, status: str = "completed", usage: bool = True,
                  model: str = MODEL) -> dict:
    """A `response` object. `tool_calls` items are `{"name", "arguments"}` (function_call output items)."""
    output = [_message_item(text)] if text is not None else []
    output += [_function_item(tc["name"], tc["arguments"], i) for i, tc in enumerate(tool_calls or [])]
    body: dict[str, Any] = {"id": "resp_1", "object": "response", "created_at": 0, "model": model, "status": status,
                            "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                            "usage": _response_usage(input_tokens, output_tokens, cached_tokens) if usage else None}
    return body


def response_events(text: str = "Hello", *, tool_call: dict | None = None, input_tokens: int = 13,
                    output_tokens: int = 5, usage: bool = True, status: str = "completed",
                    model: str = MODEL) -> list[dict]:
    """The SSE events of a streamed response: `text` in two deltas, then optionally one function call (its
    arguments in two deltas), then the terminal `response.<status>` event carrying usage."""
    created = response_body(None, status="in_progress", usage=False, model=model)
    events: list[dict] = [
        {"type": "response.created", "response": created},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {**_message_item(""), "status": "in_progress", "content": []}},
        {"type": "response.content_part.added", "output_index": 0, "content_index": 0, "item_id": "msg_1",
         "part": {"type": "output_text", "text": "", "annotations": []}},
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "item_id": "msg_1",
         "delta": text[:3], "logprobs": []},
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "item_id": "msg_1",
         "delta": text[3:], "logprobs": []},
        {"type": "response.output_text.done", "output_index": 0, "content_index": 0, "item_id": "msg_1",
         "text": text, "logprobs": []},
        {"type": "response.output_item.done", "output_index": 0, "item": _message_item(text)},
    ]
    if tool_call is not None:
        args = tool_call["arguments"]
        events += [
            {"type": "response.output_item.added", "output_index": 1,
             "item": {**_function_item(tool_call["name"], ""), "status": "in_progress"}},
            {"type": "response.function_call_arguments.delta", "output_index": 1, "item_id": "fc_0",
             "delta": args[:4]},
            {"type": "response.function_call_arguments.delta", "output_index": 1, "item_id": "fc_0",
             "delta": args[4:]},
            {"type": "response.function_call_arguments.done", "output_index": 1, "item_id": "fc_0",
             "name": tool_call["name"], "arguments": args},
            {"type": "response.output_item.done", "output_index": 1,
             "item": _function_item(tool_call["name"], args)},
        ]
    final = response_body(text, tool_calls=[tool_call] if tool_call else None, input_tokens=input_tokens,
                          output_tokens=output_tokens, usage=usage, status=status, model=model)
    events.append({"type": f"response.{status}", "response": final})
    for i, e in enumerate(events):
        e["sequence_number"] = i
    return events


# -- legacy completions -------------------------------------------------------------------------------------------


def text_completion(text: str = "hi", *, prompt_tokens: int = 7, completion_tokens: int = 2, usage: bool = True,
                    model: str = MODEL) -> dict:
    body: dict[str, Any] = {"id": "cmpl-1", "object": "text_completion", "created": 0, "model": model,
                            "choices": [{"index": 0, "text": text, "finish_reason": "stop", "logprobs": None}]}
    if usage:
        body["usage"] = _usage(prompt_tokens, completion_tokens)
    return body


def text_completion_chunk(text: str | None = None, *, finish_reason: str | None = None,
                          usage: tuple[int, int] | None = None, choices: bool = True, model: str = MODEL) -> dict:
    """A streamed legacy completion chunk. `choices=False` with `usage` is the include_usage chunk."""
    return {"id": "cmpl-1", "object": "text_completion", "created": 0, "model": model,
            "choices": [{"index": 0, "text": text or "", "finish_reason": finish_reason, "logprobs": None}]
            if choices else [],
            "usage": _usage(*usage) if usage is not None else None}
