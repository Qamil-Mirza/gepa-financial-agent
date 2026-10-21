"""Thin async client for Crusoe Managed Inference (OpenAI-compatible)."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass


class LLMError(Exception):
    pass


class JSONParseError(LLMError):
    def __init__(self, raw_text: str):
        super().__init__("model did not return a parseable JSON object")
        self.raw_text = raw_text


@dataclass
class LLMResult:
    text: str
    model: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class UsageTracker:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    def add(self, r: LLMResult) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.latency_s += r.latency_s

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_latency_s": round(self.latency_s, 2),
            "mean_latency_s": round(self.latency_s / self.calls, 2) if self.calls else 0.0,
        }


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_reasoning(text: str | None) -> str:
    """Remove reasoning blocks some Nemotron variants emit before the answer."""
    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_object(text: str) -> dict:
    text = strip_reasoning(text)
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        candidates.insert(0, fenced.group(1))
    bal = _first_balanced_object(text)
    if bal:
        candidates.append(bal)
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found")


class LLMClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout: float = 120.0,
        max_retries: int = 2,
        usage: UsageTracker | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage = usage or UsageTracker()
        self._client = None
        if api_key:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
            )

    async def complete(self, system: str, user: str, *, temperature: float | None = None) -> LLMResult:
        if self._client is None:
            raise LLMError("CRUSOE_API_KEY is not set")
        t0 = time.perf_counter()
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
        )
        usage = getattr(resp, "usage", None)
        r = LLMResult(
            text=resp.choices[0].message.content or "",
            model=self.model,
            latency_s=time.perf_counter() - t0,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        self.usage.add(r)
        return r

    async def complete_json(self, system: str, user: str, *, retries: int = 1) -> tuple[dict, LLMResult]:
        last = ""
        for attempt in range(retries + 1):
            prompt = user if attempt == 0 else (
                user + "\n\nYour previous reply was not a valid JSON object. Reply with ONLY the JSON object."
            )
            r = await self.complete(system, prompt)
            try:
                return extract_json_object(r.text), r
            except ValueError:
                last = r.text
        raise JSONParseError(last)
