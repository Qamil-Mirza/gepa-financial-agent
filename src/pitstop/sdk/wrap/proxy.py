"""Proxies: the wrapped client, the resources on the way to a metered method, and the streams it returns.

A `_Resource` stands for one attribute path from the client (`chat`, then `chat.completions`). It proxies an
attribute only while the path is a prefix of a metered method in the client's table; anything else is the
client's own object, returned unchanged. `with_raw_response` / `with_streaming_response`, at any level, keep the
path and switch the mode, so `client.with_raw_response.chat.completions.create` and
`client.chat.completions.with_raw_response.create` are both metered.
"""
from __future__ import annotations

import functools
import inspect
import logging
from typing import Any

from .call import Mode, _Call
from .shapes import Kind, Shape

log = logging.getLogger("pitstop")

# Client methods that return a new client; their result is wrapped too so it stays metered.
_REWRAP = frozenset({"with_options", "copy"})
_MODES: dict[str, Mode] = {"with_raw_response": "raw", "with_streaming_response": "streaming"}

Table = dict[tuple[str, ...], tuple[Shape, Kind]]


def _route(client: Any, inner: Any, path: tuple[str, ...], mode: Mode, table: Table, cache: dict,
           name: str) -> Any:
    """The attribute `name` of `inner` (at `path` from `client`): metered, proxied or the inner object itself."""
    attr = getattr(inner, name)
    sub = path + (name,)
    entry = table.get(sub)
    if entry is not None:
        shape, kind = entry
        return _metered(client, attr, shape, mode) if kind == "call" else attr
    if name in _MODES:
        sub, mode = path, _MODES[name]
    elif not any(k[:len(sub)] == sub for k in table):
        return attr
    cached = cache.get(name)
    if cached is None or cached._pitstop_inner is not attr:
        cached = cache[name] = _Resource(client, attr, sub, mode, table)
    return cached


class _Wrapped:
    __slots__ = ("_pitstop_client", "_pitstop_table", "_pitstop_cache", "__weakref__")

    def __init__(self, client: Any, table: Table):
        object.__setattr__(self, "_pitstop_client", client)
        object.__setattr__(self, "_pitstop_table", table)
        object.__setattr__(self, "_pitstop_cache", {})

    @property
    def __class__(self):  # isinstance(wrapped, OpenAI) stays True
        return type(self._pitstop_client)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_pitstop"):
            raise AttributeError(name)
        client = self._pitstop_client
        if name in _REWRAP:
            attr = getattr(client, name)
            if callable(attr):
                @functools.wraps(attr)
                def rewrap(*a, **k):
                    from . import wrap

                    return wrap(attr(*a, **k))
                return rewrap
            return attr
        return _route(client, client, (), "normal", self._pitstop_table, self._pitstop_cache, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._pitstop_client, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._pitstop_client, name)

    def __dir__(self):
        return dir(self._pitstop_client)

    def __repr__(self) -> str:
        return f"pitstop.wrap({self._pitstop_client!r})"

    def __enter__(self):
        r = self._pitstop_client.__enter__()
        return self if r is self._pitstop_client else r

    def __exit__(self, *exc):
        return self._pitstop_client.__exit__(*exc)

    async def __aenter__(self):
        r = await self._pitstop_client.__aenter__()
        return self if r is self._pitstop_client else r

    async def __aexit__(self, *exc):
        return await self._pitstop_client.__aexit__(*exc)


class _Resource:
    """A resource on the way to a metered method (e.g. `chat`, `chat.completions`, `with_raw_response.chat`)."""

    __slots__ = ("_pitstop_client", "_pitstop_inner", "_pitstop_path", "_pitstop_mode", "_pitstop_table",
                 "_pitstop_cache")

    def __init__(self, client: Any, inner: Any, path: tuple[str, ...], mode: Mode, table: Table):
        self._pitstop_client, self._pitstop_inner, self._pitstop_path = client, inner, path
        self._pitstop_mode, self._pitstop_table, self._pitstop_cache = mode, table, {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_pitstop"):
            raise AttributeError(name)
        return _route(self._pitstop_client, self._pitstop_inner, self._pitstop_path, self._pitstop_mode,
                      self._pitstop_table, self._pitstop_cache, name)

    def __dir__(self):
        return dir(self._pitstop_inner)

    def __repr__(self) -> str:
        return f"pitstop.wrap({self._pitstop_inner!r})"


# -- metered methods ----------------------------------------------------------------------------------------


def _is_async(fn: Any) -> bool:
    # openai's AsyncCompletions.create is a sync def (functools.wraps) around an async def, hence unwrap.
    try:
        return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(inspect.unwrap(fn))
    except Exception:
        return False


def _metered(client: Any, fn: Any, shape: Shape, mode: Mode) -> Any:
    """`fn` (a "call" kind method) metered with `shape`. A streaming-mode method returns its response manager
    synchronously even on an async client, so it stays a sync def."""
    if mode != "streaming" and _is_async(fn):
        async def method(*args, **kwargs):
            call = _Call.begin(client, shape, kwargs, mode)
            return await _finish_async(call, fn, args, None)
    else:
        def method(*args, **kwargs):
            return _call(_Call.begin(client, shape, kwargs, mode), fn, args)
    try:
        functools.update_wrapper(method, fn)
    except Exception:
        pass
    return method


def _attempt(call: _Call, fn: Any, args: tuple) -> Any:
    """Call `fn`; a 400 on PitStop's stream_options injection is retried once without it."""
    try:
        return fn(*args, **call.kwargs)
    except Exception as e:
        if not call.retryable(e):
            raise
    call.drop_injection()
    return fn(*args, **call.kwargs)


async def _attempt_async(call: _Call, fn: Any, args: tuple, first: Any) -> Any:
    async def once():
        r = fn(*args, **call.kwargs)
        return await r if inspect.isawaitable(r) else r

    try:
        return await (first if first is not None else once())
    except Exception as e:
        if not call.retryable(e):
            raise
    call.drop_injection()
    return await once()


def _call(call: _Call, fn: Any, args: tuple) -> Any:
    handed_off = False
    try:
        result = _attempt(call, fn, args)
        if call.mode != "streaming" and inspect.isawaitable(result):  # an async method we failed to detect
            handed_off = True
            return _finish_async(call, fn, args, result)
        out = _deliver(call, result)
        handed_off = out is not result
        return out
    finally:
        if not handed_off:
            call.close()


async def _finish_async(call: _Call, fn: Any, args: tuple, first: Any) -> Any:
    handed_off = False
    try:
        result = await _attempt_async(call, fn, args, first)
        call.got_result = True
        parsed: Any = _UNSET
        if call.mode == "raw" and not call.worst_case:
            try:
                parsed = result.parse()
                if inspect.isawaitable(parsed):
                    parsed = await parsed
            except Exception:
                parsed = _FAILED
                log.warning("pitstop: could not parse a raw response; charged its reservation", exc_info=True)
        out = _deliver(call, result, parsed)
        handed_off = out is not result
        return out
    finally:
        if not handed_off:
            call.close()


_UNSET, _FAILED = object(), object()


def _parse_raw(result: Any) -> Any:
    """`result.parse()` of a sync raw response, or _FAILED."""
    try:
        parsed = result.parse()
        if inspect.isawaitable(parsed):
            close = getattr(parsed, "close", None)
            if close is not None:
                close()
            raise TypeError("raw response parse() returned an awaitable on a sync call")
        return parsed
    except Exception:
        log.warning("pitstop: could not parse a raw response; charged its reservation", exc_info=True)
        return _FAILED


def _deliver(call: _Call, result: Any, parsed: Any = _UNSET) -> Any:
    """Hand the provider's result to the caller: complete it now, or wrap a stream that settles at its end.
    Returns the result itself, or the stream proxy (then the proxy owns closing the call)."""
    call.got_result = True
    if call.retried:
        call.retry_succeeded()
    if call.worst_case:
        return result  # settled at the full reservation when the caller's finally closes the call
    if call.mode == "raw":
        if parsed is _UNSET:
            parsed = _parse_raw(result)
        if parsed is _FAILED:
            call.worst_case = True
        else:
            call.complete(parsed)
        return result
    if call.stream:
        return _wrap_stream(result, call)
    call.complete(result)
    return result


# -- streams --------------------------------------------------------------------------------------------


def _wrap_stream(result: Any, call: _Call) -> Any:
    if hasattr(result, "__aiter__"):
        return _AsyncStream(result, call)
    if hasattr(result, "__iter__"):
        return _SyncStream(result, call)
    return result  # unknown shape: returned as is, settled by estimate in the caller's finally


class _SyncStream:
    """Yields the provider's chunks unchanged; settles the reservation when the stream ends or is dropped."""

    def __init__(self, inner: Any, call: _Call):
        self._pitstop_inner, self._pitstop_call, self._pitstop_it = inner, call, None

    @property
    def __class__(self):  # isinstance(stream, openai.Stream) stays True
        return type(self._pitstop_inner)

    def __iter__(self):
        return self

    def __next__(self):
        call = self._pitstop_call
        if self._pitstop_it is None:
            self._pitstop_it = iter(self._pitstop_inner)
        while True:
            try:
                chunk = next(self._pitstop_it)
            except BaseException:  # StopIteration, provider errors, KeyboardInterrupt
                call.close()
                raise
            if call.observe(chunk):
                return chunk

    def close(self) -> None:
        try:
            close = getattr(self._pitstop_inner, "close", None)
            if close is not None:
                close()
        finally:
            self._pitstop_call.close()

    def __enter__(self):
        enter = getattr(self._pitstop_inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *exc):
        try:
            exit_ = getattr(self._pitstop_inner, "__exit__", None)
            return exit_(*exc) if exit_ is not None else None
        finally:
            self._pitstop_call.close()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_pitstop"):
            raise AttributeError(name)
        return getattr(self._pitstop_inner, name)

    def __del__(self):
        try:
            self._pitstop_call.close()
        except Exception:
            pass


class _AsyncStream:
    """Async counterpart of _SyncStream."""

    def __init__(self, inner: Any, call: _Call):
        self._pitstop_inner, self._pitstop_call, self._pitstop_it = inner, call, None

    @property
    def __class__(self):  # isinstance(stream, openai.AsyncStream) stays True
        return type(self._pitstop_inner)

    def __aiter__(self):
        return self

    async def __anext__(self):
        call = self._pitstop_call
        if self._pitstop_it is None:
            self._pitstop_it = self._pitstop_inner.__aiter__()
        while True:
            try:
                chunk = await self._pitstop_it.__anext__()
            except BaseException:  # StopAsyncIteration, provider errors, CancelledError
                call.close()
                raise
            if call.observe(chunk):
                return chunk

    async def close(self) -> None:
        await self._pitstop_finish_with("close")

    async def aclose(self) -> None:
        await self._pitstop_finish_with("aclose")

    async def _pitstop_finish_with(self, method: str) -> None:
        try:
            fn = getattr(self._pitstop_inner, method, None)
            if fn is not None:
                r = fn()
                if inspect.isawaitable(r):
                    await r
        finally:
            self._pitstop_call.close()

    async def __aenter__(self):
        enter = getattr(self._pitstop_inner, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, *exc):
        try:
            exit_ = getattr(self._pitstop_inner, "__aexit__", None)
            return await exit_(*exc) if exit_ is not None else None
        finally:
            self._pitstop_call.close()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_pitstop"):
            raise AttributeError(name)
        return getattr(self._pitstop_inner, name)

    def __del__(self):
        try:
            self._pitstop_call.close()
        except Exception:
            pass
