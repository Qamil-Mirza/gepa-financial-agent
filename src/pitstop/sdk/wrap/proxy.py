"""Proxies: the wrapped client, the resources on the way to a metered method, and the streams it returns.

A `_Resource` stands for one attribute path from the client (`chat`, then `chat.completions`). It proxies an
attribute only while the path is a prefix of a metered method in the client's table; anything else is the
client's own object, returned unchanged. `with_raw_response` / `with_streaming_response`, at any level, keep the
path and switch the mode, so `client.with_raw_response.chat.completions.create` and
`client.chat.completions.with_raw_response.create` are both metered.

A "manager" kind method (the SDK's stream helpers, `chat.completions.stream` and `responses.stream`) reserves when
it is called and returns a manager proxy; entering it sends the request and yields a stream proxy that passes
every event through unchanged, and the call settles when the manager exits.
"""
from __future__ import annotations

import functools
import inspect
import logging
from typing import Any

from ..context import _safe
from .call import Mode, _Call
from .shapes import Kind, Observed, Shape, _g

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
        return _metered(client, attr, shape, mode) if kind == "call" else _managed(client, attr, shape, mode)
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


def _billed_error(call: _Call, exc: BaseException) -> None:
    """An error raised after the provider answered (openai's parse helpers raise LengthFinishReasonError /
    ContentFilterFinishReasonError carrying the billed `completion`) settles the call instead of releasing it:
    from the completion's usage, or at the full reservation when the error carries none."""
    if not isinstance(exc, Exception) or not hasattr(exc, "completion"):
        return
    call.got_result = True
    completion = _g(exc, "completion")
    if completion is None:
        call.worst_case = True  # settled at the reservation by the caller's finally
    else:
        call.complete(completion)


def _call(call: _Call, fn: Any, args: tuple) -> Any:
    handed_off = False
    try:
        try:
            result = _attempt(call, fn, args)
        except Exception as e:
            _safe(functools.partial(_billed_error, call, e))
            raise
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
        try:
            result = await _attempt_async(call, fn, args, first)
        except Exception as e:
            _safe(functools.partial(_billed_error, call, e))
            raise
        call.got_result = True
        parsed: Any = _UNSET
        if call.mode == "raw" and not call.stream:
            call.worst_case = True  # until parsed: an interrupted parse is charged the reservation
            try:
                parsed = result.parse()
                if inspect.isawaitable(parsed):
                    parsed = await parsed
            except Exception as e:
                parsed = _parse_failed(e)
        out = _deliver(call, result, parsed)
        handed_off = out is not result
        return out
    finally:
        if not handed_off:
            call.close()


_UNSET, _FAILED = object(), object()


def _parse_failed(exc: Exception) -> Any:
    """What a raw response's failed `parse()` still shows: the billed `completion` its error carries, if any
    (openai's LengthFinishReasonError), else _FAILED (charged the reservation)."""
    completion = _g(exc, "completion")
    if completion is not None:
        return completion
    log.warning("pitstop: could not parse a raw response; charged its reservation", exc_info=exc)
    return _FAILED


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
    except Exception as e:
        return _parse_failed(e)


def _deliver(call: _Call, result: Any, parsed: Any = _UNSET) -> Any:
    """Hand the provider's result to the caller: complete it now, or wrap a stream that settles at its end.
    Returns the result itself, or the stream proxy (then the proxy owns closing the call)."""
    call.got_result = True
    if call.retried:
        call.retry_succeeded()
    if call.mode == "raw" and not call.stream:
        if parsed is _UNSET:
            call.worst_case = True  # until parsed: an interrupted parse is charged the reservation
            parsed = _parse_raw(result)
        if parsed is not _FAILED:
            call.worst_case = False
            call.complete(parsed)
        return result
    if call.worst_case:
        return result  # settled at the full reservation when the caller's finally closes the call
    if call.stream:
        return _wrap_stream(result, call)
    call.complete(result)
    return result


# -- streams --------------------------------------------------------------------------------------------


def _billed_stream_error(call: _Call, exc: BaseException) -> None:
    """An error raised mid-stream that carries the billed `completion` (the chat stream helper raises
    LengthFinishReasonError at the finish chunk, before the usage chunk arrives): settled from the completion's
    usage, or at the full reservation when neither it nor the stream showed any."""
    if not isinstance(exc, Exception) or not hasattr(exc, "completion") or call.obs.prompt_tokens is not None:
        return
    probe = Observed()
    completion = _g(exc, "completion")
    if completion is not None:
        call.shape.read(completion, probe)
    if probe.prompt_tokens is not None and probe.completion_tokens is not None:
        obs = call.obs
        obs.prompt_tokens, obs.completion_tokens, obs.cached_tokens = (
            probe.prompt_tokens, probe.completion_tokens, probe.cached_tokens)
    else:
        call.worst_case = True


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
            except BaseException as e:  # StopIteration, provider errors, KeyboardInterrupt
                _safe(functools.partial(_billed_stream_error, call, e))
                call.close()
                raise
            if self._pitstop_feed(chunk):
                return chunk

    def _pitstop_feed(self, chunk: Any) -> bool:
        return self._pitstop_call.observe(chunk)

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
            except BaseException as e:  # StopAsyncIteration, provider errors, CancelledError
                _safe(functools.partial(_billed_stream_error, call, e))
                call.close()
                raise
            if self._pitstop_feed(chunk):
                return chunk

    def _pitstop_feed(self, chunk: Any) -> bool:
        return self._pitstop_call.observe(chunk)

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


# -- stream helpers ("manager" kind) ----------------------------------------------------------------------------

# Stream methods that consume the stream inside the SDK; the proxy drives the stream through itself first, so
# every event is observed.
_FINAL = frozenset({"get_final_completion", "get_final_response"})
# Accumulated results shown without PitStop's injected usage.
_SNAPSHOTS = frozenset({"current_completion_snapshot"})


def _managed(client: Any, fn: Any, shape: Shape, mode: Mode) -> Any:
    """`fn` (a stream helper) metered with `shape`. The helper returns its manager synchronously on both clients;
    the request goes out when the manager is entered."""
    def method(*args, **kwargs):
        call = _Call.begin(client, shape, kwargs, mode, stream=True)
        try:
            inner = fn(*args, **call.kwargs)
        except BaseException:  # bad arguments: nothing was sent
            call.close()
            raise
        def retry():  # a fresh manager with the kwargs left after the injection is dropped
            return fn(*args, **call.kwargs)
        return (_AsyncManager if hasattr(inner, "__aenter__") else _SyncManager)(inner, call, retry)
    try:
        functools.update_wrapper(method, fn)
    except Exception:
        pass
    return method


def _feed_event(call: _Call, event: Any) -> bool:
    """Feed one helper event to the shape (`Shape.helper_chunk` picks what it carries). False hides the event."""
    try:
        chunk = call.shape.helper_chunk(event)
    except Exception:
        log.warning("pitstop: could not read a stream helper event", exc_info=True)
        return True
    return True if chunk is None else call.observe(chunk)


def _caller_view(call: _Call, result: Any) -> Any:
    """An accumulated helper result as the unwrapped helper would show it: without the usage PitStop asked for."""
    if not call.injected:
        return result
    try:
        return call.shape.hide_injected(result)
    except Exception:
        log.warning("pitstop: could not hide injected usage", exc_info=True)
        return result


def _entered(call: _Call) -> None:
    call.got_result = True
    if call.retried:
        call.retry_succeeded()


class _ManagerBase:
    def __init__(self, inner: Any, call: _Call, retry: Any):
        self._pitstop_inner, self._pitstop_call, self._pitstop_retry = inner, call, retry
        self._pitstop_stream: Any = None

    @property
    def __class__(self):  # isinstance(manager, ChatCompletionStreamManager) stays True
        return type(self._pitstop_inner)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_pitstop"):
            raise AttributeError(name)
        return getattr(self._pitstop_inner, name)

    def __del__(self):
        # Never entered: release. Once entered, the stream proxy owns closing the call.
        try:
            if self._pitstop_stream is None:
                self._pitstop_call.close()
        except Exception:
            pass


class _SyncManager(_ManagerBase):
    def __enter__(self):
        call = self._pitstop_call
        try:
            try:
                stream = self._pitstop_inner.__enter__()
            except Exception as e:
                if not call.retryable(e):
                    raise
                call.drop_injection()
                self._pitstop_inner = self._pitstop_retry()
                stream = self._pitstop_inner.__enter__()
        except BaseException as e:
            _safe(functools.partial(_billed_error, call, e))
            call.close()
            raise
        _entered(call)
        self._pitstop_stream = _SyncHelperStream(stream, call)
        return self._pitstop_stream

    def __exit__(self, *exc):
        try:
            return self._pitstop_inner.__exit__(*exc)
        finally:
            self._pitstop_call.close()


class _AsyncManager(_ManagerBase):
    async def __aenter__(self):
        call = self._pitstop_call
        try:
            try:
                stream = await self._pitstop_inner.__aenter__()
            except Exception as e:
                if not call.retryable(e):
                    raise
                call.drop_injection()
                self._pitstop_inner = self._pitstop_retry()
                stream = await self._pitstop_inner.__aenter__()
        except BaseException as e:
            _safe(functools.partial(_billed_error, call, e))
            call.close()
            raise
        _entered(call)
        self._pitstop_stream = _AsyncHelperStream(stream, call)
        return self._pitstop_stream

    async def __aexit__(self, *exc):
        try:
            return await self._pitstop_inner.__aexit__(*exc)
        finally:
            self._pitstop_call.close()


class _SyncHelperStream(_SyncStream):
    """A helper's stream (ChatCompletionStream, ResponseStream): events pass through unchanged."""

    def _pitstop_feed(self, event: Any) -> bool:
        return _feed_event(self._pitstop_call, event)

    def until_done(self):
        for _ in self:
            pass
        return self

    def __getattr__(self, name: str) -> Any:
        attr = super().__getattr__(name)
        if name in _SNAPSHOTS:
            return _caller_view(self._pitstop_call, attr)
        if name not in _FINAL:
            return attr

        @functools.wraps(attr)
        def final(*args, **kwargs):
            self.until_done()
            return _caller_view(self._pitstop_call, attr(*args, **kwargs))
        return final


class _AsyncHelperStream(_AsyncStream):
    """Async counterpart of _SyncHelperStream."""

    def _pitstop_feed(self, event: Any) -> bool:
        return _feed_event(self._pitstop_call, event)

    async def until_done(self):
        async for _ in self:
            pass
        return self

    def __getattr__(self, name: str) -> Any:
        attr = super().__getattr__(name)
        if name in _SNAPSHOTS:
            return _caller_view(self._pitstop_call, attr)
        if name not in _FINAL:
            return attr

        @functools.wraps(attr)
        async def final(*args, **kwargs):
            await self.until_done()
            r = attr(*args, **kwargs)
            return _caller_view(self._pitstop_call, await r if inspect.isawaitable(r) else r)
        return final
