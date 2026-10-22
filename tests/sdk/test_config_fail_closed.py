"""An invalid config.toml must never silently remove the hard budget: calls fail closed with Halted."""
import asyncio
import sqlite3

import pytest

import pitstop
from pitstop.config import pitstop_home
from pitstop.sdk import runtime as runtime_mod
from pitstop.sdk.context import workflow
from pitstop.sdk.errors import Halted
from pitstop.userconfig import ConfigInvalid, load_user_config
from tests.sdk.fakes import FakeAsyncOpenAI, FakeOpenAI, make_response

MODEL = "openai/gpt-oss-120b"
MSGS = [{"role": "user", "content": "hi"}]

BAD_PERIOD = '[budgets.all]\nlimit_usd = 0.000001\nperiod = "weekly"\n'
BAD_LIMIT = '[budgets.all]\nlimit_usd = "lots"\nperiod = "total"\n'
BAD_TZ = 'timezone = "Mars/Olympus_Mons"\n[budgets.all]\nlimit_usd = 0.000001\nperiod = "day"\n'


def _write_home_config(text: str) -> None:
    home = pitstop_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(text)


def test_invalid_config_fails_closed():
    _write_home_config(BAD_PERIOD)
    fake = FakeOpenAI([make_response("hi")])
    client = pitstop.wrap(fake)
    with pytest.raises(Halted, match="pitstop config invalid: .*weekly"):
        client.chat.completions.create(model=MODEL, messages=MSGS)
    assert fake.calls == []


def test_invalid_config_fails_closed_async():
    _write_home_config(BAD_PERIOD)
    fake = FakeAsyncOpenAI([make_response("hi")])
    client = pitstop.wrap(fake)
    with pytest.raises(Halted, match="pitstop config invalid"):
        asyncio.run(client.chat.completions.create(model=MODEL, messages=MSGS))
    assert fake.calls == []


def test_invalid_timezone_fails_closed():
    _write_home_config(BAD_TZ)
    fake = FakeOpenAI([make_response("hi")])
    with pytest.raises(Halted, match="pitstop config invalid: .*Mars/Olympus_Mons"):
        pitstop.wrap(fake).chat.completions.create(model=MODEL, messages=MSGS)
    assert fake.calls == []


def test_non_numeric_limit_fails_closed():
    _write_home_config(BAD_LIMIT)
    fake = FakeOpenAI([make_response("hi")])
    with pytest.raises(Halted, match="pitstop config invalid"):
        pitstop.wrap(fake).chat.completions.create(model=MODEL, messages=MSGS)
    assert fake.calls == []


@pytest.mark.parametrize("text", [BAD_PERIOD, BAD_LIMIT, BAD_TZ, "not = [valid toml",
                                  '[prices."m"]\ninput = 1.0\n', '[budgets]\nall = 5\n'])
def test_load_user_config_raises_config_invalid(tmp_path, text):
    p = tmp_path / "config.toml"
    p.write_text(text)
    with pytest.raises(ConfigInvalid) as ei:
        load_user_config(p)
    assert isinstance(ei.value, ValueError)  # explicit pitstop.init() keeps raising a ValueError


def test_explicit_init_raises_on_bad_timezone(tmp_path):
    (tmp_path / "config.toml").write_text(BAD_TZ)
    with pytest.raises(ValueError, match="timezone"):
        pitstop.init(home=tmp_path)


def test_workflow_enter_fails_closed():
    _write_home_config(BAD_PERIOD)
    ran = []
    with pytest.raises(Halted, match="pitstop config invalid"):
        with workflow("wf"):
            ran.append(1)
    assert ran == []


def test_tool_fails_closed_inside_workflow(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text("")
    pitstop.init(home=tmp_path)
    called = []

    @pitstop.tool(side_effects=True)
    def send(x):
        called.append(x)

    with pytest.raises(Halted, match="pitstop config invalid"):
        with workflow("wf"):
            runtime_mod.reset_runtime()  # the config becomes invalid mid-run
            (tmp_path / "config.toml").write_text(BAD_PERIOD)
            monkeypatch.setenv("PITSTOP_HOME", str(tmp_path))
            send("x")
    assert called == []


def test_prompt_fails_closed():
    _write_home_config(BAD_PERIOD)
    with pytest.raises(Halted, match="pitstop config invalid"):
        pitstop.prompt("p", "default text")


def test_transient_store_error_stays_fail_open(monkeypatch):
    """A locked/busy DB is not a config error: the call proceeds unmetered (plan's fail-open rule)."""
    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(runtime_mod, "get_runtime", boom)
    monkeypatch.setattr("pitstop.sdk.wrap.call.get_runtime", boom)
    fake = FakeOpenAI([make_response("hi")])
    resp = pitstop.wrap(fake).chat.completions.create(model=MODEL, messages=MSGS)
    assert resp.choices[0].message.content == "hi" and len(fake.calls) == 1
