from __future__ import annotations

import pytest

from pitstop.userconfig import BudgetSpec, ConfigInvalid, load_user_config

TOML = """
timezone = "America/Los_Angeles"
[budgets.crusoe]
base_url = "https://api.inference.crusoecloud.com/v1"
limit_usd = 50
period = "total"
[prices."nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"]
input = 0.30
output = 2.40
cached_input = 0.15
"""


def test_missing_config_is_empty(tmp_path):
    cfg = load_user_config(tmp_path / "none.toml")
    assert cfg.budgets == ()
    assert cfg.prices == {}


def test_config_parses_budgets_and_prices(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(TOML)
    cfg = load_user_config(p)
    assert cfg.timezone == "America/Los_Angeles"
    assert cfg.budgets[0] == BudgetSpec("crusoe", 50.0, "total", "https://api.inference.crusoecloud.com/v1")
    o = cfg.prices["nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"]
    assert (o.input, o.output, o.cached_input) == (0.30, 2.40, 0.15)


def test_invalid_period_raises(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(TOML.replace('"total"', '"weekly"'))
    with pytest.raises(ValueError, match="crusoe"):
        load_user_config(p)


@pytest.mark.parametrize("body", [
    '[budgets.x]\nlimit_usd = 1\nbase_url = 5',
    '[prices."m"]\ninput = -1\noutput = 1',
    '[prices."m"]\ninput = nan\noutput = 1',
    '[prices."m"]\ninput = 1\noutput = inf',
    '[prices."m"]\ninput = "cheap"\noutput = 1',
    '[prices."m"]\ninput = 1\noutput = 1\ncached_input = -0.5',
    '[prices."m"]\ninput = true\noutput = 1',
    '[prices."m"]\ninput = 1' + '0' * 400 + '\noutput = 1',
    '[budgets.x]\nlimit_usd = 1' + '0' * 400,
])
def test_bad_values_raise_config_invalid(tmp_path, body):
    p = tmp_path / "config.toml"
    p.write_text(body)
    with pytest.raises(ConfigInvalid):
        load_user_config(p)


def test_unreadable_config_raises_config_invalid(tmp_path):
    p = tmp_path / "config.toml"
    p.mkdir()  # reading a directory raises IsADirectoryError (OSError)
    with pytest.raises(ConfigInvalid, match="cannot read"):
        load_user_config(p)


def test_non_utf8_config_raises_config_invalid(tmp_path):
    p = tmp_path / "config.toml"
    p.write_bytes(b"\xff\xfe[budgets]")
    with pytest.raises(ConfigInvalid):
        load_user_config(p)


def test_judge_defaults_when_table_absent(tmp_path):
    from pitstop.userconfig import JudgeSpec

    want = JudgeSpec("openai/gpt-oss-120b", "https://api.inference.crusoecloud.com/v1", "CRUSOE_API_KEY", 10.0)
    assert load_user_config(tmp_path / "none.toml").judge == want
    p = tmp_path / "config.toml"
    p.write_text(TOML)
    assert load_user_config(p).judge == want


def test_judge_table_parses(tmp_path):
    from pitstop.userconfig import JudgeSpec

    p = tmp_path / "config.toml"
    p.write_text('[judge]\nmodel = "acme/judge"\nbase_url = "https://j.example/v1"\napi_key_env = "J_KEY"\n'
                 'timeout_s = 2\n')
    assert load_user_config(p).judge == JudgeSpec("acme/judge", "https://j.example/v1", "J_KEY", 2.0)
    p.write_text('[judge]\nmodel = "acme/judge"\n')
    j = load_user_config(p).judge
    assert (j.model, j.api_key_env, j.timeout_s) == ("acme/judge", "CRUSOE_API_KEY", 10.0)


@pytest.mark.parametrize("body", [
    "timeout_s = 0", 'timeout_s = "x"', "timeout_s = -1", "timeout_s = nan", "timeout_s = inf", "timeout_s = true",
    "model = 3", "base_url = 1", "api_key_env = []",
])
def test_judge_config_validation(tmp_path, body):
    p = tmp_path / "config.toml"
    p.write_text(f"[judge]\n{body}\n")
    with pytest.raises(ConfigInvalid, match="judge"):
        load_user_config(p)
