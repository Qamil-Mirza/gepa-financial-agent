from __future__ import annotations

import pytest

from pitstop.userconfig import BudgetSpec, load_user_config

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
])
def test_bad_values_raise_config_invalid(tmp_path, body):
    from pitstop.userconfig import ConfigInvalid
    p = tmp_path / "config.toml"
    p.write_text(body)
    with pytest.raises(ConfigInvalid):
        load_user_config(p)


def test_unreadable_config_raises_config_invalid(tmp_path):
    from pitstop.userconfig import ConfigInvalid
    p = tmp_path / "config.toml"
    p.mkdir()  # reading a directory raises IsADirectoryError (OSError)
    with pytest.raises(ConfigInvalid, match="cannot read"):
        load_user_config(p)


def test_non_utf8_config_raises_config_invalid(tmp_path):
    from pitstop.userconfig import ConfigInvalid
    p = tmp_path / "config.toml"
    p.write_bytes(b"\xff\xfe[budgets]")
    with pytest.raises(ConfigInvalid):
        load_user_config(p)
