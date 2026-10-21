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
