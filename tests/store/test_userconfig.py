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


@pytest.mark.parametrize("url", ["", "j.example/v1", "ftp://bot:sk-secret@j.example/v1", "https://", "http:///v1",
                                 "file:///etc/passwd", "https://[bad"])
def test_empty_judge_base_url_rejected(tmp_path, url):
    p = tmp_path / "config.toml"
    p.write_text(f'[judge]\nbase_url = "{url}"\n')
    with pytest.raises(ConfigInvalid, match="judge base_url") as exc:
        load_user_config(p)
    assert "sk-secret" not in str(exc.value) and "j.example" not in str(exc.value)  # the value is never echoed
    p.write_text('[judge]\nbase_url = "http://localhost:8000/v1"\n')
    assert load_user_config(p).judge.base_url == "http://localhost:8000/v1"


@pytest.mark.parametrize("key", ["api_key_evn", "modle"])
def test_judge_rejects_unknown_keys(tmp_path, key):
    p = tmp_path / "config.toml"
    p.write_text(f'[judge]\n{key} = "sk-secret-value"\n')
    with pytest.raises(ConfigInvalid, match=f"judge: unknown key.*{key}") as exc:
        load_user_config(p)
    assert "sk-secret-value" not in str(exc.value)


def test_alerts_empty_when_absent(tmp_path):
    from pitstop.userconfig import AlertsSpec

    assert load_user_config(tmp_path / "none.toml").alerts == AlertsSpec()
    a = AlertsSpec()
    assert (a.pagerduty, a.opsgenie, a.slack, a.email, a.webhook) == (None,) * 5
    assert a.routes == {"P0": ("pagerduty", "opsgenie", "slack", "webhook"), "P1": ("slack", "email", "webhook"),
                        "digest": ("email", "slack", "webhook")}


def test_alerts_tables_parse_with_defaults(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[alerts.pagerduty]\n[alerts.opsgenie]\nregion = "eu"\n[alerts.slack]\nwebhook_url_env = "MY_HOOK"\n'
                 '[alerts.webhook]\n[alerts.email]\nsmtp_host = "smtp.example.com"\nfrom_addr = "p@example.com"\n'
                 'to = ["a@example.com", "b@example.com"]\nusername = "p"\n'
                 '[alerts.routes]\nP0 = ["pagerduty", "slack"]\ndigest = []\n')
    a = load_user_config(p).alerts
    assert a.pagerduty.routing_key_env == "PAGERDUTY_ROUTING_KEY"
    assert (a.opsgenie.api_key_env, a.opsgenie.region) == ("OPSGENIE_API_KEY", "eu")
    assert a.slack.webhook_url_env == "MY_HOOK"
    assert a.webhook.url_env == "PITSTOP_WEBHOOK_URL"
    e = a.email
    assert (e.smtp_host, e.smtp_port, e.starttls, e.username, e.password_env, e.from_addr, e.to) == (
        "smtp.example.com", 587, True, "p", "SMTP_PASSWORD", "p@example.com", ("a@example.com", "b@example.com"))
    assert a.routes == {"P0": ("pagerduty", "slack"), "P1": ("slack", "email", "webhook"), "digest": ()}


@pytest.mark.parametrize("channel,key", [("pagerduty", "routing_key"), ("opsgenie", "api_key"),
                                         ("slack", "webhook_url"), ("email", "password"), ("webhook", "url")])
def test_inline_secret_rejected(tmp_path, channel, key):
    p = tmp_path / "config.toml"
    p.write_text(f'[alerts.{channel}]\n{key} = "https://hooks.slack.test/SECRET123"\n')
    with pytest.raises(ConfigInvalid, match=f"put secrets in environment variables; use {key}_env") as e:
        load_user_config(p)
    assert "SECRET123" not in str(e.value)


def test_unknown_route_channel_rejected(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[alerts.routes]\nP0 = ["pagerduty", "sms"]\n')
    with pytest.raises(ConfigInvalid, match="sms"):
        load_user_config(p)
    p.write_text('[alerts.routes]\nP2 = ["slack"]\n')
    with pytest.raises(ConfigInvalid, match="P2"):
        load_user_config(p)


@pytest.mark.parametrize("body", [
    "[alerts.sms]",
    '[alerts.opsgenie]\nregion = "apac"',
    "[alerts.slack]\nwebhook_url_env = 3",
    '[alerts.slack]\nwebhook_url_env = ""',
    '[alerts.slack]\nchannel = "#ops"',
    '[alerts.email]\nfrom_addr = "p@example.com"\nto = ["a@example.com"]',
    '[alerts.email]\nsmtp_host = "h"\nfrom_addr = "p@example.com"\nto = []',
    '[alerts.email]\nsmtp_host = "h"\nfrom_addr = "p@example.com"\nto = "a@example.com"',
    '[alerts.email]\nsmtp_host = "h"\nfrom_addr = "p@example.com"\nto = ["a@example.com"]\nsmtp_port = 70000',
    '[alerts.email]\nsmtp_host = "h"\nfrom_addr = "p@example.com"\nto = ["a@example.com"]\nsmtp_port = true',
    '[alerts.email]\nsmtp_host = "h"\nfrom_addr = "p@example.com"\nto = ["a@example.com"]\nstarttls = "yes"',
    '[alerts.routes]\nP1 = "slack"',
    'alerts = 3',
])
def test_alerts_config_validation(tmp_path, body):
    p = tmp_path / "config.toml"
    p.write_text(body + "\n")
    with pytest.raises(ConfigInvalid, match="alerts"):
        load_user_config(p)


@pytest.mark.parametrize("channel,key,env_key", [("slack", "url", "webhook_url_env"),
                                                 ("webhook", "webhook_url", "url_env"),
                                                 ("email", "api_key", "password_env"),
                                                 ("pagerduty", "api_key", "routing_key_env"),
                                                 ("opsgenie", "routing_key", "api_key_env")])
def test_inline_secret_message_names_the_channels_env_key(tmp_path, channel, key, env_key):
    p = tmp_path / "config.toml"
    p.write_text(f'[alerts.{channel}]\n{key} = "SECRET123"\n')
    with pytest.raises(ConfigInvalid, match=f"use {env_key}$") as e:
        load_user_config(p)
    assert "SECRET123" not in str(e.value)


@pytest.mark.parametrize("value", ["1BAD", "has space", "https://hooks.slack.test/SECRET123", "A-B", "X=Y"])
def test_env_names_must_be_identifiers(tmp_path, value):
    p = tmp_path / "config.toml"
    p.write_text(f'[alerts.slack]\nwebhook_url_env = "{value}"\n')
    with pytest.raises(ConfigInvalid, match="alerts.slack webhook_url_env") as e:
        load_user_config(p)
    assert value not in str(e.value)
    p.write_text('[alerts.slack]\nwebhook_url_env = "_MY_HOOK_2"\n')
    assert load_user_config(p).alerts.slack.webhook_url_env == "_MY_HOOK_2"
