"""Root test config. The offline guard (pytest-socket) is set via addopts in pyproject.toml."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_pitstop_home(tmp_path_factory, monkeypatch):
    """No test may touch the real ~/.pitstop; the SDK runtime singleton is reset around every test."""
    from pitstop.sdk.runtime import reset_runtime

    # no test may pick up real keys (or real alert destinations) from the shell
    for key in ("CRUSOE_API_KEY", "YDC_API_KEY", "PAGERDUTY_ROUTING_KEY", "OPSGENIE_API_KEY", "SLACK_WEBHOOK_URL",
                "PITSTOP_WEBHOOK_URL", "SMTP_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PITSTOP_HOME", str(tmp_path_factory.mktemp("pitstop_home")))
    reset_runtime()
    yield
    reset_runtime()
