"""Root test config. The offline guard (pytest-socket) is set via addopts in pyproject.toml."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_pitstop_home(tmp_path_factory, monkeypatch):
    """No test may touch the real ~/.pitstop; the SDK runtime singleton is reset around every test."""
    from pitstop.sdk.runtime import reset_runtime

    monkeypatch.setenv("PITSTOP_HOME", str(tmp_path_factory.mktemp("pitstop_home")))
    reset_runtime()
    yield
    reset_runtime()
