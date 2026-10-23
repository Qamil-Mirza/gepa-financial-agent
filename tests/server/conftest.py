"""Dashboard server fixtures: a runtime on a tmp home and an in-process TestClient (no real port)."""
import pytest

from pitstop.sdk.runtime import init

BASE_URL = "http://localhost:8787"  # an allowed Host; TestClient's default "testserver" fails the Host check


@pytest.fixture
def rt(tmp_path):
    return init(home=tmp_path / "home")


@pytest.fixture
def make_client(rt, tmp_path):
    """`make_client(**create_app_kwargs)`; the static dir defaults to an empty tmp dir (dashboard not built)."""
    from fastapi.testclient import TestClient

    from pitstop.server.app import create_app

    def make(**kw):
        kw.setdefault("static_dir", tmp_path / "static")
        return TestClient(create_app(rt, **kw), base_url=BASE_URL)

    return make


@pytest.fixture
def client(make_client):
    return make_client()
