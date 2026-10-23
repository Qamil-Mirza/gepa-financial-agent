"""The dashboard server's skeleton: health, the Host and write guards, JSON errors and static / SPA serving."""
import logging
import sqlite3
import time

import pytest

import pitstop
from pitstop.store import SCHEMA_VERSION

ORIGIN = "access-control-allow-origin"


def _built(tmp_path):
    static = tmp_path / "built"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>PitStop</title><div id=root></div>")
    (static / "assets" / "app.js").write_text("console.log('pitstop')")
    return static


def test_health_reports_worker_and_schema(rt, make_client):
    from pitstop.server.worker import Worker

    body = make_client().get("/api/health").json()
    assert body == {"ok": True, "version": pitstop.__version__, "schema_version": SCHEMA_VERSION,
                    "home": str(rt.home),
                    "worker": {"enabled": False, "running": False, "last_scan_at": None, "last_scan_runs": None,
                               "last_error": None, "scan_every_s": None}}

    worker = Worker(rt.store, scan_every_s=30)
    with make_client(worker=worker) as client:  # the app's lifespan starts and stops the worker
        deadline = time.monotonic() + 2
        while worker.status()["last_scan_at"] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        w = client.get("/api/health").json()["worker"]
        assert (w["enabled"], w["running"], w["last_scan_runs"], w["last_error"], w["scan_every_s"]) == (
            True, True, 0, None, 30.0)
        assert isinstance(w["last_scan_at"], float)
    assert worker.status()["running"] is False


def test_health_is_in_the_openapi_schema():
    from pitstop.server.app import export_openapi

    schema = export_openapi()
    assert "/api/health" in schema["paths"]
    assert {"Health", "WorkerStatus"} <= set(schema["components"]["schemas"])


def test_host_check_and_write_header(make_client):
    client = make_client(allowed_hosts=("localhost", "127.0.0.1", "[::1]", "192.168.1.5"))
    responses = []

    def get(host):
        r = client.get("/api/health", headers={"Host": host})
        responses.append(r)
        return r.status_code

    assert get("evil.example") == 400
    assert responses[-1].json() == {"error": "invalid Host header"}
    for bad in ("evil.example:8787", "localhost.evil.example", "127.0.0.1.evil.example", "::1", "[::2]:8787", ""):
        assert get(bad) == 400, bad
    for good in ("localhost:8787", "localhost", "LOCALHOST:8787", "127.0.0.1:8787", "[::1]:8787", "[::1]",
                 "192.168.1.5:8787"):
        assert get(good) == 200, good

    r = client.post("/api/drifts/1/dismiss")
    responses.append(r)
    assert (r.status_code, r.json()) == (403, {"error": "missing X-PitStop header"})
    for method in ("PUT", "PATCH", "DELETE"):
        responses.append(client.request(method, "/api/drifts/1"))
        assert responses[-1].status_code == 403, method
    responses.append(client.post("/api/drifts/1/dismiss", headers={"X-PitStop": "0"}))
    assert responses[-1].status_code == 403
    responses.append(client.post("/api/drifts/1/dismiss", headers={"X-PitStop": "1"}))
    assert responses[-1].status_code != 403  # past the guard (the endpoint itself arrives in a later task)
    responses.append(client.options("/api/health", headers={"Origin": "http://evil.example",
                                                            "Access-Control-Request-Method": "POST"}))
    responses.append(client.get("/api/health", headers={"Origin": "http://evil.example"}))
    responses.append(client.get("/"))
    assert all(ORIGIN not in r.headers for r in responses)


def test_spa_fallback_and_not_built_page(make_client, tmp_path):
    static = _built(tmp_path)
    (tmp_path / "secret.txt").write_text("outside the static dir")
    client = make_client(static_dir=static)
    for path in ("/", "/w?name=x", "/w?name=a%2Fb%3F%23&tab=graph", "/settings"):
        r = client.get(path)
        assert (r.status_code, r.text) == (200, (static / "index.html").read_text()), path
        assert r.headers["content-type"].startswith("text/html")
    r = client.get("/assets/app.js")
    assert (r.status_code, r.text) == (200, "console.log('pitstop')")
    assert "javascript" in r.headers["content-type"]
    assert client.get("/assets/missing.js").status_code == 404  # a missing asset is not answered with index.html
    for sneaky in ("/assets/..%2F..%2Fsecret.txt", "/..%2Fsecret.txt", "/%2e%2e/secret.txt"):
        assert "outside the static dir" not in client.get(sneaky).text, sneaky


def test_nul_in_static_path_is_404_without_traceback(make_client, tmp_path, caplog):
    client = make_client(static_dir=_built(tmp_path))
    for path in ("/%00", "/assets/%00", "/w%00x"):
        r = client.get(path)
        assert r.status_code == 404, (path, r.status_code, r.text)
        assert r.headers["content-type"] == "application/json"
    assert not [rec for rec in caplog.records if rec.levelname == "ERROR"]

    empty = tmp_path / "empty"
    empty.mkdir()
    client = make_client(static_dir=empty)
    for path in ("/", "/w?name=x", "/assets/app.js"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers["content-type"].startswith("text/html")
        assert "Dashboard not built" in r.text
        assert "npm --prefix ui ci" in r.text and "npm --prefix ui run build" in r.text
    assert client.get("/api/health").status_code == 200  # the API works without a build
    assert "Dashboard not built" in make_client(static_dir=tmp_path / "missing").get("/").text


def test_api_404_is_json_not_index(make_client, tmp_path):
    client = make_client(static_dir=_built(tmp_path))
    for path in ("/api/nope", "/api", "/api/", "/api/workflow/x?name=y"):
        r = client.get(path)
        assert r.status_code == 404, path
        assert r.headers["content-type"] == "application/json", path
        assert "<!doctype html>" not in r.text


@pytest.mark.parametrize("exc, status, body", [
    (sqlite3.OperationalError("database is locked"), 503, {"error": "store busy, retry"}),
    (sqlite3.OperationalError("database table is locked: drifts"), 503, {"error": "store busy, retry"}),
    (sqlite3.OperationalError("SQLITE_BUSY: busy"), 503, {"error": "store busy, retry"}),
    (sqlite3.OperationalError("no such table: runs"), 500, {"error": "OperationalError"}),
    (KeyError("secret-value"), 500, {"error": "KeyError"}),
])
def test_endpoint_errors_become_json(make_client, caplog, exc, status, body):
    client = make_client()

    def boom():
        raise exc

    client.app.add_api_route("/api/boom", boom)
    with caplog.at_level(logging.ERROR):
        r = client.get("/api/boom")
    assert (r.status_code, r.json()) == (status, body)
    assert "secret-value" not in r.text and "Traceback" not in r.text
    if status == 500:  # logged on the server with its traceback
        assert any(rec.exc_info and rec.exc_info[1] is exc for rec in caplog.records)
