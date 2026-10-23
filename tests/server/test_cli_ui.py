"""`pitstop ui` / `pitstop serve`: the optional extra, fail-fast startup checks and one real-port smoke test."""
import os
import socket
import sqlite3
import subprocess
import sys
import time

import httpx
import pytest

from pitstop.cli import main
from pitstop.config import pitstop_home
from pitstop.store import Store

_KEYS = ("CRUSOE_API_KEY", "YDC_API_KEY", "PAGERDUTY_ROUTING_KEY", "OPSGENIE_API_KEY", "SLACK_WEBHOOK_URL",
         "PITSTOP_WEBHOOK_URL", "SMTP_PASSWORD")
EXTRA_MSG = 'The dashboard needs the ui extra: pip install "pitstop[ui]"'


def _env(tmp_path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _KEYS}
    env.update(PITSTOP_HOME=str(tmp_path / "home"), PYTHONUNBUFFERED="1")
    return env


def _run(code: str, tmp_path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=_env(tmp_path), capture_output=True,
                          text=True, timeout=60)


@pytest.mark.parametrize("blocked", ["fastapi", "uvicorn"])
@pytest.mark.parametrize("cmd", ["ui", "serve"])
def test_ui_without_extra_exits_2(tmp_path, blocked, cmd):
    out = _run(f"import sys; sys.modules[{blocked!r}] = None\n"
               f"from pitstop.cli import main; main([{cmd!r}, '--no-worker'])", tmp_path)
    assert out.returncode == 2, out.stderr
    assert EXTRA_MSG in out.stdout
    assert "Traceback" not in out.stderr
    assert not (tmp_path / "home" / "pitstop.db").exists()  # checked before the store is opened


def test_ui_refuses_newer_store(capsys, monkeypatch):
    path = pitstop_home() / "pitstop.db"
    Store(path).close_local()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    import uvicorn

    monkeypatch.setattr(uvicorn.Server, "run", lambda *a, **k: pytest.fail("the server must not start"))
    for cmd in (["ui", "--no-open"], ["serve"]):
        with pytest.raises(SystemExit) as e:
            main(cmd)
        assert e.value.code == 2
        out = capsys.readouterr().out
        assert "Invalid PitStop config:" in out and "schema version 99" in out


def test_ui_refuses_invalid_config(capsys):
    (pitstop_home()).mkdir(parents=True, exist_ok=True)
    (pitstop_home() / "config.toml").write_text("[budgets\n")
    with pytest.raises(SystemExit) as e:
        main(["serve"])
    assert e.value.code == 2
    assert "Invalid PitStop config:" in capsys.readouterr().out


def test_ui_port_in_use_exits_2(capsys, monkeypatch):
    import uvicorn

    monkeypatch.setattr(uvicorn.Server, "run", lambda *a, **k: pytest.fail("the server must not start"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        with pytest.raises(SystemExit) as e:
            main(["ui", "--no-open", "--no-worker", "--port", str(port)])
    assert e.value.code == 2
    assert f"port {port} is in use; pass --port" in capsys.readouterr().out


def test_ui_starts_server_worker_and_browser(capsys, monkeypatch):
    """`ui` binds, prints the URL, opens the browser and runs uvicorn on its socket with the worker attached;
    `serve` does not open the browser; a non-loopback --host warns."""
    import webbrowser

    import uvicorn

    seen = {}

    def fake_run(server, sockets=None):
        seen.update(app=server.config.app, log_level=server.config.log_level, sockets=sockets)

    opened = []
    monkeypatch.setattr(uvicorn.Server, "run", fake_run)
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    main(["ui", "--port", "0", "--scan-every", "5"])
    sock = seen["sockets"][0]
    port = sock.getsockname()[1]
    sock.close()
    out = capsys.readouterr().out
    assert f"PitStop dashboard: http://127.0.0.1:{port}" in out
    assert "no authentication" not in out
    assert opened == [f"http://127.0.0.1:{port}"]
    assert seen["log_level"] == "warning"
    worker = seen["app"].state.worker
    assert worker.status()["scan_every_s"] == 5.0 and worker.status()["running"] is False

    opened.clear()
    main(["serve", "--port", "0", "--no-worker"])
    seen["sockets"][0].close()
    assert opened == []
    assert seen["app"].state.worker is None

    main(["serve", "--host", "0.0.0.0", "--port", "0", "--no-worker"])
    seen["sockets"][0].close()
    out = capsys.readouterr().out
    assert "shows prompts and tool arguments and has no authentication" in out


def test_import_pitstop_without_fastapi(tmp_path):
    out = _run("import sys\n"
               "import pitstop, pitstop.cli, pitstop.server\n"
               "assert 'fastapi' not in sys.modules and 'uvicorn' not in sys.modules, 'imported eagerly'\n"
               "sys.modules['fastapi'] = None; sys.modules['uvicorn'] = None\n"
               "from pitstop.cli import main\n"
               "for cmd in (['status'], ['drifts'], ['budget'], ['scan'], ['fallback'], ['alerts', 'digest']):\n"
               "    main(cmd)\n"
               "print('all ok')", tmp_path)
    assert out.returncode == 0, out.stderr
    assert "all ok" in out.stdout


def test_serve_smoke_subprocess(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "pitstop.cli", "serve", "--port", str(port), "--no-worker"],
                            cwd=tmp_path, env=_env(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    try:
        body = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and proc.poll() is None:
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=1)
                body = r.json()
                break
            except httpx.TransportError:
                time.sleep(0.1)
        assert body is not None, proc.stderr.read() if proc.poll() is not None else "no answer within 10 s"
        assert body["ok"] is True and body["worker"]["enabled"] is False
        assert body["home"] == str((tmp_path / "home").resolve())
        r = httpx.get(f"http://127.0.0.1:{port}/api/health", headers={"Host": "evil.example"}, timeout=1)
        assert r.status_code == 400
    finally:
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
    assert f"PitStop dashboard: http://127.0.0.1:{port}" in stdout
    assert "Traceback" not in stderr


@pytest.mark.parametrize("port", ["-1", "65536", "http"])
def test_ui_rejects_bad_port(port, capsys):
    with pytest.raises(SystemExit) as e:
        main(["serve", "--port", port])
    assert e.value.code == 2
    assert "--port" in capsys.readouterr().err


def test_ui_unknown_host_exits_2(capsys, monkeypatch):
    def no_such_host(*a, **k):
        raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known")

    monkeypatch.setattr(socket, "getaddrinfo", no_such_host)
    with pytest.raises(SystemExit) as e:
        main(["serve", "--host", "no-such-host.invalid", "--port", "0", "--no-worker"])
    assert e.value.code == 2
    assert "cannot listen on no-such-host.invalid:0 (gaierror" in capsys.readouterr().out
