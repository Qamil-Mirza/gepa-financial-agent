from pitstop.correct.settings import Settings


def test_settings_from_env_defaults(monkeypatch):
    for k in ["CRUSOE_API_KEY", "CRUSOE_BASE_URL", "SUPERVISOR_MODEL", "REFLECTION_MODEL",
              "SUPERVISOR_MAX_CONCURRENCY", "SUPERVISOR_THINKING", "SUPERVISOR_JSON_MODE"]:
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env(load_dotenv_file=False)
    assert s.has_key is False
    assert s.base_url == "https://api.inference.crusoecloud.com/v1"
    assert s.supervisor_model == "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
    assert s.reflection_model == "deepseek-ai/DeepSeek-V4-Pro"
    assert s.max_concurrent_inferences == 4
    assert s.thinking is None
    assert s.json_mode is False


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("CRUSOE_API_KEY", "k$1")
    monkeypatch.setenv("SUPERVISOR_THINKING", "false")
    monkeypatch.setenv("SUPERVISOR_MAX_CONCURRENCY", "2")
    s = Settings.from_env(load_dotenv_file=False)
    assert s.has_key and s.crusoe_api_key == "k$1"
    assert s.thinking is False and s.max_concurrent_inferences == 2
    assert "k$1" not in repr(s)


def test_settings_dotenv_loaded_from_cwd(tmp_path):
    import os
    import subprocess
    import sys
    (tmp_path / ".env").write_text("REFLECTION_MODEL=from-dotenv\n")
    code = "from pitstop.correct.settings import Settings; print(Settings.from_env().reflection_model)"
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                         env={k: v for k, v in os.environ.items() if k != "REFLECTION_MODEL"})
    assert out.stdout.strip() == "from-dotenv", out.stderr[-500:]
