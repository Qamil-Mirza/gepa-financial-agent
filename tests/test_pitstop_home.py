from pathlib import Path

from pitstop.config import pitstop_home


def test_empty_pitstop_home_uses_default(monkeypatch):
    monkeypatch.setenv("PITSTOP_HOME", "")
    assert pitstop_home() == Path("~/.pitstop").expanduser().resolve()


def test_relative_pitstop_home_is_resolved_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PITSTOP_HOME", "state")
    home = pitstop_home()
    assert home.is_absolute()
    assert home == (tmp_path / "state").resolve()


def test_env_example_does_not_set_empty_pitstop_home():
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    assert not any(line.strip() == "PITSTOP_HOME=" for line in text.splitlines())
