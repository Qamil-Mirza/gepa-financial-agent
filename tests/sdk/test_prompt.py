import json

import pitstop
from pitstop.sdk import prompt as prompt_mod
from pitstop.sdk.context import workflow
from pitstop.sdk.prompt import prompt, set_active_version
from pitstop.sdk.runtime import get_runtime


def _init(tmp_path):
    (tmp_path / "config.toml").write_text("")
    return pitstop.init(home=tmp_path)


def _rows(name):
    return [dict(r) for r in get_runtime().store.connect().execute(
        "SELECT * FROM prompts WHERE name=? ORDER BY version", (name,)).fetchall()]


def _gepa_v2(name, text="gepa text"):
    import hashlib
    with get_runtime().store.transaction(immediate=True) as c:
        c.execute("INSERT INTO prompts(name, version, text, sha, source, created_at, active)"
                  " VALUES (?,?,?,?,?,0,0)", (name, 2, text, hashlib.sha256(text.encode()).hexdigest(), "gepa"))
    set_active_version(name, 2)


def test_first_use_creates_v1_default(tmp_path):
    _init(tmp_path)
    p = prompt("p", "A")
    assert (p.version, p.source, p.text) == (1, "default", "A")
    assert len(p.sha) == 64
    rows = _rows("p")
    assert len(rows) == 1 and rows[0]["active"] == 1


def test_same_default_is_stable(tmp_path):
    _init(tmp_path)
    assert prompt("p", "A") == prompt("p", "A")
    assert len(_rows("p")) == 1


def test_edited_default_becomes_dev_edit_and_wins(tmp_path):
    _init(tmp_path)
    prompt("p", "A")
    _gepa_v2("p")
    assert prompt("p", "A").version == 2
    p = prompt("p", "B")
    assert (p.version, p.source, p.text) == (3, "dev_edit", "B")
    assert [r["active"] for r in _rows("p")] == [0, 0, 1]


def test_unchanged_default_keeps_active_gepa_version(tmp_path):
    _init(tmp_path)
    prompt("p", "A")
    _gepa_v2("p")
    p = prompt("p", "A")
    assert (p.version, p.source, p.text) == (2, "gepa", "gepa text")
    assert len(_rows("p")) == 2


def test_prompt_versions_recorded_on_run(tmp_path):
    _init(tmp_path)
    with workflow("wf") as ctx:
        prompt("a", "A")
        prompt("b", "B")
        assert ctx.prompt_versions == {"a": 1, "b": 1}
    row = get_runtime().store.connect().execute(
        "SELECT prompt_versions FROM runs WHERE id=?", (ctx.run_id,)).fetchone()
    assert json.loads(row[0]) == {"a": 1, "b": 1}


def test_store_failure_returns_default_v0(tmp_path, monkeypatch):
    _init(tmp_path)

    def boom():
        raise RuntimeError("down")

    monkeypatch.setattr(prompt_mod, "get_runtime", boom)
    p = prompt("p", "A")
    assert (p.version, p.text, p.source) == (0, "A", "default")


def test_revert_to_earlier_default_reactivates_it(tmp_path):
    _init(tmp_path)
    prompt("p", "A")
    assert prompt("p", "B").version == 2
    p = prompt("p", "A")
    assert (p.version, p.text) == (1, "A")
    assert len(_rows("p")) == 2
    assert [r["active"] for r in _rows("p")] == [1, 0]


def test_gepa_active_then_code_reverts_activates_stored_version(tmp_path):
    _init(tmp_path)
    prompt("p", "A")
    prompt("p", "B")  # v2 dev_edit
    _gepa_v3 = "gepa three"
    import hashlib
    with get_runtime().store.transaction(immediate=True) as c:
        c.execute("INSERT INTO prompts(name, version, text, sha, source, created_at, active)"
                  " VALUES ('p',3,?,?,'gepa',0,0)", (_gepa_v3, hashlib.sha256(_gepa_v3.encode()).hexdigest()))
    set_active_version("p", 3)
    assert prompt("p", "B").version == 3  # unchanged: gepa stays
    assert prompt("p", "A").version == 1  # code changed to a stored text
    assert [r["active"] for r in _rows("p")] == [1, 0, 0]


def test_set_active_version_rollback_and_unknown(tmp_path):
    _init(tmp_path)
    prompt("p", "A")
    _gepa_v2("p")
    set_active_version("p", 1)
    assert [r["active"] for r in _rows("p")] == [1, 0]
    import pytest
    with pytest.raises(ValueError):
        set_active_version("p", 9)
