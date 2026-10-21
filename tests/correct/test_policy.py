from pitstop.correct.policy import Policy, PolicyStore


def test_policy_version_is_content_hash():
    a, b = Policy.from_text("hello"), Policy.from_text("hello", name="other")
    assert a.version == b.version and len(a.version) == 12
    assert Policy.from_text("hello ").version != a.version


def test_store_roundtrip_and_promote(tmp_path):
    store = PolicyStore(tmp_path)
    (tmp_path / "baseline.txt").write_text("BASE")
    base = store.baseline()
    assert base.text == "BASE" and base.name == "baseline"
    v = store.save_version("NEW PROMPT", "gepa_test", metadata={"val_score": 0.9})
    assert (tmp_path / "versions" / "gepa_test.txt").read_text() == "NEW PROMPT"
    assert store.list_versions()["gepa_test"]["version"] == v.version
    assert store.list_versions()["gepa_test"]["metadata"] == {"val_score": 0.9}
    cur = store.promote("gepa_test")
    assert cur.text == "NEW PROMPT" and store.current().version == v.version
    assert store.load("gepa_test").text == "NEW PROMPT"
    assert store.load(str(tmp_path / "baseline.txt")).text == "BASE"
