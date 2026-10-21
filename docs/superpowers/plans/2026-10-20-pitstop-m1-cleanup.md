# PitStop Milestone 1: Cleanup and Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the hackathon repo into one installable `pitstop` package plus a `benchmarks/` folder. Delete the superseded stack and the committed generated data. Add a README and the usual open-source repo files. The existing demo, CLI and tests keep working.

**Architecture:**
- **Runtime.** The live PitStop runtime (`pitstop/pitstop/pitstop/`) moves to `src/pitstop/`. Its prompts and scenarios ship as package data under `src/pitstop/assets/`.
- **GEPA engine.** The live parts of `gepa_financial_agent/` (optimizer, sync client, settings, Policy) become `pitstop.correct`.
- **Benchmarks.** Detector-calibration tooling, the hard-pairs eval, the hard held-out set and the trace generator move to `benchmarks/`.
- **Deleted.** Everything else that is superseded.
- **Not yet.** No new runtime features in this milestone. SDK work starts in milestone 2.

**Tech Stack:** Python ≥3.11, uv, hatchling (src layout), pydantic 2, openai, httpx, rich, gepa==0.1.4, pytest + pytest-asyncio + pytest-socket, ruff, GitHub Actions, Mermaid (in the README).

**Spec:** `docs/superpowers/specs/2026-10-20-pitstop-runtime-design.md` (§9.2, §10 milestone 1, §12).

## Global Constraints

- Package name `pitstop`, src layout, `requires-python = ">=3.11"`, Apache-2.0 license.
- `gepa==0.1.4` stays pinned. The installed package uses no `sys.path` hacks.
- The repo is **public**. Never commit `.env`, keys or tokens. `.env.example` lists variable names only.
- Every move uses `git mv`, so history is kept.
- The old demo UI (`pitstop demo`, `demo.html`), the pit-wall UI and `pitstop serve` keep working until milestone 3 replaces them (spec §12).
- The README describes only what works at the end of this milestone. Everything else goes in a labeled Roadmap (spec §9.2).
- Work happens on branch `pitstop-redesign`. Do not push, open a PR or publish a release. Those are the owner's actions; the final task hands them the exact commands.
- Commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>`.

**Plan-level decisions (deviations from the spec's end-state map, to be confirmed by the owner at handoff):**
1. **The research swarm stays in the package for now.** It (`agents/`, `orchestrator.py`, `research/`, `demo/`) stays inside `src/pitstop/` this milestone. Milestone 2 moves it to `examples/financial-research/` when it is rewired onto the SDK, because moving it earlier would mean importing package internals from an example.
2. **`demo/snapshot.json` stays until milestone 3.** The old demo serves it, and the spec says the old demo must keep working until then.
3. **LLM clients drop from four to three, not one.** `agent/crusoe_client.py` is deleted. `pitstop.llm` (async runtime), `pitstop.correct.client` (sync, with backoff) and `benchmarks/synth/synth/crusoe.py` (the generator's own client) remain. Milestone 2 unifies them under the budget ledger, which needs a single client path anyway.
4. **The trace generator keeps its own `sys.path` lines in its scripts.** `benchmarks/synth/` is standalone dev tooling with a top-level `synth` package. The no-hack rule applies to the installed package.

## Review Focus

1. **Fresh clone, no `.env` and no keys.** `uv sync && uv run pytest`, `uv run pitstop --help` and `uv run pitstop demo` (snapshot mode) all work offline. Pinned in Task 5 (`test_suite_runs_without_keys`) and by CI.
2. **Installed as a wheel, not editable.** Prompts, scenarios, `snapshot.json` and the static HTML are inside the wheel. Pinned in Task 2 (`test_wheel_contains_package_data`).
3. **Run from a directory other than the repo root.** Asset paths do not depend on the current directory, and `.env` is loaded from the current directory. Pinned in Task 2 (`test_assets_resolve_from_any_cwd`, `test_dotenv_loaded_from_cwd`).
4. **Where runs are written.** The default runs and cache directories move to `$PITSTOP_HOME` (default `~/.pitstop`), and both `PITSTOP_HOME` and `PITSTOP_RUNS_DIR` are honored. Pinned in Task 2 (`test_settings_paths_follow_pitstop_home`).
5. **No test spends money.** The test suite makes no real network calls (pytest-socket, loopback only, for the demo-server tests). Pinned in Task 5.

---

### Task 1: Baseline, data archive, ignore rules, secret check

**Files:**
- Modify: `.gitignore`
- Create (git-ignored, local only): `dist-data/pitstop-hackathon-data-2026-10-20.tar.gz`

**Interfaces:**
- Produces: `docs/superpowers/plans/m1-baseline.txt` (git-ignored scratch) recording the pass/fail counts that later tasks compare against.

- [ ] **Step 1: Record the baseline.** Run each command and save the summary lines to `docs/superpowers/plans/m1-baseline.txt`:
  - `uv run pytest -q` (root `tests/`). Expected: 100 passed.
  - `cd pitstop/pitstop && uv run --with httpx --with rich pytest -q`
  - `cd synthetic-data && uv run --with httpx pytest -q`

  Any failures here are pre-existing. List them in the file so later tasks don't chase them.
- [ ] **Step 2: Archive the committed generated data before it is deleted.** Build the tarball from:
  - `synthetic-data/{traces,traces_archive,evidence,data}`
  - `financial-agent/test-synthetic-data/{traces,traces_archive,data,evidence}`
  - `pitstop/pitstop/evals/results`
  - `docs/results`

  Run: `tar -czf dist-data/pitstop-hackathon-data-2026-10-20.tar.gz <those paths>`, then `tar -tzf … | wc -l`. Expected: a non-zero file count. The owner uploads it as a GitHub release asset at handoff.
- [ ] **Step 3: Add the ignore rules.** Append to `.gitignore`: `.superpowers/`, `dist-data/`, `dist/`, `docs/superpowers/plans/m1-baseline.txt`, `.ruff_cache/`.
- [ ] **Step 4: Check history for secrets.** Run: `git log -p --all | grep -nE "(sk-[A-Za-z0-9]{20,}|CRUSOE_API_KEY=[^ \n]+|YDC_API_KEY=[^ \n]+|ghp_[A-Za-z0-9]{30,})" | grep -v "CRUSOE_API_KEY=$\|YDC_API_KEY=$" | head`. Expected: no output. If there is any hit, **stop and report it to the owner**: the key must be rotated. Do not rewrite history yourself.
- [ ] **Step 5: Commit.** `git add .gitignore && git commit -m "chore: ignore brainstorm, data archive and build dirs"`

---

### Task 2: One installable `pitstop` package (src layout)

**Files:**
- Move: `pitstop/pitstop/pitstop/` → `src/pitstop/` (all modules, subpackages, `demo/static/`, `ui/static/`, `demo/snapshot.json` from `pitstop/pitstop/demo/snapshot.json`)
- Move: `pitstop/pitstop/prompts/{supervisor_v0.txt,worker/,gepa/}` → `src/pitstop/assets/prompts/`
- Move: `pitstop/pitstop/scenarios/` → `src/pitstop/assets/scenarios/`
- Move: `pitstop/pitstop/tests/*` → `tests/runtime/`
- Modify: `pyproject.toml`, `src/pitstop/config.py`, `src/pitstop/cli.py`, `src/pitstop/demo/{pair,fixture,export,views,server}.py`, `src/pitstop/ui/server.py`, and every other `ROOT` user (`grep -rn "ROOT" src/pitstop`)
- Test: `tests/runtime/test_packaging.py` (new)

**Interfaces:**
- Produces, in `pitstop.config`:
  - `PACKAGE_DIR: Path` (the directory of `config.py`)
  - `ASSETS = PACKAGE_DIR / "assets"`
  - `PROMPTS = ASSETS / "prompts"`
  - `SCENARIOS = ASSETS / "scenarios"`
  - `SNAPSHOT = PACKAGE_DIR / "demo" / "snapshot.json"`
  - `def pitstop_home() -> Path`: returns `Path(os.getenv("PITSTOP_HOME", "~/.pitstop")).expanduser()`
  - `Settings` and `load_settings()`: the same fields as today. The defaults change to `cache_dir = pitstop_home() / "cache" / "youcom"` and `runs_dir = pitstop_home() / "runs"`. `PITSTOP_CACHE_DIR` and `PITSTOP_RUNS_DIR` still override them.
  - `ROOT` is removed.
- Produces a console script `pitstop = "pitstop.cli:main"`, with the subcommands unchanged: check, run, compare, timeline, serve, demo.

- [ ] **Step 1: Write the failing packaging tests** in `tests/runtime/test_packaging.py`:

```python
def test_assets_resolve_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from pitstop import config
    assert (config.PROMPTS / "supervisor_v0.txt").is_file()
    assert (config.SCENARIOS / "nvda_amd.json").is_file()
    assert (config.PROMPTS / "gepa" / "pitstop_nano_800.txt").is_file()
    assert config.SNAPSHOT.is_file()

def test_settings_paths_follow_pitstop_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PITSTOP_HOME", str(tmp_path))
    monkeypatch.delenv("PITSTOP_RUNS_DIR", raising=False)
    monkeypatch.delenv("PITSTOP_CACHE_DIR", raising=False)
    from pitstop.config import load_settings
    s = load_settings()
    assert s.runs_dir == tmp_path / "runs"
    assert s.cache_dir == tmp_path / "cache" / "youcom"
    monkeypatch.setenv("PITSTOP_RUNS_DIR", str(tmp_path / "elsewhere"))
    assert load_settings().runs_dir == tmp_path / "elsewhere"

def test_dotenv_loaded_from_cwd(tmp_path):
    # run in a subprocess so module-level load_dotenv runs fresh
    (tmp_path / ".env").write_text("PITSTOP_WORKER_MODEL=from-dotenv\n")
    out = subprocess.run([sys.executable, "-c", "from pitstop.config import load_settings; print(load_settings().worker_model)"],
                         cwd=tmp_path, capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "PITSTOP_WORKER_MODEL"})
    assert out.stdout.strip() == "from-dotenv"

def test_wheel_contains_package_data(tmp_path):
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path)], check=True, cwd=REPO_ROOT)
    names = zipfile.ZipFile(next(tmp_path.glob("pitstop-*.whl"))).namelist()
    for required in ("pitstop/assets/prompts/supervisor_v0.txt", "pitstop/assets/prompts/worker/planner.txt",
                     "pitstop/assets/scenarios/nvda_amd.json", "pitstop/demo/snapshot.json",
                     "pitstop/demo/static/demo.html", "pitstop/ui/static/index.html"):
        assert required in names

def test_console_script_help():
    out = subprocess.run(["pitstop", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "demo" in out.stdout
```

  (`REPO_ROOT = Path(__file__).resolve().parents[2]`; imports `os, sys, subprocess, zipfile, Path`.)
- [ ] **Step 2: Run the new tests and confirm they fail.** Run: `uv run pytest tests/runtime/test_packaging.py -v`. Expected: FAIL or ERROR (no module `pitstop` on the path / no `PROMPTS`).
- [ ] **Step 3: Do the moves** listed under Files with `git mv`. Remove the now-empty `pitstop/pitstop/` leftovers that move elsewhere in Task 4 (`evals/`, `prompts/recursive/`) **only after** Task 4. For now, leave those two in place.
- [ ] **Step 4: Rewrite `pyproject.toml`.**
  - `name = "pitstop"`, `version = "0.1.0.dev0"`, `license = "Apache-2.0"`, `readme = "README.md"`.
  - Dependencies: the union of the current lists (`pydantic>=2.7, openai>=1.60, httpx>=0.27, rich>=13, python-dotenv>=1.0, gepa==0.1.4, cloudpickle>=3.0`).
  - `[project.scripts] pitstop = "pitstop.cli:main"`.
  - `[tool.hatch.build.targets.wheel] packages = ["src/pitstop"]`. Hatch includes non-.py files inside the package by default; confirm with the wheel test.
  - Dev group: `pytest>=8, pytest-asyncio>=0.24, pytest-socket>=0.7, ruff>=0.6`.
  - `[tool.pytest.ini_options] asyncio_mode = "auto"`, `testpaths = ["tests"]`, `pythonpath = ["."]`.
  - Run `uv lock`.
- [ ] **Step 5: Implement the `pitstop.config` interface above.**
  - Load the env with `load_dotenv(find_dotenv(usecwd=True))`.
  - Replace every `ROOT / "prompts"` with `PROMPTS`, every `ROOT / "scenarios"` with `SCENARIOS`, and `ROOT / "demo" / "snapshot.json"` with `SNAPSHOT`.
  - In `demo/pair.py` and `demo/fixture.py`, delete `PROMPT_B_FALLBACK` / `FALLBACK_GEPA_PROMPT` and their branches. The primary GEPA prompt now always ships.
  - In `demo/views.py`, replace `REPO` with `_display_path(path: Path) -> str`, which returns the path relative to `Path.cwd()` when possible and the absolute path otherwise.
- [ ] **Step 6: Fix the runtime tests' imports.**
  - In `tests/runtime/conftest.py`, drop the `sys.path.insert`.
  - In `test_supervisor_validation.py`, change `from tests.test_schemas import make_input` to `from test_schemas import make_input`.
  - In `test_demo.py`, change `REAL_GEPA_RUN` to `Path(__file__).resolve().parents[2] / "runs" / "gepa" / "pitstop_nano_800"`.
  - Replace any `ROOT` imports with the new constants.
- [ ] **Step 7: Run all the runtime tests.** Run: `uv sync && uv run pytest tests/runtime -q`. Expected: all new tests PASS, and the moved tests match the baseline counts from Task 1.
- [ ] **Step 8: Smoke-test the demo.**
  1. Start `uv run pitstop demo --port 8770` in the background.
  2. `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8770/api/snapshot`. Expected: `200`.
  3. `curl -s http://127.0.0.1:8770/ | grep -c "<title>"`. Expected: `1`.
  4. Stop the server.
- [ ] **Step 9: Commit.** `git add -A src tests/runtime pyproject.toml uv.lock pitstop && git commit -m "refactor: make pitstop an installable src-layout package with packaged assets"`

---

### Task 3: GEPA engine moves into `pitstop.correct`

**Files:**
- Move and rename:
  - `gepa_financial_agent/crusoe_client.py` → `src/pitstop/correct/client.py`
  - `gepa_financial_agent/config.py` → `src/pitstop/correct/settings.py`
  - `gepa_financial_agent/policy_store.py` → `src/pitstop/correct/policy.py`
  - `gepa_financial_agent/optimizer.py` → `src/pitstop/correct/gepa.py`
- Create: `src/pitstop/correct/__init__.py`
- Move tests:
  - `tests/test_crusoe_client.py` → `tests/correct/test_client.py`
  - `tests/test_config.py` → `tests/correct/test_settings.py`
  - `tests/test_policy_store.py` → `tests/correct/test_policy.py`
  - `tests/test_optimizer.py` → `tests/correct/test_gepa.py`

**Interfaces:**
- Produces in `pitstop.correct`, with the same signatures as today:
  - `client`: `CrusoeClient`, `MockModelClient`, `ModelClient`, `CompletionResult`, `UsageTracker`, `estimate_cost_usd`, `ConfigError`, `ModelUnavailableError`
  - `settings`: `Settings`, `CRUSOE_BASE_URL`, `DEFAULT_SUPERVISOR_MODEL`, `DEFAULT_REFLECTION_MODEL`
  - `policy`: `Policy`, `PolicyStore`
  - `gepa`: `CrusoeReflectionLM`, `OptimizeConfig`, `OptimizationArtifacts`, `strip_code_fences`, and

    ```python
    run_optimization(seed_policy: Policy, train: list, val: list, supervisor_client: ModelClient,
                     reflection_lm: Callable[..., str], cfg: OptimizeConfig, *,
                     evaluator: Callable[[str, Any], tuple[float, dict]], objective: str, background: str) -> OptimizationArtifacts
    ```

    `evaluator`, `objective` and `background` are now **required** keyword arguments.
- Removed: `make_evaluator`, `_to_prompt_ids`, and the old-stack `OBJECTIVE` / `BACKGROUND` defaults. These are the only places that imported the superseded `evaluator`, `supervisor` and `schemas` modules.

- [ ] **Step 1: Rewrite `tests/correct/test_gepa.py` so it no longer needs the old stack.**
  - Delete `test_make_evaluator_returns_score_and_side_info`, `test_reflector_side_info_uses_aliases_and_omits_ids` and `test_background_states_the_actual_scoring_weights`. They test removed old-stack scoring; the live scorer is tested in Task 4.
  - Rewrite `test_run_optimization_end_to_end_with_mocks` to use a stub evaluator and plain-string examples:

```python
def test_run_optimization_requires_and_uses_caller_evaluator(tmp_path):
    seen = []
    def evaluator(candidate: str, example: str):
        seen.append(example); return (1.0 if "x" in example else 0.0), {"feedback": "ok"}
    art = run_optimization(Policy.from_text("seed prompt"), ["x1", "y1"], ["x2"], MockModelClient(),
                           CrusoeReflectionLM(MockModelClient()), OptimizeConfig(max_metric_calls=6, max_workers=1,
                           run_dir=str(tmp_path / "run")), evaluator=evaluator, objective="o", background="b")
    assert seen and (tmp_path / "run" / "artifacts.json").is_file()
    assert art.seed_policy.text == "seed prompt"

def test_run_optimization_has_no_default_evaluator():
    import inspect
    p = inspect.signature(run_optimization).parameters
    assert p["evaluator"].default is inspect.Parameter.empty
    assert p["objective"].default is inspect.Parameter.empty
```

  - Keep `test_reflection_lm_wraps_client`. Keep the mock-run and test-split tests only if they don't use `gepa_financial_agent.dataset`; otherwise delete them (they belong to the old stack).
- [ ] **Step 2: Run the tests and confirm they fail.** Run: `uv run pytest tests/correct -q`. Expected: FAIL (`ModuleNotFoundError: pitstop.correct`).
- [ ] **Step 3: Do the moves** with `git mv`. Make the imports relative within `pitstop.correct` (`from .client import ModelClient`, …). Apply the `run_optimization` signature change. Update the imports in the moved tests to `pitstop.correct.*`.
- [ ] **Step 4: Run the tests.** Run: `uv run pytest tests/correct tests/runtime -q`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -am "refactor: move GEPA engine into pitstop.correct; evaluator is caller-supplied"` (after `git add src/pitstop/correct tests/correct`).

---

### Task 4: `benchmarks/`: hard pairs, detector calibration, held-out set, trace generator

**Files:**
- Move `pitstop/pitstop/evals/` → `benchmarks/hard_pairs/`:
  - `hard_cases.py`, `run_hard_cases.py`
  - `data/{train,val,test}.jsonl`
  - `results/baseline_v0_{nano,super}.json`. These are small reference baselines and stay committed.
- Create `benchmarks/__init__.py`, `benchmarks/hard_pairs/__init__.py` and `benchmarks/calibration/__init__.py`.
- Move into `benchmarks/calibration/`:
  - `gepa_financial_agent/adapters/pitstop.py` → `adapter.py`
  - `gepa_financial_agent/recursive.py` → `recursive.py`
  - `scripts/optimize_pitstop.py` → `optimize.py`
  - `scripts/recursive_loop.py` → `loop.py`
  - `scripts/evaluate_system.py` → `evaluate_system.py`
  - `scripts/summarize_pitstop.py` → `summarize.py`
  - `pitstop/pitstop/prompts/recursive/state.json` → `state.json`
- Move into `benchmarks/held_out/`: `financial-agent/test-synthetic-data/{README.md,hard_tasks.py,build_hard_set.py}`.
- Move into `benchmarks/synth/`:
  - `synthetic-data/{synth/,scripts/,README.md,DATASET_FORMAT.md,HANDOFF.md}`. Drop `requirements.txt` and `pytest.ini`.
  - `synthetic-data/tests/{test_rollout.py,test_verify_export.py}` → `tests/benchmarks/synth/`
  - `synthetic-data/tests/helpers.py` → `tests/benchmarks/synth/synth_helpers.py`
- Move tests:
  - `tests/test_pitstop_adapter.py` → `tests/benchmarks/test_calibration_adapter.py`
  - `tests/test_recursive.py` → `tests/benchmarks/test_recursive.py`
  - `tests/test_summarize_pitstop.py` → `tests/benchmarks/test_summarize.py`
- Modify: `pyproject.toml` (`pythonpath = [".", "benchmarks/synth"]`).

**Interfaces:**
- Consumes `pitstop.config.PROMPTS` (Task 2) and `pitstop.correct.*` (Task 3).
- Produces in `benchmarks.calibration.adapter`:
  - `HARD_PAIRS_DATA = Path(<benchmarks/hard_pairs/data>)`
  - `SEED_PROMPT = pitstop.config.PROMPTS / "supervisor_v0.txt"`
  - `load_hard_cases(split: str, data_dir: Path = HARD_PAIRS_DATA) -> list[HardCase]`
  - `HardCase`, `score_judgment`, `make_pitstop_evaluator`, `OBJECTIVE`, `BACKGROUND`, with behavior unchanged
  - `PITSTOP_ROOT` and `_pitstop()` are removed. PitStop is imported normally (`from pitstop import llm, schemas`; `from pitstop.supervisor import policy, supervisor`).
- `benchmarks.calibration.recursive.STATE_PATH` becomes `Path(__file__).parent / "state.json"`.
- GEPA prompt outputs from `optimize.py` and `loop.py` are still written to `pitstop.config.PROMPTS / "gepa"` and `/ "recursive"`. This is a dev-only write into the source tree; the demo reads from the same place.
- Invocation:
  - `uv run python -m benchmarks.calibration.optimize …`
  - `uv run python -m benchmarks.hard_pairs.run_hard_cases …`
  - The trace generator's scripts stay direct-run: `uv run python benchmarks/synth/scripts/generate.py …`
- `benchmarks/held_out/build_hard_set.py` changes its `sys.path` line to `HERE.parent / "synth"`.

- [ ] **Step 1: Update the imports in the moved benchmark tests** (`benchmarks.calibration.adapter`, `pitstop.correct.*`). In `test_recursive.py` and `test_summarize.py`, load the scripts with `importlib.import_module("benchmarks.calibration.loop")` / `("benchmarks.calibration.summarize")` instead of loading them from file paths. Add:

```python
def test_adapter_needs_no_sys_path_hack():
    import sys, benchmarks.calibration.adapter as a
    assert not any(p.endswith("pitstop/pitstop") for p in sys.path)
    assert a.SEED_PROMPT.is_file() and len(a.load_hard_cases("test")) > 0
```

- [ ] **Step 2: Run the benchmark tests and confirm they fail.** Run: `uv run pytest tests/benchmarks -q`. Expected: FAIL (`ModuleNotFoundError: benchmarks.calibration`).
- [ ] **Step 3: Do the moves** with `git mv` and apply the interface changes above.
  - Remove the `sys.path.insert` lines from all four moved calibration scripts.
  - Change `from tests.helpers import` to `from synth_helpers import` in `test_verify_export.py`.
  - In `benchmarks/synth/synth/config.py`, `ROOT` (`parents[1]`) now resolves to `benchmarks/synth/`. Its data directories are generated on demand and git-ignored: add `benchmarks/synth/{traces,traces_archive,evidence,data}/` and `benchmarks/held_out/{traces,traces_archive,data,evidence}/` to `.gitignore`.
- [ ] **Step 4: Run the tests.** Run: `uv run pytest tests/benchmarks tests/correct tests/runtime -q`. Expected: PASS, with counts matching the Task 1 baseline for the moved tests.
- [ ] **Step 5: Check the CLIs.**
  - Run `uv run python -m benchmarks.calibration.optimize --mock --max-metric-calls 6 --name m1_smoke`. Expected: it exits 0 and prints the mock-run notice. Then delete the produced `runs/gepa/m1_smoke`.
  - Run `uv run python -m benchmarks.hard_pairs.run_hard_cases --help`. Expected: exit 0.
- [ ] **Step 6: Commit.** `git add -A benchmarks tests/benchmarks pyproject.toml .gitignore && git commit -m "refactor: collect eval sets and detector calibration tooling under benchmarks/"`

---

### Task 5: Delete the superseded stack and data; one offline test command

**Files:**
- Delete:
  - dead code: `agent/`, `main.py`, `data/`, `gepa/`, `policies/`
  - the rest of `gepa_financial_agent/`: `__init__.py`, `supervisor.py`, `session_manager.py`, `evaluator.py`, `dataset.py`, `schemas.py`, `adapters/{__init__,base,mock_agent}.py`
  - scripts: `scripts/{check_crusoe,demo_supervision,evaluate_supervisor,make_dev_fixtures,optimize_supervisor,smoke_test,summarize_repeats}.py`
  - old-stack tests: `tests/{test_adapters,test_dataset,test_evaluator,test_fixtures,test_schemas,test_sessions,test_supervisor}.py`, `tests/fixtures/`, root `tests/conftest.py` if empty
  - `requirements.txt`
  - leftovers: `pitstop/` (everything left: `README.md`, `.env.example`, `.gitignore`, `requirements.txt`, `pytest.ini`, `docs/PROGRESS.md`; fold anything useful from its README into the new README in Task 7 first), `synthetic-data/` (its remaining generated data), `financial-agent/`
  - `docs/results/`, already archived in Task 1
- Create: `tests/conftest.py` (root: socket guard), `tests/test_repo_hygiene.py`
- Modify: `pyproject.toml` (pytest `addopts`)

**Interfaces:**
- Consumes: everything from Tasks 2–4.
- Produces: a single command, `uv run pytest`, that runs runtime, correct and benchmark tests offline.

- [ ] **Step 1: Write the hygiene tests** in `tests/test_repo_hygiene.py`:

```python
REPO = Path(__file__).resolve().parents[1]
GONE = ["agent", "main.py", "gepa", "policies", "gepa_financial_agent", "synthetic-data", "financial-agent",
        "pitstop/pitstop", "requirements.txt", "data"]

@pytest.mark.parametrize("rel", GONE)
def test_superseded_paths_are_gone(rel):
    assert not (REPO / rel).exists()

def test_no_stale_references_in_code():
    hits = subprocess.run(["git", "grep", "-nE", r"gepa_financial_agent|pitstop/pitstop|synthetic-data/|policies/versions",
                           "--", "*.py", "*.toml", "*.yml", "*.md", ":!docs/history/**", ":!docs/superpowers/**", ":!CHANGELOG.md"],
                          cwd=REPO, capture_output=True, text=True).stdout
    assert hits == ""

def test_suite_runs_without_keys():
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/runtime", "tests/correct", "-p", "no:cacheprovider"],
                         cwd=REPO, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-2000:]
```

- [ ] **Step 2: Run the hygiene tests and confirm they fail.** Run: `uv run pytest tests/test_repo_hygiene.py -q`. Expected: FAIL on the `GONE` paths that still exist and on stale references.
- [ ] **Step 3: Delete the files** listed above with `git rm -r`. Set pytest `addopts = "--disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,localhost"` (pytest-socket).
- [ ] **Step 4: Run the full suite.** Run: `uv run pytest -q`. Expected: all PASS. If a test is blocked by the socket guard, it was making a real network call. Fix it to use the existing mocks; do not loosen the guard.
- [ ] **Step 5: Confirm the leftovers are gone.** Run: `git ls-files | wc -l`. Expected: well under the starting 796. Run: `du -sh .git/../* | sort -h | tail -5`. Expected: none of the remaining folders contain generated traces.
- [ ] **Step 6: Commit.** `git commit -m "chore: remove superseded supervisor stack, dead scaffold and committed generated data"`

---

### Task 6: Open-source repo basics

**Files:**
- Create: `LICENSE`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, `.github/ISSUE_TEMPLATE/{bug_report.yml,feature_request.yml,config.yml}`, `.github/workflows/ci.yml`
- Modify: `CONTRIBUTING.md`, `.env.example`, `.github/pull_request_template.md`, `pyproject.toml` (ruff config)
- Move: `docs/{ARCHITECTURE,PRD,INTEGRATION_CONTRACT,DATASET_CONTRACT,DATA_FEEDBACK,RESULTS,PITSTOP_RESULTS}.md`, `PROGRESS.md` and `docs/skills/` → `docs/history/`. Add `docs/history/README.md`: one paragraph saying these are hackathon-era documents, kept for context.

**Interfaces:**
- Produces a CI job named `test` that later milestones extend.

- [ ] **Step 1: `LICENSE`.** Use the full Apache License 2.0 text from https://www.apache.org/licenses/LICENSE-2.0.txt, with the appendix line `Copyright 2026 The PitStop Authors`.
- [ ] **Step 2: `CODE_OF_CONDUCT.md`.** Use Contributor Covenant 2.1, verbatim from https://www.contributor-covenant.org/version/2/1/code_of_conduct/. Set the enforcement contact to "open a private security advisory on GitHub or email the maintainers listed in SECURITY.md".
- [ ] **Step 3: `SECURITY.md`.** Cover:
  - **Supported versions:** main only, pre-1.0.
  - **How to report:** GitHub private vulnerability reporting. Not public issues.
  - **Scope note:** PitStop handles your API keys locally and never sends them anywhere except the endpoints you configure.
  - **What not to post:** keys or `.env` contents in issues.
- [ ] **Step 4: Issue and pull-request templates, and the changelog.**
  - Issue templates are GitHub issue forms. The bug form asks for version/commit, OS, Python version, steps, expected vs actual, and logs, with a "redact keys" warning. The feature form asks for the problem, the proposal and alternatives. `config.yml` sets `blank_issues_enabled: false`.
  - The PR template gets a checklist: tests added, `uv run pytest` passes, `uv run ruff check` passes, no secrets, docs updated.
  - `CHANGELOG.md` follows Keep a Changelog. Its `[Unreleased]` section summarizes Tasks 2–5.
- [ ] **Step 5: `CONTRIBUTING.md` and `.env.example`.**
  - CONTRIBUTING covers:
    - setup: `uv sync`, then copy `.env.example` to `.env`
    - repo map: one line each for `src/pitstop`, `benchmarks`, `tests`, `docs`
    - running tests: `uv run pytest`, which is offline by design
    - lint: `uv run ruff check`
    - that tests must not call paid APIs
    - branch and PR flow
    - the design-spec link
  - `.env.example` lists only the variables the code actually reads, found with `git grep -hoE 'getenv\("[A-Z_]+"' src benchmarks | sort -u`. Group them as **Inference** (`CRUSOE_*`), **Runtime** (`PITSTOP_*`), **Research** (`YDC_API_KEY`, `PITSTOP_YOUCOM_*`) and **Calibration** (`SUPERVISOR_*`, `REFLECTION_MODEL`). Every value is empty or a non-secret default.
- [ ] **Step 6: Ruff and CI.**
  - In `pyproject.toml`: `[tool.ruff] line-length = 120`, `target-version = "py311"`, `[tool.ruff.lint] select = ["F"]` (pyflakes only, to start low-noise).
  - Run `uv run ruff check --fix` and fix the rest by hand.
  - `ci.yml`:
    - triggers on push and pull_request
    - runs on ubuntu-latest with Python 3.11 and 3.12
    - steps: `astral-sh/setup-uv@v6`, `uv sync --locked`, `uv run ruff check`, `uv run pytest -q`
    - no secrets in the workflow env
- [ ] **Step 7: Verify.**
  - Run: `uv run ruff check && uv run pytest -q`. Expected: both pass.
  - Validate the workflow with `uvx --from check-jsonschema check-jsonschema --builtin-schema vendor.github-workflows .github/workflows/ci.yml`. Expected: `ok`.
- [ ] **Step 8: Commit.** `git add -A && git commit -m "docs: add license, code of conduct, security policy, templates, changelog and CI"`

---

### Task 7: README rewrite with architecture diagrams

**Files:**
- Modify: `README.md` (full rewrite; the old one moves to `docs/history/README-hackathon.md` first)
- Create: `docs/assets/hero.png`, a still of the landing hero (from `docs/superpowers/specs/2026-10-20-pitstop-mockups/landing-hero.html`, captured with reduced motion so it shows the drift-and-recovery frame)

**Interfaces:**
- Consumes: the commands verified in Tasks 2–6.

- [ ] **Step 1: Write the README** with these sections, in this order (spec §9.2):
  1. **Title and pitch.**
     - The wordmark "PitStop" and the pitch: "Keep your agents on the racing line: drift detection and GEPA-powered prompt correction for agent workflows. Open source, runs on your machine."
     - Badges: CI (`actions/workflows/ci.yml/badge.svg`), license Apache-2.0, Python ≥3.11. No PyPI badge yet.
  2. **Hero image:** `docs/assets/hero.png`.
  3. **Why PitStop:** 3–4 lines on off-task vs degraded drift, urgency, and existing tools stopping at alerts.
  4. **Status banner:** "Pre-release. Today the repo contains the PitStop research-swarm runtime, its drift supervisor, the GEPA calibration tooling and benchmarks. The drop-in SDK, local dashboard and agent-prompt correction are on the Roadmap."
  5. **Quickstart (run the example).** Every command must be verified in Step 3.
     - `git clone`, `uv sync`, `uv run pitstop demo`, then open http://127.0.0.1:8770 (works offline, from the bundled snapshot).
     - With keys: `cp .env.example .env`, fill in `CRUSOE_API_KEY` and `YDC_API_KEY`, then `uv run pitstop check` and `uv run pitstop run --correct-on-drift`.
     - Where runs are written: `~/.pitstop/runs`, or `PITSTOP_HOME` / `PITSTOP_RUNS_DIR`.
  6. **How it works today.** Two Mermaid diagrams:
     - A flowchart of one run: scenario → worker agents propose a query → supervisor (signals + LLM judgment + evidence validation) → AlertPolicy → steer (correction) or halt → findings → report synthesizer → run dir (`events.jsonl`, `decisions.jsonl`, `timeline.jsonl`) → `pitstop timeline` / demo UI.
     - The calibration loop: hard pairs → GEPA (`pitstop.correct.gepa`) → gate (accuracy up, false alarms not up) → promoted detector prompt.
  7. **Where it's going (architecture).** Labeled "Planned (milestones 2–4)". Two Mermaid diagrams drawn from spec §4 and §6.2/§7.3:
     - the runtime: user process with wrap/@tool/workflow → SQLite → server worker → dashboard;
     - detection layers L0–L5 → P0–P3 → steer/block/fallback/halt, plus the correction loop.
     - Link the design spec.
  8. **Repo map.** A tree of the post-M1 layout (`src/pitstop/{agents,supervisor,correct,demo,ui,research,assets}`, `benchmarks/{hard_pairs,calibration,held_out,synth}`, `tests/`, `docs/`), one line per entry.
  9. **Benchmarks.** The hard-pairs table carried over from the old README (Nano 0.817 → 0.878, false alarms 3 → 0; Super 0.861 → 0.956), with its limitations stated honestly:
     - 36 test cases, not statistically significant;
     - transfer to the held-out set was weak;
     - the prompt grew about 5×;
     - recursive round 0 was rejected by the gate.

     Also how to reproduce (`python -m benchmarks.hard_pairs.run_hard_cases`), and that the raw traces are a release asset.
  10. **Roadmap.** Milestones 1–4 from spec §10, with M1 marked done.
  11. **Contributing:** link to CONTRIBUTING.md.
  12. **License:** Apache-2.0.
  13. **Acknowledgements:** built at the hackathon on Crusoe Managed Inference by Qamil Mirza, Gautam Bhooma and Arjun Mittal; GEPA (`gepa` library); You.com Finance Research API.
- [ ] **Step 2: Validate the Mermaid diagrams.** Extract each Mermaid block to `/tmp/m1-mermaid-N.mmd` and run `npx -y @mermaid-js/mermaid-cli@11.4.2 -i /tmp/m1-mermaid-N.mmd -o /tmp/m1-mermaid-N.svg`. Expected: each exits 0.
- [ ] **Step 3: Verify every Quickstart command from a fresh clone.**
  - Run: `git clone . /tmp/pitstop-fresh && cd /tmp/pitstop-fresh && uv sync && uv run pitstop --help && (uv run pitstop demo --port 8771 & sleep 3; curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8771/api/snapshot; kill %1)`. Expected: help text, then `200`.
  - Do **not** run the paid `pitstop run` command. Mark it "requires keys" in the README.
- [ ] **Step 4: Check every relative link.** Run `git grep -oE '\]\(([^)#]+)' README.md CONTRIBUTING.md | sed 's/.*](//' | grep -v '^http' | xargs -I{} test -e {} || echo MISSING`. Expected: no `MISSING`.
- [ ] **Step 5: Commit.** `git add README.md docs/assets docs/history && git commit -m "docs: rewrite README with quickstart, architecture diagrams, repo map and roadmap"`

---

### Task 8: Final verification and owner handoff

- [ ] **Step 1: Run the full check.** Run: `uv sync --locked && uv run ruff check && uv run pytest -q`. Expected: all pass. Compare the test counts with `m1-baseline.txt`: the only drops should be the old-stack tests deleted in Tasks 3 and 5.
- [ ] **Step 2: Report to the owner.** Give them the exact commands to run themselves (outward-facing, so not run here):
  - `git push -u origin pitstop-redesign`
  - `gh pr create --title "Milestone 1: cleanup and restructure" --body-file <summary>`
  - `gh release create data-2026-10-20 dist-data/pitstop-hackathon-data-2026-10-20.tar.gz --title "Hackathon data archive" --notes "Generated traces, evidence and results removed from the tree in milestone 1."`

  Also flag:
  - the open teammate branches (`arjun/pitstop`, `gautam-bh/*`) will need a rebase onto the new layout;
  - check whether the `pitstop` name is available on PyPI before milestone 2;
  - any secret-scan hits from Task 1.
