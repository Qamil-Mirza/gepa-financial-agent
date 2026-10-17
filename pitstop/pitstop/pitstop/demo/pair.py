"""Paired live run: the same scenario twice, concurrently, one swarm per supervisor prompt.

  left  (pane a)  Nano + prompts/supervisor_v0.txt (the original prompt)
  right (pane b)  Nano + the GEPA prompt

Each supervisor is in charge of its own swarm (it corrects drift, never halts), and each swarm's report agent
writes that side's team report. Both share one You.com client: workers run at temperature 0 from identical
starting states, so many queries are identical and are paid for once (cache, or in-flight de-duplication).

  runs_dir/pair-<6 hex>/pair.json        {"pair_id", "objective", "left", "right", "started_at"[, "finished_at"]}
  runs_dir/pair-<6 hex>/<left run_id>/   an ordinary swarm run dir, likewise the right one

The swarm stack (rich, httpx, openai) is imported only when a pair is built, so the demo server starts without it.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..config import ROOT, Settings
from ..schemas import utc_now
from .views import PAIR_FILE

if TYPE_CHECKING:
    from ..orchestrator import Swarm

SCENARIO = ROOT / "scenarios" / "nvda_amd.json"
NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
PROMPT_A = ROOT / "prompts" / "supervisor_v0.txt"
PROMPT_B = ROOT / "prompts" / "gepa" / "pitstop_nano_800.txt"
PROMPT_B_FALLBACK = ROOT.parent.parent / "policies" / "versions" / "gepa_nano_150.txt"
LIVE_MAX_RETRIES = 6
LIVE_MAX_CORRECTIONS = 2
MAX_OBJECTIVE_CHARS = 600


class LiveError(Exception):
    status = 400


@dataclass
class Pair:
    pair_id: str
    pair_dir: Path
    left: "Swarm"
    right: "Swarm"

    @classmethod
    def create(cls, runs_dir: Path, make_swarm: Callable[[str, Path], "Swarm"]) -> "Pair":
        """make_swarm(side, pair_dir) builds the "left" / "right" swarm with runs_dir=pair_dir.
        pair.json is written before anything runs, so the pair can be viewed at once."""
        pair_id = f"pair-{uuid.uuid4().hex[:6]}"
        pair_dir = Path(runs_dir) / pair_id
        pair_dir.mkdir(parents=True)
        pair = cls(pair_id, pair_dir, make_swarm("left", pair_dir), make_swarm("right", pair_dir))
        pair._write(pair_id=pair_id, objective=pair.left.scenario.swarm_objective, left=pair.left.run.run_id,
                    right=pair.right.run.run_id, started_at=utc_now())
        return pair

    def _write(self, **fields: Any) -> None:
        path = self.pair_dir / PAIR_FILE
        doc = json.loads(path.read_text()) if path.exists() else {}
        path.write_text(json.dumps({**doc, **fields}, indent=2))

    async def run_all(self) -> None:
        """Both swarms to the end. One side failing does not stop the other; the first error is re-raised
        after both have stopped (pair.json then also gets "errors")."""
        results = await asyncio.gather(self.left.run_all(), self.right.run_all(), return_exceptions=True)
        errors = {side: f"{type(r).__name__}: {r}" for side, r in zip(("left", "right"), results)
                  if isinstance(r, BaseException)}
        self._write(finished_at=utc_now(), **({"errors": errors} if errors else {}))
        for r in results:
            if isinstance(r, BaseException):
                raise r


def build_pair(objective: str, runs_dir: Path, settings: Settings) -> Pair:
    """Two real swarms, built the way `cli run` builds one. Raises LiveError (400) when keys are missing."""
    missing = [k for k, v in (("CRUSOE_API_KEY", settings.crusoe_api_key), ("YDC_API_KEY", settings.ydc_api_key))
               if not v]
    if missing:
        raise LiveError(f"{' and '.join(missing)} not set; live mode needs both (see .env.example)")
    from rich.console import Console

    from ..agents.synthesizer import ReportSynthesizer
    from ..llm import LLMClient, UsageTracker
    from ..orchestrator import Swarm
    from ..research.youcom import FinanceResearchClient
    from ..schemas import Scenario
    from ..supervisor.supervisor import Supervisor

    scenario = Scenario.model_validate_json(SCENARIO.read_text())
    if objective.strip():
        scenario = scenario.model_copy(update={"swarm_objective": objective.strip()[:MAX_OBJECTIVE_CHARS]})
    # More retries than the CLI default: on stage a 429 should cost a few seconds, not a failed check.
    llm = lambda model, t, **kw: LLMClient(model, api_key=settings.crusoe_api_key, base_url=settings.crusoe_base_url,
                                           temperature=t, usage=UsageTracker(), max_retries=LIVE_MAX_RETRIES, **kw)
    research = FinanceResearchClient(  # one call per step per swarm at most: never let the cap cut a side short
        api_key=settings.ydc_api_key, cache_dir=settings.cache_dir, mode="live", effort=settings.youcom_effort,
        max_live_calls=max(settings.max_research_calls, 2 * sum(a.max_steps for a in scenario.agents)))
    prompts = {"left": PROMPT_A, "right": PROMPT_B if PROMPT_B.exists() else PROMPT_B_FALLBACK}

    def side(name: str, pair_dir: Path) -> Swarm:
        # Worker at temperature 0, so both sides start identically and a rehearsal warms the You.com cache.
        return Swarm(scenario, worker_llm=llm(settings.worker_model, 0.0),
                     supervisor=Supervisor(llm(NANO, 0.0), prompts[name]), research=research,
                     prompts_dir=ROOT / "prompts" / "worker", runs_dir=pair_dir,
                     halt_on_drift=False, correct_on_drift=True, max_corrections=LIVE_MAX_CORRECTIONS,
                     synthesizer=ReportSynthesizer(llm(settings.report_model, 0.2, max_tokens=4000),
                                                   ROOT / "prompts" / "worker"),
                     console=Console(quiet=True), verbose_events=False)

    return Pair.create(runs_dir, side)
