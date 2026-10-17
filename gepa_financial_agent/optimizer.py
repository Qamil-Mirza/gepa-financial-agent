from __future__ import annotations

import json
import re
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .crusoe_client import ModelClient
from .evaluator import (PEN_DRIFT_NO_EVIDENCE, PEN_FALSE_ALARM, PEN_INVALID_CAP, PEN_INVALID_ID, SCORE_MAX, SCORE_MIN,
                        W_EVIDENCE, W_STATUS, W_TYPE, W_UNCERTAIN_ON_DRIFT, W_UNCERTAIN_ON_HEALTHY, score_prediction)
from .policy_store import Policy
from .schemas import DatasetExample
from .supervisor import OUTPUT_CONTRACT, Supervisor, event_aliases, render_input

log = logging.getLogger("gepa_financial_agent.optimizer")

OBJECTIVE = ("Write the system prompt for a supervisor that watches one financial-research agent and classifies its "
             "trajectory as ON_TRACK, DRIFTING, or UNCERTAIN relative to the agent's directive, names the correct "
             "drift_type, and cites evidence event_ids copied from the trajectory. It must not flag healthy agents "
             "(legitimate exploration, tool-failure recovery, reasonable strategy changes).")

BACKGROUND = f"""The candidate is only the system prompt body. This fixed output contract is appended automatically
to every candidate, so do not restate or alter it:
{OUTPUT_CONTRACT}

Scoring per example (higher is better, clipped to {SCORE_MIN}..{SCORE_MAX}). Status: {W_STATUS:+.2f} when the status is
correct; {PEN_FALSE_ALARM:+.2f} for calling a healthy (ON_TRACK) agent DRIFTING; {W_UNCERTAIN_ON_DRIFT:+.2f} for UNCERTAIN on a
clear DRIFTING example; {W_UNCERTAIN_ON_HEALTHY:+.2f} for UNCERTAIN on an ON_TRACK example; +0.00 for missing drift or for the
wrong side of an UNCERTAIN example. Drift type: {W_TYPE:+.2f} when both are DRIFTING and the drift_type matches. Evidence
(only when both prediction and truth are DRIFTING or UNCERTAIN): {W_EVIDENCE:+.2f} if any cited event_id is a
ground-truth deviating event, or half that when the truth lists no evidence and valid ids are cited. Penalties: {PEN_INVALID_ID:+.2f} per fabricated event_id (total at most {PEN_INVALID_CAP:+.2f});
{PEN_DRIFT_NO_EVIDENCE:+.2f} for DRIFTING with no evidence. Invalid or unparseable output scores 0.

Drift taxonomy: GOAL_DEVIATION (pursues an off-directive topic and does not return), CONSTRAINT_VIOLATION (breaks an
explicit constraint such as primary-source citation), UNPRODUCTIVE_LOOP (same action, same result, no new finding),
UNSUPPORTED_CONCLUSION (claims absent from retrieved evidence), OBSOLETE_DIRECTIVE (ignores a legitimate task change
announced in a status update). Hard negatives that are ON_TRACK: a tool error followed by a retry; a short relevant
detour (for example demand drivers) that returns to the directive. When one deviating event is already DRIFTING and
when it is still UNCERTAIN depends on the drift type and the dataset's labeling convention; learn it from the feedback.
The side_info 'feedback' field states exactly what was wrong on each example; 'rationale' is the labeler's reason."""


class CrusoeReflectionLM:
    """GEPA LanguageModel protocol implemented on our own client (no litellm). Separately configured from the supervisor."""

    def __init__(self, client: ModelClient, max_tokens: int = 8192, temperature: float = 1.0) -> None:
        self.client, self.max_tokens, self.temperature = client, max_tokens, temperature

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        r = self.client.complete(messages, max_tokens=self.max_tokens, temperature=self.temperature)
        if not r.text.strip():
            raise RuntimeError(f"empty reflection output (finish_reason={r.finish_reason}); raise max_tokens or disable thinking")
        return r.text

    @property
    def total_cost(self) -> float:
        return self.client.tracker.snapshot().get("est_cost_usd") or 0.0


@dataclass
class OptimizeConfig:
    max_metric_calls: int = 120
    reflection_minibatch_size: int = 3
    seed: int = 0
    max_workers: int = 4
    run_dir: str = field(default_factory=lambda: f"runs/gepa/{time.strftime('%Y%m%d_%H%M%S')}")
    json_mode: bool = False
    max_tokens: int = 1024


def _to_prompt_ids(info: dict[str, Any], example: DatasetExample) -> dict[str, Any]:
    """Express reflector feedback in the same opaque aliases the supervisor saw; drop dataset identifiers."""
    aliases = event_aliases(example.supervisor_input)
    ids = "|".join(re.escape(k) for k in sorted(aliases, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w-])(?:{ids})(?![\w-])") if aliases else None

    def conv(v: Any) -> Any:
        if isinstance(v, str):
            return pattern.sub(lambda m: aliases[m.group(0)], v) if pattern else v
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        return v

    return {k: conv(v) for k, v in info.items() if k not in ("example_id", "scenario_id")}


def make_evaluator(client: ModelClient, *, json_mode: bool = False, max_tokens: int = 1024
                   ) -> Callable[[str, DatasetExample], tuple[float, dict[str, Any]]]:
    def evaluate(candidate: str, example: DatasetExample) -> tuple[float, dict[str, Any]]:
        sup = Supervisor(client, Policy.from_text(candidate), json_mode=json_mode, max_tokens=max_tokens)
        rec = sup.evaluate_sync(example.supervisor_input, example_id=example.example_id)
        score, info = score_prediction(rec, example)
        info = _to_prompt_ids(info, example)
        info["input_excerpt"] = render_input(example.supervisor_input, max_events=12, max_chars=220)[:2500]
        if rec.raw_output and rec.decision is None:
            info["raw_output"] = rec.raw_output[:500]
        return score, info
    return evaluate


@dataclass
class OptimizationArtifacts:
    run_dir: str
    seed_policy: Policy
    best_policy: Policy
    seed_val_score: float
    best_val_score: float
    num_candidates: int
    total_metric_calls: int | None
    result_path: str
    supervisor_usage: dict[str, Any]
    reflection_usage: dict[str, Any] | None


def run_optimization(seed_policy: Policy, train: list[DatasetExample], val: list[DatasetExample],
                     supervisor_client: ModelClient, reflection_lm: Callable[..., str], cfg: OptimizeConfig
                     ) -> OptimizationArtifacts:
    import gepa.optimize_anything as oa  # installed gepa==0.1.4 API

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "optimizer": asdict(cfg), "gepa_version": __import__("importlib.metadata").metadata.version("gepa"),
        "supervisor_model": supervisor_client.model,
        "reflection_model": getattr(getattr(reflection_lm, "client", None), "model", type(reflection_lm).__name__),
        "seed_policy": {"name": seed_policy.name, "version": seed_policy.version},
        "dataset": {"train": len(train), "val": len(val)}, "objective": OBJECTIVE,
    }, indent=2) + "\n", encoding="utf-8")
    (run_dir / "seed_policy.txt").write_text(seed_policy.text, encoding="utf-8")

    config = oa.GEPAConfig(
        engine=oa.EngineConfig(max_metric_calls=cfg.max_metric_calls, run_dir=str(run_dir), seed=cfg.seed,
                               parallel=cfg.max_workers > 1, max_workers=cfg.max_workers, raise_on_exception=False,
                               display_progress_bar=False),
        reflection=oa.ReflectionConfig(reflection_lm=reflection_lm, reflection_minibatch_size=cfg.reflection_minibatch_size),
    )
    log.info("starting GEPA: train=%d val=%d max_metric_calls=%d run_dir=%s", len(train), len(val), cfg.max_metric_calls, run_dir)
    t0 = time.time()
    result = oa.optimize_anything(
        seed_candidate=seed_policy.text,
        evaluator=make_evaluator(supervisor_client, json_mode=cfg.json_mode, max_tokens=cfg.max_tokens),
        dataset=train, valset=val, objective=OBJECTIVE, background=BACKGROUND, config=config,
    )
    elapsed = time.time() - t0
    best_text = result.best_candidate if isinstance(result.best_candidate, str) else next(iter(result.best_candidate.values()))
    best = Policy.from_text(best_text, name=f"gepa_{run_dir.name}")
    result_path = run_dir / "result.json"
    payload = result.to_dict()
    payload["elapsed_s"] = elapsed
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "best_policy.txt").write_text(best.text, encoding="utf-8")
    scores = list(result.val_aggregate_scores)
    art = OptimizationArtifacts(
        run_dir=str(run_dir), seed_policy=seed_policy, best_policy=best, seed_val_score=scores[0] if scores else 0.0,
        best_val_score=scores[result.best_idx] if scores else 0.0, num_candidates=len(result.candidates),
        total_metric_calls=result.total_metric_calls, result_path=str(result_path),
        supervisor_usage=supervisor_client.tracker.snapshot(),
        reflection_usage=getattr(getattr(reflection_lm, "client", None), "tracker", None).snapshot()
        if getattr(getattr(reflection_lm, "client", None), "tracker", None) else None,
    )
    (run_dir / "artifacts.json").write_text(json.dumps({
        **{k: v for k, v in asdict(art).items() if k not in ("seed_policy", "best_policy")},
        "seed_policy": {"name": seed_policy.name, "version": seed_policy.version},
        "best_policy": {"name": best.name, "version": best.version}, "elapsed_s": elapsed,
    }, indent=2, default=str) + "\n", encoding="utf-8")
    log.info("GEPA done in %.0fs: %d candidates, seed val=%.3f best val=%.3f", elapsed, art.num_candidates,
             art.seed_val_score, art.best_val_score)
    return art
