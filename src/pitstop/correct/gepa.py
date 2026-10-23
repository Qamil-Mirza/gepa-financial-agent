from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .client import ModelClient
from .policy import Policy
from ..sdk.errors import Halted


log = logging.getLogger("pitstop.correct.gepa")

class CrusoeReflectionLM:
    """GEPA LanguageModel protocol implemented on our own client (no litellm). Separately configured from the
    supervisor."""

    def __init__(self, client: ModelClient, max_tokens: int = 8192, temperature: float = 1.0) -> None:
        self.client, self.max_tokens, self.temperature = client, max_tokens, temperature

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        r = self.client.complete(messages, max_tokens=self.max_tokens, temperature=self.temperature)
        if not r.text.strip():
            raise RuntimeError(
                f"empty reflection output (finish_reason={r.finish_reason}); raise max_tokens or disable thinking"
            )
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
    acceptance_criterion: str = "strict_improvement"  # or "improvement_or_equal" (gepa 0.1.4)
    perfect_score: float | None = None  # when set, skip reflection on minibatches the parent already aces


def strip_code_fences(text: str) -> str:
    """Drop a code fence the reflection model wrapped around its proposal (docs/RESULTS.md, Known issues)."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


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


class _HaltGuard:
    """Wrap a GEPA callable so a Halted (it swallows exceptions) is stored once, re-raised, and stops later calls."""

    def __init__(self, fn: Callable[..., Any], halted: list[Halted], lock: threading.Lock) -> None:
        self._fn, self._halted, self._lock = fn, halted, lock

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._halted:
            raise self._halted[0]
        try:
            return self._fn(*args, **kwargs)
        except Halted as e:
            with self._lock:
                if not self._halted:
                    self._halted.append(e)
            raise self._halted[0]

    def __getattr__(self, name: str) -> Any:  # forward e.g. total_cost / client of the wrapped callable
        return getattr(self._fn, name)


def run_optimization(seed_policy: Policy, train: list[Any], val: list[Any],
                     supervisor_client: ModelClient, reflection_lm: Callable[..., str], cfg: OptimizeConfig, *,
                     evaluator: Callable[[str, Any], tuple[float, dict[str, Any]]],
                     objective: str, background: str) -> OptimizationArtifacts:
    """Run GEPA on the policy text. The caller supplies the evaluator, objective and background."""
    import gepa.optimize_anything as oa  # installed gepa==0.1.4 API

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "optimizer": asdict(cfg), "gepa_version": __import__("importlib.metadata").metadata.version("gepa"),
        "supervisor_model": supervisor_client.model,
        "reflection_model": getattr(getattr(reflection_lm, "client", None), "model", type(reflection_lm).__name__),
        "seed_policy": {"name": seed_policy.name, "version": seed_policy.version},
        "dataset": {"train": len(train), "val": len(val)}, "objective": objective,
    }, indent=2) + "\n", encoding="utf-8")
    (run_dir / "seed_policy.txt").write_text(seed_policy.text, encoding="utf-8")

    halted: list[Halted] = []
    lock = threading.Lock()
    evaluator = _HaltGuard(evaluator, halted, lock)
    reflection_lm = _HaltGuard(reflection_lm, halted, lock)
    config = oa.GEPAConfig(
        engine=oa.EngineConfig(
            max_metric_calls=cfg.max_metric_calls,
            run_dir=str(run_dir),
            seed=cfg.seed,
            parallel=cfg.max_workers > 1,
            max_workers=cfg.max_workers,
            raise_on_exception=False,
            display_progress_bar=False,
            acceptance_criterion=cfg.acceptance_criterion,
        ),
        reflection=oa.ReflectionConfig(
            reflection_lm=reflection_lm,
            reflection_minibatch_size=cfg.reflection_minibatch_size,
            skip_perfect_score=cfg.perfect_score is not None,
            perfect_score=cfg.perfect_score,
        ),
    )
    log.info(
        "starting GEPA: train=%d val=%d max_metric_calls=%d run_dir=%s",
        len(train),
        len(val),
        cfg.max_metric_calls,
        run_dir,
    )
    t0 = time.time()
    try:
        result = oa.optimize_anything(
            seed_candidate=seed_policy.text,
            evaluator=evaluator,
            dataset=train, valset=val, objective=objective, background=background, config=config,
        )
    except BaseException:
        if not halted:
            raise
        result = None
    if halted:
        h = halted[0]
        (run_dir / "aborted.json").write_text(json.dumps(
            {"reason": h.reason, "drift_id": h.drift_id, "type": type(h).__name__}, indent=2) + "\n", encoding="utf-8")
        log.warning("GEPA aborted: %s", h.reason)
        raise h
    elapsed = time.time() - t0
    best_text = (
        result.best_candidate if isinstance(result.best_candidate, str) else next(iter(result.best_candidate.values()))
    )
    if result.best_idx != 0:  # the seed is kept byte-for-byte so an unchanged prompt keeps its version hash
        best_text = strip_code_fences(best_text)
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
