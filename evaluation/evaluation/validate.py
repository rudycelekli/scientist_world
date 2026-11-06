"""Host-side completeness check of the agent's scoring output.

Never runs inside the agent's container and is never visible in the
agent's workspace. The harness runs it after the agent signals ``done``
to decide whether the run is complete.

What it checks, and nothing else:

1. ``<env>/scoring/scores_train.json`` exists and is valid JSON.
2. For every method the paper's own ``source/scoring/scores_train.json``
   reports, except those tagged ``type: "baseline"`` in
   ``reference.json``, the agent's file contains:
     - that method name
     - every numeric metric the source reports for it
3. ``validate_test`` applies the same check to ``scores_test.json``
   (written by the grader), plus a degenerate-output check.

It checks presence, not values: whether the numbers are honest is the
grader's job (:mod:`evaluation.verifier`).

Methods marked ``type: "baseline"`` in ``reference.json`` are not the
agent's responsibility (paper baselines it is not expected to
reproduce) and are skipped. Methods with no entry in ``reference.json``
default to required: an explicit ``baseline`` tag is the only thing
that exempts a method.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MissingCell:
    """A cell present in source's scoring file but absent in agent's."""
    experiment: str
    method: str
    metric: str | None  # None when the method itself is entirely absent


@dataclass
class ValidationResult:
    ok: bool
    phase: str                              # "train" | "test"
    missing: list[MissingCell] = field(default_factory=list)
    file_missing: bool = False              # agent's scoring file is absent / unreadable
    file_invalid: bool = False              # exists but not valid JSON
    degenerate: bool = False                # present + parseable but every proposed cell scored 0 (all-abstain)

    @property
    def n_missing(self) -> int:
        return len(self.missing)

    def summary(self) -> str:
        if self.file_missing:
            return f"agent's scoring/scores_{self.phase}.json is missing"
        if self.file_invalid:
            return f"agent's scoring/scores_{self.phase}.json is not valid JSON"
        if self.degenerate:
            return f"agent's scoring/scores_{self.phase}.json scored every proposed cell as 0 (all-abstain)"
        if self.ok:
            return f"{self.phase}: ok"
        return f"{self.phase}: {self.n_missing} cell(s) missing"

    def format_for_prompt(self) -> str:
        """Render a human-readable bullet list of gaps for the agent's prompt.

        Groups by (experiment, method) so the agent can act on each block
        rather than seeing 60 individual lines. Method-level gaps (entire
        method missing) are rendered separately from metric-level gaps
        (method present but missing metrics).
        """
        if self.file_missing:
            return f"- `scoring/scores_{self.phase}.json` does not exist."
        if self.file_invalid:
            return f"- `scoring/scores_{self.phase}.json` exists but is not valid JSON."
        # Group: (experiment, method) → list[metric|None]
        from collections import defaultdict
        by_key: dict[tuple[str, str], list[str | None]] = defaultdict(list)
        for c in self.missing:
            by_key[(c.experiment, c.method)].append(c.metric)
        lines = []
        for (exp, method), metrics in sorted(by_key.items()):
            if metrics == [None] or any(m is None for m in metrics):
                lines.append(f"- `{exp}` / `{method}`: method entirely missing — produce a `results` entry for it with all metrics from the train evaluator.")
            else:
                names = ", ".join(f"`{m}`" for m in sorted(metrics) if m is not None)
                lines.append(f"- `{exp}` / `{method}`: missing metrics {names}.")
        return "\n".join(lines)


def _load_json(path: Path) -> tuple[dict | None, bool]:
    """Return (data, file_invalid). data is None if file missing or unreadable."""
    if not path.exists():
        return None, False
    try:
        return json.loads(path.read_text()), False
    except (OSError, json.JSONDecodeError):
        return None, True


def _flatten_experiments(score: dict) -> dict:
    """Both layouts exist: wrapped (``{experiments: {...}}``) and flat
    (top-level keys are experiments). Return the experiments dict."""
    if not isinstance(score, dict):
        return {}
    exps = score.get("experiments")
    if isinstance(exps, dict):
        return exps
    return {k: v for k, v in score.items() if isinstance(v, dict)}


def _flatten_results(exp: dict) -> dict:
    """Per-experiment layout: ``{results: {method: ...}}`` or flat
    ``{method: ...}``. Return the methods dict."""
    if not isinstance(exp, dict):
        return {}
    res = exp.get("results")
    if isinstance(res, dict):
        return res
    return {k: v for k, v in exp.items() if isinstance(v, dict)}


def _is_baseline(reference: dict | None, exp_name: str, method: str) -> bool:
    """True only when reference.json explicitly tags this method as
    ``type: "baseline"``. Methods with no entry in reference.json or no
    ``type`` field default to NOT baseline (i.e. required)."""
    if not isinstance(reference, dict):
        return False
    exps = reference.get("experiments") or {}
    exp = exps.get(exp_name) if isinstance(exps, dict) else None
    if not isinstance(exp, dict):
        return False
    methods = exp.get("results") or {}
    entry = methods.get(method) if isinstance(methods, dict) else None
    if not isinstance(entry, dict):
        return False
    return entry.get("type") == "baseline"


def _higher_is_better(reference: dict | None, exp_name: str, metric: str) -> bool:
    """Metric direction from reference.json; defaults to True when unknown.

    Used to scope the degenerate (all-abstain) check to higher-is-better
    metrics — 0.0 there means "no signal", whereas 0.0 on a lower-is-better
    metric (error/loss = perfect) must never be flagged.
    """
    if not isinstance(reference, dict):
        return True
    exps = reference.get("experiments") or {}
    exp = exps.get(exp_name) if isinstance(exps, dict) else None
    metrics = exp.get("metrics") if isinstance(exp, dict) else None
    meta = metrics.get(metric) if isinstance(metrics, dict) else None
    if isinstance(meta, dict) and isinstance(meta.get("higher_is_better"), bool):
        return meta["higher_is_better"]
    return True


def _validate_one(
    agent_path: Path, source_path: Path, reference: dict | None, phase: str,
) -> ValidationResult:
    agent_score, agent_invalid = _load_json(agent_path)
    if agent_invalid:
        return ValidationResult(ok=False, phase=phase, file_invalid=True)
    if agent_score is None:
        return ValidationResult(ok=False, phase=phase, file_missing=True)

    source_score, _ = _load_json(source_path)
    if not isinstance(source_score, dict):
        # No source to compare against — can't validate. Treat as ok
        # (we don't know what the agent owes).
        return ValidationResult(ok=True, phase=phase)

    missing: list[MissingCell] = []
    present: list[tuple[float, float, bool]] = []   # (agent_val, src_val, higher_is_better)
    src_exps = _flatten_experiments(source_score)
    agent_exps = _flatten_experiments(agent_score)

    for exp_name, src_exp in src_exps.items():
        src_methods = _flatten_results(src_exp)
        agent_exp = agent_exps.get(exp_name)
        agent_methods = _flatten_results(agent_exp) if isinstance(agent_exp, dict) else {}

        for method, src_metrics in src_methods.items():
            if not isinstance(src_metrics, dict):
                continue
            if _is_baseline(reference, exp_name, method):
                continue   # baseline — not the agent's responsibility
            agent_metrics = agent_methods.get(method)
            if not isinstance(agent_metrics, dict):
                # Whole method missing.
                missing.append(MissingCell(experiment=exp_name, method=method, metric=None))
                continue
            for metric, src_val in src_metrics.items():
                if not isinstance(src_val, (int, float)):
                    continue
                agent_val = agent_metrics.get(metric)
                if not isinstance(agent_val, (int, float)):
                    missing.append(MissingCell(experiment=exp_name, method=method, metric=metric))
                else:
                    present.append((float(agent_val), float(src_val),
                                    _higher_is_better(reference, exp_name, metric)))

    # Degenerate (test phase only): on every higher-is-better cell where the
    # source had real signal (src != 0), the agent scored exactly 0 — the
    # all-abstain signature (e.g. predictions keyed so the held-out scorer
    # matched nothing). Scoped to higher-is-better so a genuine 0.0 on a
    # lower-is-better metric (error/loss = perfect) is never mis-flagged, and
    # to source-nonzero cells so diagnostics like abstention_rate don't mask
    # it. Left unflagged, this normalizes into a real-looking negative floor.
    scored = [a for a, s, hib in present if hib and s != 0.0]
    degenerate = phase == "test" and bool(scored) and all(a == 0.0 for a in scored)

    return ValidationResult(
        ok=not missing and not degenerate,
        phase=phase, missing=missing, degenerate=degenerate,
    )


def validate_train(env_dir: Path) -> ValidationResult:
    """Validate the agent's main-phase output (``scoring/scores_train.json``)."""
    source_dir = env_dir.parent / "source" / "scoring"
    reference, _ = _load_json(source_dir / "reference.json")
    return _validate_one(
        agent_path=env_dir / "scoring" / "scores_train.json",
        source_path=source_dir / "scores_train.json",
        reference=reference,
        phase="train",
    )


def validate_test(env_dir: Path) -> ValidationResult:
    """Validate the held-out test phase output (``scoring/scores_test.json``)."""
    source_dir = env_dir.parent / "source" / "scoring"
    reference, _ = _load_json(source_dir / "reference.json")
    return _validate_one(
        agent_path=env_dir / "scoring" / "scores_test.json",
        source_path=source_dir / "scores_test.json",
        reference=reference,
        phase="test",
    )


__all__ = ["ValidationResult", "MissingCell", "validate_train", "validate_test"]
