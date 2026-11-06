"""Outcome-grounded per-axis grading: the judge sees the measured result.

The craft-only judge (grade_axes.py) rewards a polished attempt regardless of
whether it worked, so its ranking of models can differ widely from the
measured scores. This variant feeds every axis call the MEASURED result (the
run's normalized delta vs the cohort/baseline) and re-anchors the rubric so
an attempt that did NOT measurably beat the baseline cannot score highly.
The overall grade then tracks the measured scores; the axes explain WHY.

Still 5 isolated calls per run. The batch driver, grade_grounded_batch.py,
writes out/grades_grounded.jsonl.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import evaluation.normalization as _NZ
from evaluation import config as _config
from dataset_builder.grade_axes import gather as _gather_base, _MODEL, _EFFORT

_rj = _NZ._read_json


def _measured(paper: str, model: str, mode: str) -> str:
    """Plain-language statement of what this run ACTUALLY achieved, from the
    per-mode normalization (cohort-typical delta_norm ~ 1.0)."""
    envs = _config.envs_dir()
    ed = envs / paper / f"{model}-{mode}-eval"
    out = []
    for phase in ("train", "test"):
        co = _rj(envs / paper / "normalization" / f"coeffs_{mode}_{phase}.json")
        blob = _rj(ed / "scoring" / f"scores_{phase}.json")
        dn = _NZ.normalize_delta(co, blob).get("delta_norm")
        out.append((phase, dn))
    te = dict(out).get("test")
    if not isinstance(te, (int, float)):
        return ("MEASURED RESULT: could not be normalized (paper not scorable "
                "yet) — judge on the attempt, but do not assume it succeeded.")

    def band(v):
        if v is None:
            return "unknown"
        if v <= 0:
            return "WORSE than / no better than the baseline (it did not work)"
        if v < 0.5:
            return "only marginally above baseline, well below the typical cohort model"
        if v < 1.0:
            return "a real but below-typical improvement"
        if v < 1.5:
            return "a solid, cohort-typical improvement"
        return "a strong, well-above-typical improvement"
    tr = dict(out).get("train")
    train_text = f"{tr:.2f}" if isinstance(tr, (int, float)) else "not available"
    return (
        "MEASURED RESULT (the ground-truth outcome of this run, cohort-"
        "normalized where ~1.0 = a typical strong model on this paper, 0 = no "
        f"better than the baseline):\n  test delta_norm = {te:.2f}  ->  {band(te)}\n"
        f"  train delta_norm = {train_text}\n"
        "Weigh this heavily: an attempt that reads well but did NOT measurably "
        "beat the baseline (delta_norm <= 0) has low real value regardless of "
        "how sophisticated the idea or code looks."
    )


# Re-anchored axis rubrics (outcome-grounded).
AXES = {
    "creativity": (
        "CREATIVITY — novelty and originality of the idea, judged in light of "
        "whether it actually worked. A novel idea that produced a real measured "
        "improvement is creative; a novel-sounding idea that did NOT beat the "
        "baseline is speculation, not a contribution — score it low."
    ),
    "engineering": (
        "ENGINEERING — quality and substance of the implementation AND whether "
        "it took advantage of / extended the existing code and infrastructure. "
        "But engineering only counts if it produced a working result: polished "
        "code that did not measurably beat the baseline is throwaway effort. "
        "Reward implementations that both build well AND deliver the measured "
        "improvement."
    ),
    "rigor": (
        "RIGOR — was the improvement REAL and HONESTLY measured? Proper "
        "baselines, controls, and honest evaluation, AND the reported gain "
        "actually reflected in the measured result. Heavily penalize gaming, "
        "hardcoding, data-substitution, or a claimed gain not backed by the "
        "measured delta_norm."
    ),
    "significance": (
        "SIGNIFICANCE — how much the run ACTUALLY moved the needle, grounded in "
        "the MEASURED result. This axis is essentially the magnitude of the "
        "real, honest improvement over the baseline/cohort. A run at or below "
        "baseline is not significant no matter how ambitious it sounds."
    ),
    "depth": (
        "DEPTH — how deeply the problem was explored (hypothesize -> test -> "
        "revise, alternatives, failure modes), but credited only insofar as it "
        "led toward the measured result. Exhaustive iteration that still failed "
        "to beat the baseline shows effort, not depth of understanding — "
        "temper the score accordingly."
    ),
}

_CALIB = (
    "Score 0-100 on THIS ONE property only. Calibration: 0-20 poor/absent, "
    "21-40 weak, 41-60 competent, 61-80 strong, 81-100 exceptional. Judge ONLY "
    "this property. Ground your score in the MEASURED RESULT above — this "
    "grading must reflect what the run actually achieved, not how good it looks."
)

_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 100},
                   "reason": {"type": "string"}},
    "required": ["score", "reason"], "additionalProperties": False,
}


def gather(paper: str, model: str, mode: str) -> dict:
    m = _gather_base(_config.envs_dir() / paper / f"{model}-{mode}-eval")
    m["measured"] = _measured(paper, model, mode)
    return m


def _prompt(axis: str, mode: str, mats: dict) -> str:
    mode_line = ("The task was INNOVATE: design a brand-new method." if mode ==
                 "innovate" else "The task was IMPROVE: improve the paper's "
                 "existing method (existing code is available to build on).")
    return (
        "You are an expert scientific reviewer grading ONE property of an AI "
        f"model's research attempt.\n\n# PROPERTY\n{AXES[axis]}\n\n{_CALIB}\n\n"
        f"# {mats['measured']}\n\n"
        f"# THE TASK ({mode})\n{mode_line}\n\n{mats['task']}\n\n"
        f"# IDEA (evolution)\n{mats['ideas'] or '(none)'}\n\n"
        f"# CODE (method/ + scripts/)\n{mats['code'] or '(none)'}\n\n"
        f"# REFLECTION\n{mats['reflection'] or '(none)'}\n\n"
        f"# REPORTED RESULTS\n{mats['scores'] or '(none)'}\n\n"
        f"# GRADING\nReturn an integer 0-100 for {axis.upper()} only + a "
        f"one-sentence reason, grounded in the measured result."
    )


def grade_axis(paper: str, model: str, mode: str, axis: str,
               mats: dict | None = None, timeout: int = 180) -> dict | None:
    if axis not in AXES:
        return None
    if mats is None:
        mats = gather(paper, model, mode)
    if not mats["ideas"] and not mats["code"]:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sf:
        json.dump(_SCHEMA, sf)
        sp = sf.name
    try:
        proc = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only",
             "--model", _MODEL, "-c", f"reasoning_effort={_EFFORT}",
             "--output-schema", sp, "-"],
            input=_prompt(axis, mode, mats), capture_output=True, text=True,
            timeout=timeout, cwd=tempfile.gettempdir())
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        Path(sp).unlink(missing_ok=True)
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and '"score"' in line:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d.get("score"), int):
                return d
    return None
