"""Per-axis LLM-judge grading: one isolated judge call per property.

Grading all five properties in a single call makes the judge cohere them into
one latent "quality" factor (halo effect: the 5 scores end up correlated
0.6-0.8 and every model shows the same profile shape). This module grades each
property in its OWN call, seeing only that property's rubric, so the axes can
diverge and a model's real specialty shows.

One call -> {score 0-100, reason}. Resumable batch in grade_axes_batch.py.

The judge is called through the ``codex`` CLI on the host (``codex exec``
with a read-only sandbox). Model and reasoning effort come from
``$GRADE_MODEL`` and ``$GRADE_EFFORT``.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

# Defaults are the judge settings used for the paper.
_MODEL = os.environ.get("GRADE_MODEL", "gpt-5.5")
_EFFORT = os.environ.get("GRADE_EFFORT", "medium")

_MAX_FILE_CHARS = 12000
_MAX_CODE_CHARS = 40000   # method/ + scripts/ combined (engineering needs full code)
_MAX_IDEA_CHARS = 24000

_CALIB = (
    "Score 0-100 on THIS ONE property only. Calibration: 0-20 poor/absent, "
    "21-40 weak, 41-60 competent, 61-80 strong, 81-100 exceptional. Use the "
    "full range; do not hedge toward the middle. Judge ONLY this property — "
    "ignore how good the attempt is on other dimensions. A high reported "
    "number obtained by gaming, hardcoding, data-substitution, or fabrication "
    "does NOT earn credit here."
)

# Per-axis rubric. Each is graded in isolation (no other axis in view).
AXES = {
    "creativity": (
        "CREATIVITY — novelty and originality of the idea. Is it a genuinely "
        "new, non-obvious, or clever approach to the problem, or a routine / "
        "textbook / obvious move? Reward original ideas that are grounded in "
        "the problem; penalize trivial tweaks and derivative reuse."
    ),
    "engineering": (
        "ENGINEERING — the quality and substance of the implementation work. "
        "Judge: (a) is the code correct, clean, and a faithful realization of "
        "the stated method; (b) does the model TAKE ADVANTAGE OF THE EXISTING "
        "CODE AND INFRASTRUCTURE — reusing, extending, and cleanly integrating "
        "with the paper's / task's existing codebase rather than reinventing "
        "from scratch or bypassing it; (c) the AMOUNT and depth of real "
        "engineering effort (nontrivial systems work, careful data/eval "
        "plumbing, handling edge cases). In IMPROVE mode especially, reward "
        "building meaningfully on the existing method's code. Penalize broken, "
        "hacky, shortcut, or throwaway implementations that don't actually "
        "implement the idea."
    ),
    "rigor": (
        "RIGOR — scientific rigor of the evaluation. Proper experimental "
        "design, controls, ablations, appropriate baselines, and HONEST "
        "measurement. Heavily penalize gaming the metric, hardcoding results, "
        "substituting data, or rewriting the evaluator to inflate scores."
    ),
    "significance": (
        "SIGNIFICANCE — if the approach works, how much does it matter? "
        "Magnitude and relevance of the contribution to the actual problem. "
        "Reward meaningful, consequential improvements; penalize trivial, "
        "marginal, or irrelevant ones."
    ),
    "depth": (
        "DEPTH — how deeply the problem was explored. Evidence of iteration "
        "(hypothesize -> test -> revise across idea versions), alternatives "
        "considered, failure modes diagnosed and addressed, vs a shallow "
        "one-shot attempt."
    ),
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


def _read(p: Path, cap: int = _MAX_FILE_CHARS) -> str:
    try:
        t = p.read_text(errors="replace")
    except OSError:
        return ""
    return t[:cap] + ("\n...[truncated]" if len(t) > cap else "")


def gather(run_dir: Path) -> dict:
    R = run_dir / "READMEs"
    # method.md included so the judge knows the EXISTING method to build on.
    task = "\n\n".join(
        f"## {name}\n{_read(R / f)}"
        for name, f in (("PROBLEM", "problem.md"), ("DIRECTION", "DIRECTION.md"),
                        ("CONSTRAINTS", "CONSTRAINTS.md"), ("TARGETS", "TARGETS.md"),
                        ("EXISTING METHOD", "method.md"))
        if (R / f).exists()
    )
    idea_dir = run_dir / "idea"
    ideas = ""
    if idea_dir.is_dir():
        vs = sorted(idea_dir.glob("idea_v*.md"),
                    key=lambda f: int("".join(c for c in f.stem if c.isdigit()) or 0))
        ideas = "\n\n".join(f"===== {f.name} =====\n{_read(f)}" for f in vs)[:_MAX_IDEA_CHARS]
    # method/ + scripts/ code — the model's full engineering output
    code_parts = []
    for sub in ("method", "scripts"):
        d = run_dir / sub
        if d.is_dir():
            for f in sorted(d.glob("*.py")):
                if f.name == "__init__.py" and f.stat().st_size == 0:
                    continue
                code_parts.append(f"----- {sub}/{f.name} -----\n{_read(f)}")
    code = "\n\n".join(code_parts)[:_MAX_CODE_CHARS]
    reflection = _read(run_dir / "reflection.md")
    scores = ""
    for name in ("scores_train.json", "scores_test.json"):
        p = run_dir / "scoring" / name
        if p.exists():
            scores += f"\n## {name}\n{_read(p, 4000)}"
    return {"task": task, "ideas": ideas, "code": code,
            "reflection": reflection, "scores": scores}


def _prompt(axis: str, mode: str, mats: dict) -> str:
    mode_line = ("The task was INNOVATE: design a brand-new method for the "
                 "problem." if mode == "innovate" else
                 "The task was IMPROVE: improve the paper's EXISTING method "
                 "(existing code/infrastructure is available to build on).")
    return (
        "You are an expert scientific reviewer grading ONE property of an AI "
        f"model's research attempt.\n\n# PROPERTY TO GRADE\n{AXES[axis]}\n\n"
        f"{_CALIB}\n\n"
        f"# THE TASK ({mode})\n{mode_line}\n\n{mats['task']}\n\n"
        f"# THE MODEL'S IDEA (evolution)\n{mats['ideas'] or '(none)'}\n\n"
        f"# THE MODEL'S CODE (method/ + scripts/)\n{mats['code'] or '(none)'}\n\n"
        f"# THE MODEL'S REFLECTION\n{mats['reflection'] or '(none)'}\n\n"
        f"# REPORTED RESULTS\n{mats['scores'] or '(none)'}\n\n"
        f"# YOUR GRADING\nReturn an integer 0-100 for {axis.upper()} only, "
        f"plus a one-sentence reason. Follow the calibration."
    )


def grade_axis(run_dir: Path, mode: str, axis: str, mats: dict | None = None,
               timeout: int = 180) -> dict | None:
    """Grade a single property. Pass a pre-gathered ``mats`` to avoid re-reading
    the run for each of the 5 axes."""
    if axis not in AXES:
        return None
    if mats is None:
        mats = gather(run_dir)
    if not mats["ideas"] and not mats["code"]:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sf:
        json.dump(_SCHEMA, sf)
        schema_path = sf.name
    try:
        proc = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only",
             "--model", _MODEL, "-c", f"reasoning_effort={_EFFORT}",
             "--output-schema", schema_path, "-"],
            input=_prompt(axis, mode, mats),
            capture_output=True, text=True, timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        Path(schema_path).unlink(missing_ok=True)
    out = proc.stdout or ""
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{") and '"score"' in line:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d.get("score"), int):
                return d
    return None
