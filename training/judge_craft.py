"""Craft judge: score a research idea's intrinsic quality from the task and the
idea text alone, with no access to the outcome, the score or the code.

Optional component. It was built as a GRPO reward source (`judge`,
`judge_novelty` in data_prep.grpo_dataset) but was not used for the reported
results, whose rewards are measured outcomes only.

Unlike a grader that sees the measured result, this judge is independent of
the outcome: a well-conceived idea that happened to fail can still score high.
It needs only the idea text, so it can also score newly sampled ideas.

One `codex exec` call per idea -> {novelty, soundness, overall, reason}. Needs
the `codex` CLI, logged in. The model and reasoning effort default to gpt-5.5
and medium and can be set with SW_CODEX_MODEL / SW_CODEX_EFFORT or per call.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

DEFAULT_MODEL = os.environ.get("SW_CODEX_MODEL") or "gpt-5.5"
DEFAULT_EFFORT = os.environ.get("SW_CODEX_EFFORT") or "medium"

_SYSTEM = (
    "You are an expert reviewer scoring the INTRINSIC quality of a proposed "
    "research idea, judged ONLY from the idea itself and the task it addresses. "
    "You do NOT know whether it worked — do not speculate about results. Score "
    "the idea as a proposal. Three scores, each 0-100:\n"
    "- novelty: how original / non-obvious the idea is for this task (vs a "
    "routine, generic, or textbook move).\n"
    "- soundness: how technically sound and plausible the idea is — is the "
    "mechanism coherent, does it fit the constraints, could it credibly work.\n"
    "- overall: your holistic judgment of the idea's quality as a research "
    "proposal.\n"
    "Calibration: 0-20 poor, 21-40 weak, 41-60 competent, 61-80 strong, "
    "81-100 exceptional. Use the full range. Judge the idea on its merits as "
    "conceived, independent of execution or luck."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "novelty": {"type": "integer", "minimum": 0, "maximum": 100},
        "soundness": {"type": "integer", "minimum": 0, "maximum": 100},
        "overall": {"type": "integer", "minimum": 0, "maximum": 100},
        "reason": {"type": "string"},
    },
    "required": ["novelty", "soundness", "overall", "reason"],
    "additionalProperties": False,
}


def _prompt(task: str, idea: str) -> str:
    return (
        f"{_SYSTEM}\n\n# THE TASK\n{task}\n\n# THE PROPOSED IDEA\n{idea}\n\n"
        f"# YOUR SCORES\nReturn novelty, soundness, overall (0-100 integers) and "
        f"a one-sentence reason. Judge the idea only — you do not know its result."
    )


def judge(task: str, idea: str, timeout: int = 150, model: str | None = None,
          effort: str | None = None) -> dict | None:
    """Return {novelty, soundness, overall, reason} or None on failure."""
    if not idea or not idea.strip():
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sf:
        json.dump(_SCHEMA, sf)
        sp = sf.name
    try:
        proc = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only",
             "--model", model or DEFAULT_MODEL,
             "-c", f"reasoning_effort={effort or DEFAULT_EFFORT}",
             "--output-schema", sp, "-"],
            input=_prompt(task, idea), capture_output=True, text=True,
            timeout=timeout, cwd=tempfile.gettempdir())
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        Path(sp).unlink(missing_ok=True)
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and '"overall"' in line:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(k in d for k in ("novelty", "soundness", "overall")):
                return d
    return None
