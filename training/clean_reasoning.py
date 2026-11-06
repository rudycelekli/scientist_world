"""Rewrite a run's chain of idea versions as a forward reasoning trace, used as
the <think> block of the SFT target.

The raw idea versions were written between experiments and cite their outcomes
(job ids, measured numbers). A model that proposes ideas without running
anything would learn from them to invent results. The rewrite keeps how the
idea evolved, as design reasoning made before any experiment: consider an
approach, weigh its trade-offs conceptually, revise, with no reference to
actual results, job ids or measured numbers. The final idea itself (already
consolidated in the dataset) is not touched.

One `codex exec` call per run -> {reasoning}. Needs the `codex` CLI, logged
in. The model and reasoning effort default to the published values (gpt-5.5,
medium) and can be set with SW_CODEX_MODEL / SW_CODEX_EFFORT or per call.
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
    "You convert an AI researcher's iterative idea notes into a single clean "
    "FORWARD reasoning trace that shows how they reasoned toward their final "
    "idea. STRICT RULES:\n"
    "(1) Write it as a-priori DESIGN reasoning — consider candidate approaches, "
    "weigh their tradeoffs conceptually, and revise — as if reasoning BEFORE "
    "running any experiments.\n"
    "(2) REMOVE every reference to actual experimental results, job IDs, run "
    "outcomes, or measured metric values. The reasoning must NOT cite any number "
    "that was obtained by running code.\n"
    "(3) When a later version revised an earlier one because an experiment "
    "regressed, REFRAME that as conceptual reasoning about why the earlier "
    "approach is weaker (e.g. 'spending budget on raw seeds likely underperforms "
    "because ...'), NOT as an observed result.\n"
    "(4) Do NOT fabricate results or invent numbers. Do NOT add new ideas.\n"
    "(5) End by pointing toward the final approach. No preamble, no meta-"
    "commentary — output the reasoning only."
)

_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}},
    "required": ["reasoning"],
    "additionalProperties": False,
}


def _prompt(task: str, idea_versions: list[str]) -> str:
    chain = "\n\n".join(f"===== idea_v{i} =====\n{v}"
                        for i, v in enumerate(idea_versions, 1))
    return (
        f"{_SYSTEM}\n\n# TASK THE RESEARCHER WAS GIVEN\n{task}\n\n"
        f"# THE RESEARCHER'S IDEA ITERATIONS (v1 .. v{len(idea_versions)})\n"
        f"{chain}\n\n# YOUR JOB\nWrite the clean forward reasoning trace per the "
        f"strict rules. Return it in the reasoning field."
    )


def clean_reasoning(task: str, idea_versions: list[str],
                    timeout: int = 180, model: str | None = None,
                    effort: str | None = None) -> str | None:
    if not idea_versions or len(idea_versions) < 2:
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
            input=_prompt(task, idea_versions), capture_output=True, text=True,
            timeout=timeout, cwd=tempfile.gettempdir())
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        Path(sp).unlink(missing_ok=True)
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and '"reasoning"' in line:
            try:
                return (json.loads(line).get("reasoning") or "").strip() or None
            except json.JSONDecodeError:
                continue
    return None
