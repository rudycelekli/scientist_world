"""Consolidate a run's cumulative idea-version chain into ONE self-contained
final-idea statement, faithfully, WITHOUT improving it.

The final ``idea_vN.md`` is written as a delta on earlier versions ("v6 said
X", "return to V1's approach"), so it can't be read standalone. A model
restates *the final version's method* as a self-contained description,
resolving back-references into plain statements and changing nothing else.

It is deliberately EXTRACTIVE: no new ideas, no improvement, no judgement.

The call goes through the ``codex`` CLI on the host (``codex exec`` with a
read-only sandbox and ``--output-schema``), so it uses whatever login that
CLI has. Model and reasoning effort come from ``$CONSOLIDATE_MODEL`` and
``$CONSOLIDATE_EFFORT``.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

_SYSTEM = (
    "You consolidate an AI researcher's iterative idea notes into a single "
    "self-contained statement of their FINAL idea. You are a faithful "
    "transcriber, not a reviewer or co-author. STRICT RULES: "
    "(1) Output ONLY the final version's method/idea, as it stands at the last "
    "iteration. "
    "(2) Resolve references to earlier versions ('v6 showed', 'unchanged from "
    "v2', 'return to v1') into plain self-contained statements, using earlier "
    "versions only to fill in what the final version refers to. "
    "(3) Do NOT improve, extend, critique, or add anything. Do NOT invent "
    "details not present in the notes. "
    "(4) If the final version is already self-contained, restate it faithfully. "
    "(5) No preamble or meta-commentary — the idea text only."
)

_SCHEMA = {
    "type": "object",
    "properties": {"final_idea": {"type": "string"}},
    "required": ["final_idea"],
    "additionalProperties": False,
}

# Defaults are the settings used for the released dataset. Override them
# through the environment for a later batch.
_MODEL = os.environ.get("CONSOLIDATE_MODEL", "gpt-5.5")
_EFFORT = os.environ.get("CONSOLIDATE_EFFORT", "medium")


def _prompt(inputs: dict, idea_versions: list[str]) -> str:
    task = "\n\n".join(
        f"## {k.upper()}\n{inputs[k]}" for k in
        ("problem", "direction", "constraint", "target")
        if inputs.get(k)
    )
    chain = "\n\n".join(
        f"===== idea_v{i} =====\n{v}" for i, v in enumerate(idea_versions, 1)
    )
    return (
        f"{_SYSTEM}\n\n"
        f"# TASK THE RESEARCHER WAS GIVEN\n{task}\n\n"
        f"# THE RESEARCHER'S IDEA ITERATIONS (v1 .. v{len(idea_versions)})\n"
        f"{chain}\n\n"
        f"# YOUR JOB\nWrite the FINAL idea (idea_v{len(idea_versions)}) as a "
        f"single self-contained description, following the strict rules. "
        f"Return it in the final_idea field."
    )


def consolidate(inputs: dict, idea_versions: list[str],
                timeout: int = 180) -> str | None:
    """Return the consolidated final-idea text, or None on failure.

    Single-version (or empty) chains need no call — returned as-is.
    """
    if not idea_versions:
        return None
    if len(idea_versions) == 1:
        return idea_versions[0]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sf:
        json.dump(_SCHEMA, sf)
        schema_path = sf.name
    try:
        proc = subprocess.run(
            ["codex", "exec",
             "--skip-git-repo-check",
             "--sandbox", "read-only",
             "--model", _MODEL,
             "-c", f"reasoning_effort={_EFFORT}",
             "--output-schema", schema_path,
             "-"],
            input=_prompt(inputs, idea_versions),
            capture_output=True, text=True, timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        Path(schema_path).unlink(missing_ok=True)

    # codex prints session noise; the structured answer is the last JSON
    # object on stdout matching our schema.
    out = proc.stdout or ""
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{") and "final_idea" in line:
            try:
                return (json.loads(line).get("final_idea") or "").strip() or None
            except json.JSONDecodeError:
                continue
    return None
