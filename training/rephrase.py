"""Faithful rewording of an SFT completion (reasoning + idea).

SFT-only augmentation: the best-scoring examples are added a second time in a
different surface form, to enlarge the SFT set while the held-out papers stay
excluded. This suits SFT (imitation), where there is no reward that a reworded
copy could misattribute; the GRPO set does not use it.

The rewording keeps the same reasoning steps, the same method and the same
details; only sentence structure and word choice change. The <think>...</think>
structure is preserved.

One `codex exec` call per completion -> {reworded}. Needs the `codex` CLI,
logged in. The model and reasoning effort default to the published values
(gpt-5.5, medium) and can be set with SW_CODEX_MODEL / SW_CODEX_EFFORT or per
call.
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
    "You faithfully REWORD a research reasoning-and-idea text. STRICT RULES:\n"
    "(1) Keep the meaning EXACTLY the same — same reasoning steps in the same "
    "order, same proposed method, same design choices, same details and any "
    "numbers.\n"
    "(2) Change ONLY sentence structure and word choice. Do NOT add, remove, "
    "reorder, or alter any idea, step, or detail.\n"
    "(3) If the text contains a <think> ... </think> block followed by the "
    "idea, preserve that exact structure: reword the content inside <think> and "
    "the idea after it, keeping the <think> and </think> tags in place.\n"
    "(4) No preamble or commentary — output only the reworded text."
)

_SCHEMA = {
    "type": "object",
    "properties": {"reworded": {"type": "string"}},
    "required": ["reworded"],
    "additionalProperties": False,
}


def rephrase(completion: str, timeout: int = 180, model: str | None = None,
             effort: str | None = None) -> str | None:
    if not completion or not completion.strip():
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sf:
        json.dump(_SCHEMA, sf)
        sp = sf.name
    prompt = f"{_SYSTEM}\n\n# TEXT TO REWORD\n{completion}\n\n# OUTPUT\nReturn the reworded text in the reworded field."
    try:
        proc = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only",
             "--model", model or DEFAULT_MODEL,
             "-c", f"reasoning_effort={effort or DEFAULT_EFFORT}",
             "--output-schema", sp, "-"],
            input=prompt, capture_output=True, text=True,
            timeout=timeout, cwd=tempfile.gettempdir())
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        Path(sp).unlink(missing_ok=True)
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and '"reworded"' in line:
            try:
                return (json.loads(line).get("reworded") or "").strip() or None
            except json.JSONDecodeError:
                continue
    return None
