"""Prompt builders for the builder agents.

Each builder loads a static markdown template from ``prompts/`` and fills
its ``str.format`` placeholders with runtime context.
"""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# ── Tools sections (per scaffold) ────────────────────────────────────────────

TOOLS_CLAUDE_CODE = """\

---

## Tools

You are running as a Claude Code agent. You have access to:

- **Bash** — run shell commands (git, pip, python, curl, wget, etc.)
- **Read** / **Write** / **Edit** — file operations
- **Glob** / **Grep** — search files by name or content
- **WebSearch** / **WebFetch** — search the web and fetch URLs

Use Bash to run code, install packages, download data, and interact with git.
Use the file tools for reading and writing code. Prefer Edit over sed/awk."""

TOOLS_CODEX = """\

---

## Tools

You are running as a Codex agent. You have access to:

- **shell** — run shell commands (git, pip, python, curl, wget, etc.)
- **web_search** — search the web for information

Use shell commands to read/write files, run code, install packages, download data, and interact with git."""

TOOLS_BY_SCAFFOLD = {
    "claude-code": TOOLS_CLAUDE_CODE,
    "codex": TOOLS_CODEX,
}


# ── Prompt builders ──────────────────────────────────────────────────────────

def build_reproduce_prompt(
    paper_md_path: str,
    paper_images_dir: str,
    paper_pdf_path: str,
    workspace_path: str,
    gpu_budget: float,
    max_gpus: int = 4,
    job_time_limit: str = "24:00:00",
    shared_datasets_path: str = "/shared/datasets",
    shared_models_path: str = "/shared/models",
    scaffold: str = "claude-code",
) -> str:
    """Build the full system prompt for the reproduce() call.

    Loads reproduce_start.md and fills in runtime context placeholders.

    Args:
        paper_md_path: Path to the paper's markdown file.
        paper_images_dir: Path to the paper's images directory.
        paper_pdf_path: Path to the paper's PDF file.
        workspace_path: Path to the agent's workspace directory.
        gpu_budget: Total GPU-hours budget for this paper.
        max_gpus: Maximum GPUs the agent can request per job.
        job_time_limit: Wall time limit per job (HH:MM:SS format).
        shared_datasets_path: Path to shared datasets directory.
        shared_models_path: Path to shared models directory.
        scaffold: Agent scaffold name (for tools section).

    Returns:
        Complete prompt string ready to send to the agent.
    """
    template = (PROMPT_DIR / "reproduce_start.md").read_text()
    tools_section = TOOLS_BY_SCAFFOLD.get(scaffold, TOOLS_CLAUDE_CODE)

    return template.format(
        paper_md_path=paper_md_path,
        paper_images_dir=paper_images_dir,
        paper_pdf_path=paper_pdf_path,
        workspace_path=workspace_path,
        gpu_budget=gpu_budget,
        max_gpus=max_gpus,
        job_time_limit=job_time_limit,
        shared_datasets_path=shared_datasets_path,
        shared_models_path=shared_models_path,
        tools_section=tools_section,
    )


def build_job_completion_prompt(
    run_id: str,
    stdout_path: str,
    stderr_path: str,
    returncode: int,
) -> str:
    """Build the prompt delivered when a compute job finishes.

    Args:
        run_id: Job identifier.
        stdout_path: Path to the stdout log file.
        stderr_path: Path to the stderr log file.
        returncode: Process exit code (0 = success).

    Returns:
        Complete prompt string.
    """
    template = (PROMPT_DIR / "on_job_completion.md").read_text()
    return template.format(
        run_id=run_id,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        returncode=returncode,
    )


def build_job_progress_prompt(
    run_id: str,
    stdout_path: str,
    stderr_path: str,
    elapsed_min: int = 0,
) -> str:
    """Build the prompt for periodic job monitoring.

    Args:
        run_id: Job identifier.
        stdout_path: Path to the stdout log file.
        stderr_path: Path to the stderr log file.
        elapsed_min: Minutes the hive job has been in the *running*
            state (queued time not counted).  Displayed to the agent
            so it can tell how long it has been since real work began.

    Returns:
        Complete prompt string.
    """
    template = (PROMPT_DIR / "on_job_progress.md").read_text()
    return template.format(
        run_id=run_id,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        elapsed_min=elapsed_min,
    )


def build_job_cancelled_prompt(
    run_id: str,
    stdout_path: str,
    stderr_path: str,
) -> str:
    """Build the prompt when a job is cancelled by the agent.

    Args:
        run_id: Job identifier.
        stdout_path: Path to the stdout log file.
        stderr_path: Path to the stderr log file.

    Returns:
        Complete prompt string.
    """
    template = (PROMPT_DIR / "on_job_cancelled.md").read_text()
    return template.format(
        run_id=run_id,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def build_container_build_failure_prompt(
    build_log_path: str,
    *,
    phase: str = "build",
    failure_class: str = "unknown",
    failure_signature: str = "",
    failure_hint: str = "",
    env_comparison: str = "first_failure",
    same_signature: bool = False,
) -> str:
    """Build the prompt when the container fails to build.

    Includes the last 40 lines of the build log directly in the prompt
    so the agent sees the error immediately without an extra tool call.

    Args:
        build_log_path: Path to the build log file.

    Returns:
        Complete prompt string.
    """
    # Read tail of build log to include inline
    log_tail = ""
    try:
        lines = Path(build_log_path).read_text().splitlines()
        log_tail = "\n".join(lines[-40:])
    except Exception:
        log_tail = "(could not read build log)"

    template = (PROMPT_DIR / "on_container_build_failure.md").read_text()
    if env_comparison == "unchanged":
        env_change_note = (
            "Your `environment/container.def` and `environment/setup.sh` are "
            "unchanged since the previous failed environment attempt."
        )
    elif env_comparison == "changed":
        env_change_note = (
            "Your environment files changed since the previous failed "
            "environment attempt."
        )
    else:
        env_change_note = (
            "This is the first recorded environment failure in this session, "
            "so there is no previous environment attempt to compare against."
        )

    if same_signature:
        retry_note = (
            "The scheduler thinks this looks like the same failure signature as "
            "the previous environment attempt."
        )
    else:
        retry_note = (
            "The scheduler does not think this matches the previous failure "
            "signature exactly."
        )

    return template.format(
        build_log_path=build_log_path,
        phase=phase,
        phase_cap=phase.capitalize(),
        failure_class=failure_class,
        failure_signature=failure_signature or "unknown",
        failure_hint=failure_hint or "No scheduler hint is available for this failure.",
        env_change_note=env_change_note,
        retry_note=retry_note,
        log_tail=log_tail,
    )


def build_budget_exhausted_prompt(
    gpu_hours_used: float,
    best_milestone: str | None,
) -> str:
    """Build the prompt when budget is spent. Combine with wrap-up.

    Args:
        gpu_hours_used: Total GPU-hours consumed.
        best_milestone: Highest milestone reached so far (or "none").

    Returns:
        Complete prompt string (budget exhausted + wrap up).
    """
    exhausted = (PROMPT_DIR / "on_budget_exhausted.md").read_text().format(
        gpu_hours_used=gpu_hours_used,
        best_milestone=best_milestone or "none",
    )
    wrap_up = (PROMPT_DIR / "on_wrap_up.md").read_text().format(
        best_milestone=best_milestone or "none",
    )
    return exhausted + "\n" + wrap_up


def build_continue_prompt() -> str:
    """Build the prompt for resuming a previous session.

    Returns:
        Complete prompt string.
    """
    return (PROMPT_DIR / "on_continue.md").read_text()


def build_iterate_prompt(
    previous_milestone: str,
    starting_milestone: str,
) -> str:
    """Build the prompt for an iteration agent auditing previous work.

    Args:
        previous_milestone: The milestone claimed by the previous agent.
        starting_milestone: The milestone this agent starts at (one below).

    Returns:
        Complete prompt string.
    """
    template = (PROMPT_DIR / "on_iterate.md").read_text()
    return template.format(
        previous_milestone=previous_milestone,
        starting_milestone=starting_milestone,
    )


def build_verify_prompt(
    previous_milestone: str,
) -> str:
    """Build the prompt for a verifier agent doing final quality check.

    Args:
        previous_milestone: The milestone reached by previous agents.

    Returns:
        Complete prompt string.
    """
    template = (PROMPT_DIR / "on_verify.md").read_text()
    return template.format(
        previous_milestone=previous_milestone,
    )


def build_packager_prompt(
    workspace_path: str,
    scaffold: str = "claude-code",
) -> str:
    """Build the full first-turn prompt for the packager role.

    Packager does not see reproduce_start.md; this is the only system
    prompt it gets. No paper content is injected (saves tokens).

    Args:
        workspace_path: Path to the agent's workspace inside the container.
        scaffold: Agent scaffold name (for tools section).

    Returns:
        Complete prompt string.
    """
    template = (PROMPT_DIR / "on_packager.md").read_text()
    tools_section = TOOLS_BY_SCAFFOLD.get(scaffold, TOOLS_CLAUDE_CODE)
    return template.format(
        workspace_path=workspace_path,
        tools_section=tools_section,
    )


def build_designer_prompt(
    workspace_path: str,
    paper_md_path: str,
    previous_milestone: str,
    scaffold: str = "claude-code",
    stage: int | None = None,
) -> str:
    """Build the full first-turn prompt for the designer role.

    Designer does not see reproduce_start.md. Paper is referenced by
    path; the agent decides whether to open it.

    Args:
        workspace_path: Path to the agent's workspace inside the container.
        paper_md_path: Path to the paper markdown (agent may open if needed).
        previous_milestone: Milestone reached by previous agents (inherited).
        scaffold: Agent scaffold name (for tools section).
        stage: If 1/2/3, load on_designer_stage_N.md (three-stage split:
            architect / auditor / runner). If None, load on_designer.md
            (legacy single-stage).

    Returns:
        Complete prompt string.
    """
    if stage in (1, 2, 3):
        template_name = f"on_designer_stage_{stage}.md"
    else:
        template_name = "on_designer.md"
    template = (PROMPT_DIR / template_name).read_text()
    tools_section = TOOLS_BY_SCAFFOLD.get(scaffold, TOOLS_CLAUDE_CODE)
    return template.format(
        workspace_path=workspace_path,
        paper_md_path=paper_md_path,
        previous_milestone=previous_milestone,
        tools_section=tools_section,
    )


def build_validation_failure_prompt(
    validation_output: str,
    attempt: int,
    max_attempts: int,
) -> str:
    """Build the prompt when validate.py fails after verifier signals done.

    Args:
        validation_output: stdout from validate.py.
        attempt: Current attempt number.
        max_attempts: Maximum attempts allowed.

    Returns:
        Complete prompt string.
    """
    template = (PROMPT_DIR / "on_validation_failure.md").read_text()
    return template.format(
        validation_output=validation_output,
        attempt=attempt,
        max_attempts=max_attempts,
    )


def build_invalid_milestone_prompt(
    bad_milestone: str,
    attempt: int,
    max_attempts: int,
) -> str:
    """Build the prompt when action.yaml has a milestone not in the canonical set.

    Used to wake the agent and have them pick a valid milestone name when
    their action.yaml declared an invented one (e.g. "extended_results").
    """
    template = (PROMPT_DIR / "on_invalid_milestone.md").read_text()
    return template.format(
        bad_milestone=bad_milestone,
        attempt=attempt,
        max_attempts=max_attempts,
    )


def build_push_forward_prompt(
    current_milestone: str,
    target_milestone: str,
    attempt: int = 1,
    max_attempts: int = 1,
) -> str:
    """Build the prompt to push the agent past a milestone threshold.

    Used when the agent signals done but hasn't reached the target.
    """
    template = (PROMPT_DIR / "on_push_forward.md").read_text()
    return template.format(
        current_milestone=current_milestone or "none",
        target_milestone=target_milestone,
        attempt=attempt,
        max_attempts=max_attempts,
    )


def build_wrap_up_prompt(
    best_milestone: str | None,
) -> str:
    """Build the prompt to finalize work and clean up.

    Used when the agent signals action: done.

    Args:
        best_milestone: Highest milestone reached so far (or "none").

    Returns:
        Complete prompt string.
    """
    template = (PROMPT_DIR / "on_wrap_up.md").read_text()
    return template.format(
        best_milestone=best_milestone or "none",
    )
