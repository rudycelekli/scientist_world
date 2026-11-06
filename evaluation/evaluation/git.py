"""Local-only git operations for per-run workspaces.

Every run workspace is its own local repository; the harness commits
after each job and at the end of the run. Safety rules, enforced in
code:

1. This module never runs ``git push`` or ``git remote add``. Publishing
   a finished run is a separate, opt-in step (:mod:`evaluation.git_publish`)
   that works on a temporary copy, never on the workspace repository.
2. Never touch ``<env>/../source/``: that is the paper's cloned
   repository, owned by sciworld and read-only for the harness.
3. Never operate outside a per-run workspace
   ``<data_dir>/envs/<paper>/<run_name>/``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from evaluation import config as _config


class GitSafetyError(RuntimeError):
    """Raised when an operation would violate a hard safety rule."""


def _assert_per_run_workspace(env_dir: Path) -> None:
    """Reject anything outside ``<data_dir>/envs/<paper>/<run>/``."""
    env_dir = env_dir.expanduser().resolve()
    envs_root = _config.envs_dir().resolve()

    try:
        rel = env_dir.relative_to(envs_root)
    except ValueError:
        raise GitSafetyError(
            f"refusing to operate on {env_dir!r} — not under envs dir {envs_root!r}"
        )

    # Must be exactly two levels below envs/: <paper>/<run>/
    parts = rel.parts
    if len(parts) != 2:
        raise GitSafetyError(
            f"refusing to operate on {env_dir!r} — "
            f"expected <data_dir>/envs/<paper>/<run>/, got depth {len(parts)}"
        )

    # Never touch the sciworld-managed clone.
    if env_dir.name == "source" or parts[-1] == "source":
        raise GitSafetyError(
            f"refusing to operate on {env_dir!r} — that's the paper source clone"
        )


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a git subcommand with push-blocking."""
    if args and args[0] == "push":
        raise GitSafetyError("git push is forbidden in the Evaluation pipeline")
    if args[:2] == ["remote", "add"]:
        raise GitSafetyError("git remote add is forbidden in the Evaluation pipeline")
    return subprocess.run(
        ["git", *args], cwd=str(cwd),
        capture_output=True, text=True, check=False,
    )


def init_repo(env_dir: Path) -> None:
    """Initialize a fresh local-only repo at *env_dir*.

    Idempotent — no-op if ``.git`` already exists.
    """
    _assert_per_run_workspace(env_dir)
    if (env_dir / ".git").is_dir():
        return

    result = _run(["init", "--initial-branch=main"], env_dir)
    if result.returncode != 0:
        raise RuntimeError(f"git init failed: {result.stderr.strip()}")

    # Identity for commits (git refuses to commit without one).
    _run(["config", "user.name", "evaluation"], env_dir)
    _run(["config", "user.email", "evaluation@local"], env_dir)


def commit(env_dir: Path, message: str, add_all: bool = True) -> bool:
    """Stage and commit. Returns True if a commit was created.

    False means there was nothing to commit (still a success, just an
    empty diff).

    Retries through transient failures seen on network filesystems
    (object creation, index lock contention): two retries with a brief
    backoff, then the error is raised.
    """
    import time as _time

    _assert_per_run_workspace(env_dir)
    if not (env_dir / ".git").is_dir():
        raise RuntimeError(f"not a git repo: {env_dir}")

    if add_all:
        _run(["add", "-A"], env_dir)

    status = _run(["diff", "--cached", "--quiet"], env_dir)
    if status.returncode == 0:
        return False

    last_err = ""
    for attempt in range(3):
        if attempt:
            _time.sleep(2 * attempt)  # 0, 2, 4 s
        result = _run(["commit", "-m", message], env_dir)
        if result.returncode == 0:
            return True
        combined = (result.stdout + result.stderr).lower()
        if "nothing to commit" in combined:
            return False
        last_err = result.stderr.strip() or result.stdout.strip()
        # Transient filesystem / lockfile errors worth retrying.
        if not any(s in combined for s in (
            "unable to create",
            "index.lock",
            "input/output error",
            "stale file handle",
            "no space left on device",  # can be a transient quota race
        )):
            break
        # Clean up a stale index.lock between attempts.
        idx_lock = env_dir / ".git" / "index.lock"
        if idx_lock.exists():
            try:
                idx_lock.unlink()
            except OSError:
                pass
    raise RuntimeError(f"git commit failed: {last_err}")


# Lines removed from the template: the harness tracks trajectories and
# build logs in the run repository (small, and needed for later analysis).
_GITIGNORE_DROP = frozenset({"trajectory/", "logs/"})

# Lines appended to the template: agent credential and session files the
# harness copies into the workspace, which must never be committed. The
# ``*.agent-original`` names are where the verifier parks the agent's
# Claude session state before a Claude grader starts.
_GITIGNORE_EXTRA = (
    "# Agent credentials and session state copied in by the harness",
    ".claude.json",
    ".claude.agent-original/",
    ".claude.json.agent-original",
    ".kimi/",
    ".gemini/",
)


def ensure_credentials_ignored(env_dir: Path) -> bool:
    """Add any missing ``_GITIGNORE_EXTRA`` entry to the workspace ``.gitignore``.

    For workspaces whose ``.gitignore`` was installed before those entries
    existed: the verifier calls this before it parks the agent's Claude
    session state, so the parked credential file is never committed.
    Returns True if the file was changed.
    """
    _assert_per_run_workspace(env_dir)
    path = env_dir / ".gitignore"
    lines = path.read_text().splitlines() if path.exists() else []
    present = {line.strip() for line in lines}
    missing = [entry for entry in _GITIGNORE_EXTRA if entry not in present]
    if not missing:
        return False
    if lines and lines[-1].strip():
        lines.append("")
    path.write_text("\n".join(lines + missing) + "\n")
    return True


def install_gitignore(env_dir: Path, template_gitignore: Path) -> None:
    """Install the workspace ``.gitignore`` from a template, patched.

    The template is the worker's workspace ``.gitignore``. Two entries
    are removed (see ``_GITIGNORE_DROP``) and the agent credential paths
    in ``_GITIGNORE_EXTRA`` are appended.
    """
    _assert_per_run_workspace(env_dir)
    src = Path(template_gitignore).expanduser()
    if not src.exists():
        raise FileNotFoundError(
            f"workspace .gitignore template not found: {src}. Run from a "
            "full checkout of the repository or set workspace_gitignore "
            "in the evaluation config."
        )

    patched = [
        line for line in src.read_text().splitlines()
        if line.strip() not in _GITIGNORE_DROP
    ]
    patched.append("")
    patched.extend(_GITIGNORE_EXTRA)
    (env_dir / ".gitignore").write_text("\n".join(patched) + "\n")
