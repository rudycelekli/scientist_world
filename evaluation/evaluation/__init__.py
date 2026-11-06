"""Evaluation harness for Scientist World environments.

Runs a "scientist" agent on a paper environment, records what it did,
has an independent grader re-score it, and normalizes the scores across
papers (see ``README.md`` and ``docs/scoring.md``).

Python entry points:

    from evaluation import run, bench, serve

    run(paper_id, profile_name, mode="innovate")   # one trial, blocking
    bench(papers, profiles, modes=("innovate",))   # queue a matrix of trials
    serve(port=38582)                              # dashboard + queue (blocks)
"""

from __future__ import annotations

from evaluation.orchestrator import bench
from evaluation.runner import run_one as run

__all__ = ["run", "bench", "serve"]


def serve(port: int | None = None, host: str | None = None) -> None:
    """Launch the Flask dashboard (blocks)."""
    from evaluation.dashboard import serve as _serve
    _serve(port=port, host=host)
