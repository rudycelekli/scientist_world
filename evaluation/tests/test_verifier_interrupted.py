"""A grader session we lost is not a verdict.

Provider quota, expired auth, gateway blips, failed container builds and
exhausted turn budgets must leave a cell RETRYABLE. Only a grader that
actually audited the run and still could not conclude earns 'exhausted'.
Otherwise an outage that never graded anything would record a verdict
for every run it touched.
"""
import pytest

from evaluation import verifier


def _classify(reason: str, turns: int, scores=None) -> str:
    """Mirror of the status decision in verify_run's finally block."""
    ver = dict(scores or {})
    if ver.get("status") in verifier.TERMINAL_STATUSES:
        return ver["status"]
    interrupted = (
        not turns
        or any(m in str(reason) for m in verifier._RETRYABLE_REASON_MARKERS)
    )
    return verifier.RETRYABLE_STATUS if interrupted else "exhausted"


@pytest.mark.parametrize("reason,turns", [
    # the grader never launched
    ("grader session error (attempt 2/2): TransientError: agent exited rc=1", 0),
    # quota ran out in the middle of an audit
    ("grader session error (attempt 1/2): TransientError: agent exited rc=1", 5),
    ("provider quota exceeded", 12),
    ("rate limit reached", 3),
    ("HTTP 429 from gateway", 7),
    ("setup error: RuntimeError: apptainer build failed (rc=255)", 0),
    ("setup error: RuntimeError: setup.sh failed (rc=1)", 2),
    ("max_turns", 60),
])
def test_interrupted_sessions_stay_retryable(reason, turns):
    assert _classify(reason, turns) == verifier.RETRYABLE_STATUS


def test_retryable_status_is_not_terminal():
    """Retryable cells must be picked up again, never treated as done."""
    assert verifier.RETRYABLE_STATUS not in verifier.TERMINAL_STATUSES


def test_genuine_failure_to_conclude_is_exhausted():
    """Grader ran a full session, produced no verdict -> real exhausted."""
    assert _classify("done", 12) == "exhausted"


def test_real_verdicts_are_untouched():
    for status in ("verified", "unfixable"):
        assert _classify("done", 12, {"status": status}) == status
