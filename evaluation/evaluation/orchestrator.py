"""Bench fan-out — evaluate a matrix of (papers × models) concurrently.

Two-level admission control:

- ``max_concurrent`` (global) caps total in-flight runs.
- ``profile_concurrency`` (per-profile, optional) caps how many runs of
  any one profile can be in-flight simultaneously, to stay under one
  provider's rate limits.

A background dispatcher thread admits queued items into the executor
only when *both* caps have headroom. ``bench()`` itself is non-blocking
and only appends to the pending list — it never holds an executor slot.

The queue lives in memory only. On a process restart, pending runs are
dropped; runs that were in flight can be queued again and resume in
place (see ``runner.run_one``).
"""

from __future__ import annotations

import concurrent.futures as _futures
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from evaluation import config as _config
from evaluation.runner import RunResult, run_one

log = logging.getLogger(__name__)


@dataclass
class QueuedRun:
    paper_id: str
    profile_name: str
    mode: str = "innovate"


# Process-wide queues + executor. The dashboard and the bench CLI both
# enqueue into the same pool.
_lock = threading.Lock()
_executor: _futures.ThreadPoolExecutor | None = None
_dispatcher_thread: threading.Thread | None = None
_pending: list[QueuedRun] = []
_running: dict[str, QueuedRun] = {}  # key → QueuedRun
_completed: list[RunResult] = []
_wakeup = threading.Event()  # signal the dispatcher when state changes

# Live overrides for concurrency caps. These take precedence over the
# YAML config so an operator can resize the pool from the dashboard
# without restarting the server. None = no override (use yaml).
_max_concurrent_override: int | None = None
_profile_caps_override: dict[str, int] | None = None


def _runkey(q: QueuedRun) -> str:
    return f"{q.paper_id}|{q.profile_name}|{q.mode}"


def _max_concurrent() -> int:
    if _max_concurrent_override is not None:
        return _max_concurrent_override
    return int(_config.load().get("max_concurrent", 10))


def _profile_caps() -> dict[str, int]:
    """Per-profile concurrency limits. Profiles not listed are uncapped
    (only constrained by ``max_concurrent``)."""
    if _profile_caps_override is not None:
        return dict(_profile_caps_override)
    raw = _config.load().get("profile_concurrency") or {}
    out: dict[str, int] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            log.warning("Bad profile_concurrency entry %r=%r — skipping", k, v)
    return out


def set_caps(
    max_concurrent: int | None = None,
    profile_caps: dict[str, int] | None = None,
) -> dict:
    """Update concurrency caps live. Either field may be None to leave
    that override untouched. Pass an empty dict for ``profile_caps`` to
    clear all per-profile caps. Resizes the executor up if needed and
    wakes the dispatcher so the new caps admit any queued runs at once.
    """
    global _max_concurrent_override, _profile_caps_override
    with _lock:
        if max_concurrent is not None:
            _max_concurrent_override = int(max_concurrent)
        if profile_caps is not None:
            _profile_caps_override = {str(k): int(v) for k, v in profile_caps.items()}
        # Resize the pool up to match the new global cap. (Lowering does
        # not terminate threads; they drain naturally as runs complete.)
        if _executor is not None:
            target = _max_concurrent()
            if target > _executor._max_workers:
                _executor._max_workers = target
    _wakeup.set()
    return current_caps()


def current_caps() -> dict:
    """Snapshot of the caps currently in force (overrides if set, else yaml)."""
    return {
        "max_concurrent": _max_concurrent(),
        "profile_concurrency": _profile_caps(),
        "override_active": _max_concurrent_override is not None
                           or _profile_caps_override is not None,
        "pool_max_workers": _executor._max_workers if _executor is not None else None,
    }


def _ensure_executor() -> _futures.ThreadPoolExecutor:
    global _executor
    if _executor is None:
        # Pool sized at max_concurrent. The dispatcher only submits when
        # a slot is genuinely free, so the pool's queue stays empty —
        # no wasted threads blocked on a per-profile semaphore.
        _executor = _futures.ThreadPoolExecutor(
            max_workers=_max_concurrent(),
            thread_name_prefix="eval-run",
        )
    return _executor


def _start_dispatcher() -> None:
    global _dispatcher_thread
    if _dispatcher_thread is not None and _dispatcher_thread.is_alive():
        return
    _dispatcher_thread = threading.Thread(
        target=_dispatch_loop, name="eval-dispatch", daemon=True,
    )
    _dispatcher_thread.start()


def _dispatch_loop() -> None:
    """Admit pending runs into the executor while respecting both caps.

    Wakes on every enqueue / completion (via ``_wakeup``) and on a
    background tick (in case a wakeup is missed). Polite scan: O(pending)
    per pass, but we break early once the global cap is full.
    """
    while True:
        _wakeup.wait(timeout=1.0)
        _wakeup.clear()
        try:
            _admit_pass()
        except Exception:  # never let the dispatcher die silently
            log.exception("dispatcher pass crashed")


def _admit_pass() -> None:
    cap_global = _max_concurrent()
    cap_per_profile = _profile_caps()
    exe = _ensure_executor()
    with _lock:
        in_flight = len(_running)
        per_profile = Counter(q.profile_name for q in _running.values())
        admitted: list[QueuedRun] = []
        # Single pass through pending, in submission order.
        for q in list(_pending):
            if in_flight >= cap_global:
                break
            cap_p = cap_per_profile.get(q.profile_name)
            if cap_p is not None and per_profile[q.profile_name] >= cap_p:
                continue  # this profile is full, skip but keep its queue order
            _pending.remove(q)
            _running[_runkey(q)] = q
            admitted.append(q)
            in_flight += 1
            per_profile[q.profile_name] += 1
    for q in admitted:
        exe.submit(_run_and_record, q)


def _run_and_record(q: QueuedRun) -> RunResult:
    key = _runkey(q)
    try:
        result = run_one(q.paper_id, q.profile_name, mode=q.mode)
    except Exception as exc:
        log.exception("Run %s crashed", key)
        result = RunResult(
            run_id=key, paper_id=q.paper_id, model=q.profile_name,
            mode=q.mode, status="failed", attempt=0, env_dir="",
            train_score=None, test_score=None, error=str(exc),
        )
    with _lock:
        _running.pop(key, None)
        _completed.append(result)
    _wakeup.set()  # a slot just freed — let the dispatcher fill it
    return result


def bench(
    papers: Iterable[str],
    models: Iterable[str],
    modes: Iterable[str] = ("innovate",),
) -> None:
    """Enqueue every (paper, model, mode) triple. Non-blocking.

    Admission to the executor is handled by the background dispatcher
    thread, which respects both the global ``max_concurrent`` and
    per-profile ``profile_concurrency`` caps from config (or the live
    overrides set through :func:`set_caps`).
    """
    _ensure_executor()
    _start_dispatcher()
    with _lock:
        for paper_id in papers:
            for profile_name in models:
                for mode in modes:
                    _pending.append(QueuedRun(
                        paper_id=paper_id,
                        profile_name=profile_name,
                        mode=mode,
                    ))
    _wakeup.set()


def remove_pending(
    models: Iterable[str] | None = None,
    papers: Iterable[str] | None = None,
    modes: Iterable[str] | None = None,
) -> int:
    """Remove queued (not-yet-running) cells from the pending queue.

    Filters are ANDed; a ``None`` filter matches everything on that field.
    Only affects ``_pending`` — runs already dispatched to ``_running`` are
    untouched (they must finish or be cancelled separately). Returns the
    number of pending cells removed.
    """
    mset = set(models) if models is not None else None
    pset = set(papers) if papers is not None else None
    dset = set(modes) if modes is not None else None
    removed = 0
    with _lock:
        keep: list[QueuedRun] = []
        for q in _pending:
            if ((mset is None or q.profile_name in mset)
                    and (pset is None or q.paper_id in pset)
                    and (dset is None or q.mode in dset)):
                removed += 1
            else:
                keep.append(q)
        _pending[:] = keep
    return removed


def snapshot() -> dict:
    """Dashboard view: what's pending, running, and most-recently done."""
    with _lock:
        return {
            "pending": [vars(q) for q in _pending],
            "running": [vars(q) for q in _running.values()],
            "completed": [vars(r) for r in _completed[-50:]],
        }


def wait_all(poll_interval: float = 5.0) -> None:
    """Block until both pending and running queues are empty.

    ``bench()`` is non-blocking, so blocking callers (the ``bench`` CLI
    command) poll here until the queue has drained.
    """
    while True:
        with _lock:
            if not _pending and not _running:
                return
        time.sleep(poll_interval)


__all__ = ["bench", "snapshot", "wait_all", "remove_pending", "QueuedRun"]
