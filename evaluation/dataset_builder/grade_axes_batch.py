"""Batch per-axis grading: 5 isolated judge calls per run (see grade_axes.py).

Resumable + parallel. Writes one JSON line per (paper, mode, model, axis) to
out/grades_axes.jsonl; re-running skips already-graded (cell, axis) pairs and
grades only newly-completed runs.

    python -m dataset_builder.grade_axes_batch [--papers ID,ID] [--models NAME,NAME]
                                               [--workers 6] [--limit N]

Without --papers / --models every completed run under ``<data_dir>/envs`` is
graded. ``model`` is the run's profile name.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluation import config as _config
from evaluation.git_publish import iter_completed_runs
from dataset_builder.grade_axes import grade_axis, gather, AXES

MODES = ('innovate', 'improve')

_OUT = Path(__file__).resolve().parent / "out"
_CKPT = _OUT / "grades_axes.jsonl"


def split_arg(value: str | None) -> set[str] | None:
    """Comma-separated CLI value -> set, or None when the flag was not given."""
    if not value:
        return None
    return {s.strip() for s in value.split(",") if s.strip()}


def completed_cells(envs: Path, papers: set[str] | None = None,
                    models: set[str] | None = None) -> list[tuple[str, str, str]]:
    """(paper, mode, model) of every completed run under *envs*.

    *papers* / *models* restrict the list when given. Runs in a variant
    workspace (a directory other than ``<model>-<mode>-eval``) are
    skipped: the graders address a run by that directory name.
    """
    cells = []
    for env_dir, meta in iter_completed_runs(envs):
        paper, model, mode = meta.get("paper_id"), meta.get("model"), meta.get("mode")
        if not paper or not model or mode not in MODES:
            continue
        if env_dir.name != f"{model}-{mode}-eval" or env_dir.parent.name != paper:
            continue
        if papers is not None and paper not in papers:
            continue
        if models is not None and model not in models:
            continue
        cells.append((paper, mode, model))
    return cells


def _key(paper, mode, model, axis):
    return f"{paper}|{mode}|{model}|{axis}"


def _load_done() -> set:
    done = set()
    if _CKPT.exists():
        for ln in _CKPT.read_text().splitlines():
            try:
                d = json.loads(ln)
                if d.get("score") is not None:
                    done.add(_key(d["paper"], d["mode"], d["model"], d["axis"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# Cache gathered materials per run (5 axes share them).
_MATS: dict[str, dict] = {}


def _one(paper, mode, model, axis, envs):
    env_dir = envs / paper / f"{model}-{mode}-eval"
    rk = str(env_dir)
    mats = _MATS.get(rk)
    if mats is None:
        mats = gather(env_dir)
        _MATS[rk] = mats
    g = grade_axis(env_dir, mode, axis, mats=mats)
    rec = {"paper": paper, "mode": mode, "model": model, "axis": axis}
    if g:
        rec["score"] = g["score"]
        rec["reason"] = g.get("reason")
    else:
        rec["error"] = "grade_failed"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=None,
                    help="comma-separated paper ids (default: all)")
    ap.add_argument("--models", default=None,
                    help="comma-separated profile names (default: all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    envs = _config.envs_dir()
    _OUT.mkdir(exist_ok=True)
    done = _load_done()

    todo = []
    for paper, mode, model in completed_cells(
            envs, split_arg(args.papers), split_arg(args.models)):
        for axis in AXES:
            if _key(paper, mode, model, axis) not in done:
                todo.append((paper, mode, model, axis))
    if args.limit:
        todo = todo[: args.limit]
    print(f"axes: {len(AXES)} | (cell,axis) done: {len(done)} | to grade: "
          f"{len(todo)} | workers: {args.workers}", flush=True)

    n_ok = n_fail = 0
    with _CKPT.open("a") as ck, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, p, m, mdl, ax, envs): (p, m, mdl, ax)
                for p, m, mdl, ax in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            p, m, mdl, ax = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"paper": p, "mode": m, "model": mdl, "axis": ax,
                       "error": str(e)[:120]}
            ck.write(json.dumps(rec) + "\n")
            ck.flush()
            if rec.get("error"):
                n_fail += 1
            else:
                n_ok += 1
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ok={n_ok} fail={n_fail}", flush=True)
    print(f"DONE: ok={n_ok} fail={n_fail} | checkpoint={_CKPT}", flush=True)


if __name__ == "__main__":
    main()
