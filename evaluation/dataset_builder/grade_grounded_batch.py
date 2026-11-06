"""Batch outcome-grounded per-axis grading. Resumable. -> out/grades_grounded.jsonl

    python -m dataset_builder.grade_grounded_batch [--papers ID,ID] [--models NAME,NAME]
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
from dataset_builder.grade_axes_batch import completed_cells, split_arg
from dataset_builder.grade_grounded import grade_axis, gather, AXES

_OUT = Path(__file__).resolve().parent / "out"
_CKPT = _OUT / "grades_grounded.jsonl"


def _key(p, m, mdl, a):
    return f"{p}|{m}|{mdl}|{a}"


def _load_done():
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


_MATS = {}


def _one(paper, mode, model, axis, envs):
    rk = f"{paper}|{mode}|{model}"
    mats = _MATS.get(rk)
    if mats is None:
        mats = gather(paper, model, mode)
        _MATS[rk] = mats
    g = grade_axis(paper, model, mode, axis, mats=mats)
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
    print(f"grounded | done: {len(done)} | to grade: {len(todo)} | "
          f"workers: {args.workers}", flush=True)
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
            n_fail += 1 if rec.get("error") else 0
            n_ok += 0 if rec.get("error") else 1
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ok={n_ok} fail={n_fail}", flush=True)
    print(f"DONE: ok={n_ok} fail={n_fail} | checkpoint={_CKPT}", flush=True)


if __name__ == "__main__":
    main()
