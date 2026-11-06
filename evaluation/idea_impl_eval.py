"""Idea-implementation ablation, separate from the dashboard and its queue.

Takes the ideas produced by the three training stages (base / SFT / GRPO)
on the held-out papers and has one implementer profile implement each of
them in the normal innovate environment, with the idea fixed: the agent
is told to follow the given idea exactly and find the best method it can
under that framing. The scores then show how the training stage of the
*idea author* changes the downstream implemented result.

This drives ``evaluation.runner.run_one`` directly with its
``injected_idea`` and ``variant`` arguments. It does not use the
orchestrator queue, and variant runs are left out of normalization,
publishing and independent grading, so a cell gets train scores only.
Runs are namespaced ``<profile>-innovate-impl-<stage>-eval`` so they
never collide with the main evaluation or with each other.

    # from the evaluation/ directory, with the evaluation config in place:
    python idea_impl_eval.py --profile <implementer-profile> --smoke
    python idea_impl_eval.py --profile <implementer-profile> --concurrency 3

Idea sources, one idea per held-out paper and stage, written by
``training/baseline_untrained.py`` (override the directory with
``--ideas-dir``):

    base -> ../training/out/baseline_untrained.jsonl
    sft  -> ../training/out/sft_trained_gen.jsonl
    grpo -> ../training/out/grpo_trained_gen.jsonl

Each line is a JSON object with ``task_id`` (the paper id) and ``idea``.
Results go to ``<data_dir>/idea_impl_results`` unless ``--out-dir`` is
given: one ``<paper>_<stage>.json`` per cell plus ``SUMMARY.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluation import config as _config
from evaluation.runner import (
    run_one, _is_preexisting_completed, _has_substantial_work,
)

_HERE = Path(__file__).resolve().parent
_DEFAULT_IDEAS_DIR = _HERE.parent / "training" / "out"
_STAGE_FILES = {
    "base": "baseline_untrained.jsonl",
    "sft": "sft_trained_gen.jsonl",
    "grpo": "grpo_trained_gen.jsonl",
}


def _load_ideas(stage: str, ideas_dir: Path) -> dict[str, str]:
    """paper_id -> idea text for one stage (the final idea, not the <think>)."""
    path = ideas_dir / _STAGE_FILES[stage]
    ideas: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        idea = (r.get("idea") or "").strip()
        if idea:
            ideas[r["task_id"]] = idea
    return ideas


def _worklist(stages: list[str], papers: list[str] | None,
              ideas_dir: Path = _DEFAULT_IDEAS_DIR):
    """(stage, paper, idea) triples, intersected across the requested stages."""
    per_stage = {s: _load_ideas(s, ideas_dir) for s in stages}
    common = set.intersection(*(set(v) for v in per_stage.values()))
    chosen = [p for p in sorted(common) if (papers is None or p in papers)]
    return [(s, p, per_stage[s][p]) for p in chosen for s in stages]


def _rel_env_dir(env_dir: Path | str) -> str:
    """Workspace path relative to ``data_dir`` (as recorded in result files)."""
    try:
        return Path(env_dir).resolve().relative_to(_config.data_dir().resolve()).as_posix()
    except ValueError:
        return Path(env_dir).name


def _run_cell(profile: str, stage: str, paper: str, idea: str, out_dir: Path):
    t0 = time.time()
    variant = f"-impl-{stage}"
    # Idempotent / resumable: skip a cell that's already completed or has
    # in-flight work on disk (e.g. the smoke cell still running), rather
    # than tripping run_one's "refusing to wipe" guard.
    env_dir = _config.data_dir() / "envs" / paper / f"{profile}-innovate{variant}-eval"
    if env_dir.exists() and (_is_preexisting_completed(env_dir)
                             or _has_substantial_work(env_dir)):
        out_dir.mkdir(parents=True, exist_ok=True)
        state = "completed" if _is_preexisting_completed(env_dir) else "in_flight"
        rec = {"paper_id": paper, "stage": stage, "profile": profile,
               "variant": variant, "status": f"skipped_{state}",
               "env_dir": _rel_env_dir(env_dir), "elapsed_s": 0.0}
        (out_dir / f"{paper}_{stage}.json").write_text(json.dumps(rec, indent=2, default=str))
        print(f"[{stage:4s}] {paper}: skipped ({state} on disk)", flush=True)
        return rec
    try:
        res = run_one(paper, profile, mode="innovate",
                      variant=variant, injected_idea=idea)
        rec = {
            "paper_id": paper, "stage": stage, "profile": profile,
            "variant": variant, "status": res.status,
            "train_score": res.train_score, "test_score": res.test_score,
            "env_dir": _rel_env_dir(res.env_dir) if res.env_dir else "",
            "error": res.error,
            "elapsed_s": round(time.time() - t0, 1),
        }
    except Exception as exc:  # never let one cell kill the batch
        rec = {"paper_id": paper, "stage": stage, "profile": profile,
               "variant": variant, "status": "driver_error",
               "error": str(exc)[:500], "elapsed_s": round(time.time() - t0, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{paper}_{stage}.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"[{stage:4s}] {paper}: {rec['status']}  ({rec['elapsed_s']}s)", flush=True)
    return rec


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", required=True,
                    help="implementer profile (a name from the evaluation config)")
    ap.add_argument("--stages", default="base,sft,grpo",
                    help="comma-separated subset of: " + ", ".join(_STAGE_FILES))
    ap.add_argument("--papers", default=None,
                    help="comma-separated paper ids (default: every paper "
                         "that has an idea in all requested stages)")
    ap.add_argument("--ideas-dir", default=str(_DEFAULT_IDEAS_DIR),
                    help="directory holding the per-stage idea files "
                         "(default: ../training/out)")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="how many implementation runs in parallel (they share the compute pool)")
    ap.add_argument("--out-dir", default=None,
                    help="where result files go (default: <data_dir>/idea_impl_results)")
    ap.add_argument("--smoke", action="store_true",
                    help="run only ONE cell (first paper; the grpo stage when requested) end-to-end")
    args = ap.parse_args(argv)
    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in args.stages if s not in _STAGE_FILES]
    if unknown or not args.stages:
        ap.error(f"--stages must name some of {sorted(_STAGE_FILES)}; got {unknown or 'nothing'}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    stages = args.stages
    papers = ([p.strip() for p in args.papers.split(",")] if args.papers else None)
    out_dir = (Path(args.out_dir) if args.out_dir
               else _config.data_dir() / "idea_impl_results")

    work = _worklist(stages, papers, Path(args.ideas_dir))
    if not work:
        raise SystemExit("no (stage, paper) cell to run: no paper has an idea "
                         "in every requested stage")
    if args.smoke:
        # one representative cell: first paper, prefer grpo stage
        first_paper = work[0][1]
        work = [w for w in work if w[1] == first_paper and w[0] == "grpo"][:1] or work[:1]

    print(f"idea-impl ablation | profile={args.profile} | stages={stages} | "
          f"{len(work)} cells | concurrency={args.concurrency} | out={out_dir}",
          flush=True)
    for s, p, idea in work:
        print(f"   queued: [{s}] {p}  (idea {len(idea)} chars)", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(_run_cell, args.profile, s, p, idea, out_dir)
                for s, p, idea in work]
        for f in as_completed(futs):
            results.append(f.result())

    # summary
    summary_path = out_dir / "SUMMARY.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    ok = sum(1 for r in results if r.get("status") == "completed")
    print(f"\nDONE: {ok}/{len(results)} completed. Summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
