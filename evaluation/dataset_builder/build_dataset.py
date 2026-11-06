"""Build the scientist-world-traces dataset from grader-verified runs.

One row per run whose verdict is ``verified``: an independent grader re-ran
the run's frozen method, and the scores of the row are the grader's
measurements, not the numbers the agent filed. Columns are kept SEPARATE (inputs, trace, idea
versions, outputs, scores), not pre-combined, so training-sample assembly
can happen downstream.

Usage:
    python -m dataset_builder.build_dataset                 # all verified runs
    python -m dataset_builder.build_dataset --limit 20      # quick sample
    python -m dataset_builder.build_dataset --out /path/out.parquet

Reads the runs under ``<data_dir>/envs`` of the evaluation config and
writes a parquet (+ a small JSONL preview) under dataset_builder/out/.
Scores are FNI on the visible and the held-out split and their mean, plus
the unscaled values the training code uses; the measured score tables are
kept as JSON strings. Free-text columns pass through the same
secret redaction as published run branches.

Requires the ``datasets`` package.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation import config as _config
from evaluation import normalization as _norm
from evaluation.git_publish import sanitize_text

from dataset_builder import extract as _x
from dataset_builder.model_ids import combined_model_id, scaffold_for

MODES = ("innovate", "improve")


def _read_json(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


# Consolidated final-idea checkpoint (run_id -> self-contained final idea),
# produced by consolidate_batch. Loaded once.
_CONSOLIDATED_CKPT = Path(__file__).resolve().parent / "out" / "consolidated.jsonl"


def _load_consolidated() -> dict:
    out = {}
    if _CONSOLIDATED_CKPT.exists():
        for ln in _CONSOLIDATED_CKPT.read_text().splitlines():
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if d.get("run_id") and d.get("final_idea_consolidated"):
                out[d["run_id"]] = d["final_idea_consolidated"]
    return out


_CONSOLIDATED = _load_consolidated()


def _delta_norm(paper_dir: Path, paper_id: str, mode: str, phase: str,
                agent_score: dict | None) -> dict:
    coeffs = _read_json(
        paper_dir / "normalization" / f"coeffs_{mode}_{phase}.json")
    return _norm.normalize_delta(coeffs, agent_score)


def build_row(run_dir: Path) -> dict | None:
    # Only "verified" means the grader re-ran the frozen method and its
    # checks held. The verdict also decides that a run finished: the status
    # in .evaluation.json tracks liveness and can be left behind.
    _ver = _read_json(run_dir / "scoring" / "verification.json") or {}
    if _ver.get("status") != "verified":
        return None
    meta = _read_json(run_dir / ".evaluation.json") or {}
    paper_dir = run_dir.parent
    # Run directories are named <profile>-<mode>-eval.
    mode = meta.get("mode") or next(
        (m for m in MODES if run_dir.name.endswith(f"-{m}-eval")), None)
    if mode not in MODES:
        return None
    profile = meta.get("model") or run_dir.name[: -len(f"-{mode}-eval")]
    paper_id = meta.get("paper_id") or paper_dir.name
    source_dir = paper_dir / "source" / "scoring"
    source_brief = paper_dir / "source"

    scaffold = (meta.get("usage") or {}).get("scaffold") or scaffold_for(profile)

    # Inputs (leak-free; problem/direction/constraint/target/...)
    inputs = _x.extract_inputs(source_brief)
    # Per-paper metadata from source/manifest.json (title, taxonomy topics),
    # shared across all runs of the paper.
    pman = _x.extract_paper_manifest(source_brief)

    # Thinking trace + idea evolution + final output
    trace_segs, has_cot = _x.extract_thinking_trace(run_dir, scaffold)
    idea_versions, final_idea = _x.extract_ideas(run_dir)
    reflection = _x.extract_reflection(run_dir)

    # Scores: what the grader measured, normalized with the paper's
    # coefficient files.
    train_blob = _ver.get("scores_train")
    test_blob = _ver.get("scores_test")
    tr = _delta_norm(paper_dir, paper_id, mode, "train", train_blob)
    te = _delta_norm(paper_dir, paper_id, mode, "test", test_blob)
    coeffs_test = _read_json(
        paper_dir / "normalization" / f"coeffs_{mode}_test.json")
    locked = bool(coeffs_test and coeffs_test.get("locked"))

    # FNI is the paper score times 100; a run's FNI is the mean of the
    # splits that have one.
    fni_train, fni_test = (
        100.0 * s if isinstance(s, (int, float)) else None
        for s in (tr.get("delta_norm"), te.get("delta_norm")))
    _have = [s for s in (fni_train, fni_test) if s is not None]

    usage = meta.get("usage") or {}
    # Git branch URL for this run's published artifacts (per-paper repo,
    # branch = {mode}-{profile}). None when the run wasn't pushed.
    pub = meta.get("publish") or {}
    branch_url = None
    if pub.get("repo") and pub.get("branch"):
        branch_url = f"https://github.com/{pub['repo']}/tree/{pub['branch']}"
    row = {
        # identity
        "task_id": paper_id,
        "paper_title": pman["paper_title"],
        "paper_topics": pman["paper_topics"],
        "mode": mode,
        "model": combined_model_id(profile),  # canonical id + effort tier
        "scaffold": scaffold,
        "run_id": meta.get("run_id"),
        "branch_url": branch_url,
        # inputs (separate columns)
        "problem": inputs.get("problem"),
        "direction": inputs.get("direction"),
        "constraint": inputs.get("constraint"),
        "target": inputs.get("target"),
        "experiments": inputs.get("experiments"),
        "overview": inputs.get("overview"),
        "evaluation": inputs.get("evaluation"),
        # thinking trace + idea evolution
        "thinking_trace": "\n\n".join(trace_segs) if trace_segs else None,
        "thinking_segments": trace_segs,
        "has_real_cot": has_cot,
        "n_trace_segments": len(trace_segs),
        "idea": idea_versions[0] if idea_versions else None,  # v1: self-contained proposal
        "idea_versions": idea_versions,                       # full evolution chain (raw last = idea_versions[-1])
        "n_idea_versions": len(idea_versions),
        # THE final idea: self-contained. Consolidated by consolidate_batch
        # for multi-version runs (back-references resolved, nothing improved);
        # v1 as-is when single.
        "final_idea": (
            _CONSOLIDATED.get(meta.get("run_id"))
            or (idea_versions[0] if len(idea_versions) == 1 else None)
        ),
        # retrospective summary the agent wrote at the end — NOT the idea
        # itself; a report of what was submitted. Use as aux, not as target.
        "reflection": reflection,
        "has_reflection": reflection is not None,
        # FNI, the leaderboard scale (None if the paper has no coefficients)
        "fni": sum(_have) / len(_have) if _have else None,
        "fni_train": fni_train,
        "fni_test": fni_test,
        # the same scores unscaled (FNI / 100), as the training code reads them
        "train_score": tr.get("delta_norm"),
        "test_score": te.get("delta_norm"),
        "train_score_reason": tr.get("reason"),
        "test_score_reason": te.get("reason"),
        "score_locked": locked,
        "n_models_cohort": (coeffs_test or {}).get("n_models"),
        # the grader that verified the run
        "grader_profile": _ver.get("grader_profile"),
        # the score tables the grader measured (json strings)
        "train_score_raw": json.dumps(train_blob) if train_blob else None,
        "test_score_raw": json.dumps(test_blob) if test_blob else None,
        # run metadata
        # GPU-hours the run consumed, by the runner's own accounting (sum
        # over the run's jobs of time waited * GPUs requested). This is the
        # figure the GPU budget is enforced against.
        "gpu_hours": (round(float(_ghu), 4)
                      if isinstance((_ghu := meta.get("gpu_hours_used")),
                                    (int, float)) else None),
        "cost_usd": usage.get("cost_usd"),
        "turn_count": usage.get("turn_count") or meta.get("current_turn"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
    }
    return {k: _redact(v) for k, v in row.items()}


def _redact(value):
    """Apply the publish-time secret redaction to every string in a row.

    Traces, ideas and reflections are text the agent wrote and may quote
    a credential it saw (an API key in an error message, a token in a
    URL); score blobs carry free-text notes.
    """
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def iter_run_dirs(envs: Path):
    for paper_dir in sorted(envs.iterdir()):
        if not paper_dir.is_dir():
            continue
        for run_dir in sorted(paper_dir.iterdir()):
            if run_dir.is_dir() and run_dir.name.endswith("-eval"):
                yield run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="max rows (0 = all)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    envs = _config.envs_dir()
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "scientist_world_traces.parquet"

    rows = []
    scanned = 0
    for run_dir in iter_run_dirs(envs):
        scanned += 1
        try:
            row = build_row(run_dir)
        except Exception as e:  # never let one bad run kill the build
            print(f"  WARN {run_dir.name}: {e}", file=sys.stderr)
            row = None
        if row:
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break

    print(f"scanned {scanned} run dirs -> {len(rows)} verified rows")
    # score coverage stats
    n_train = sum(1 for r in rows if r["train_score"] is not None)
    n_test = sum(1 for r in rows if r["test_score"] is not None)
    n_refl = sum(1 for r in rows if r["has_reflection"])
    n_cot = sum(1 for r in rows if r["has_real_cot"])
    print(f"  train_score present: {n_train} | test_score present: {n_test}")
    print(f"  reflection present: {n_refl} | real-CoT trace: {n_cot}")
    from collections import Counter
    print(f"  by model: {dict(Counter(r['model'] for r in rows))}")

    # Write parquet via datasets (handles list/str columns cleanly).
    from datasets import Dataset
    ds = Dataset.from_list(rows)
    ds.to_parquet(str(out_path))
    print(f"wrote {out_path} ({len(ds)} rows)")
    # small JSONL preview
    preview = out_dir / "preview_20.jsonl"
    with preview.open("w") as f:
        for r in rows[:20]:
            f.write(json.dumps({k: (v if not isinstance(v, list) else f"[{len(v)} items]")
                                for k, v in r.items()})[:4000] + "\n")
    print(f"wrote preview {preview}")


if __name__ == "__main__":
    main()
