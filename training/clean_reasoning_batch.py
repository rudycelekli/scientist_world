"""Batch clean-reasoning over multi-version runs. Resumable + parallel.

For every run with >= 2 idea versions, produce a clean forward reasoning trace
(references to run results removed) for the <think> block. Output ->
out/reasoning.jsonl keyed by run_id. Single-version runs are skipped (there is
no iteration to show; they stay final-idea-only in SFT). Re-running skips the
run_ids that already have a trace and retries the failed ones.

    python clean_reasoning_batch.py [--workers 6] [--limit N] [--data PARQUET]
        [--model gpt-5.5] [--effort medium]

Needs the `codex` CLI on PATH, logged in.
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from clean_reasoning import DEFAULT_EFFORT, DEFAULT_MODEL, clean_reasoning
from data_prep import _load_rows, add_data_argument, build_prompt

_OUT = Path(__file__).resolve().parent / "out"
_CKPT = _OUT / "reasoning.jsonl"


def _load_done() -> set:
    done = set()
    if _CKPT.exists():
        for ln in _CKPT.read_text().splitlines():
            try:
                d = json.loads(ln)
                if d.get("reasoning"):
                    done.add(d["run_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _one(row, model=None, effort=None) -> dict:
    txt = clean_reasoning(build_prompt(row), list(row["idea_versions"]),
                          model=model, effort=effort)
    rec = {"run_id": row["run_id"], "task_id": row["task_id"], "mode": row["mode"]}
    if txt:
        rec["reasoning"] = txt
    else:
        rec["error"] = "clean_failed"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model passed to `codex exec` (published: gpt-5.5; "
                         "default also settable with SW_CODEX_MODEL)")
    ap.add_argument("--effort", default=DEFAULT_EFFORT,
                    help="reasoning effort passed to `codex exec` (published: "
                         "medium; default also settable with SW_CODEX_EFFORT)")
    add_data_argument(ap)
    args = ap.parse_args()
    _OUT.mkdir(exist_ok=True)
    ds = _load_rows(args.data)
    done = _load_done()
    todo = [r for r in ds
            if r["final_idea"] and r["run_id"] and r["n_idea_versions"] >= 2
            and r["run_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"multi-version runs: {sum(1 for r in ds if r['n_idea_versions']>=2)} | "
          f"already cleaned: {len(done)} | to clean: {len(todo)} | "
          f"workers: {args.workers} | codex model: {args.model} "
          f"(effort {args.effort})", flush=True)
    if todo and shutil.which("codex") is None:
        raise SystemExit("the `codex` CLI was not found on PATH; install it "
                         "and log in before running this pass")
    n_ok = n_fail = 0
    with _CKPT.open("a") as ck, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, r, args.model, args.effort): r["run_id"]
                for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"run_id": futs[fut], "error": str(e)[:120]}
            ck.write(json.dumps(rec) + "\n")
            ck.flush()
            n_fail += 1 if rec.get("error") else 0
            n_ok += 0 if rec.get("error") else 1
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ok={n_ok} fail={n_fail}", flush=True)
    print(f"DONE: ok={n_ok} fail={n_fail} | checkpoint={_CKPT}", flush=True)


if __name__ == "__main__":
    main()
