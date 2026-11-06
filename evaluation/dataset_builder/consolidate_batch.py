"""Batch-consolidate every multi-version run's final idea (see consolidate.py).

Resumable + parallel. Reads out/scientist_world_traces.parquet (written by
build_dataset) and writes one JSON line per run to out/consolidated.jsonl
keyed by run_id; re-running skips run_ids already present. Single-version
runs need no call and are not listed. Run build_dataset again afterwards: it
reads this checkpoint to fill the final_idea column.

    python -m dataset_builder.consolidate_batch [--workers 6] [--limit N]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset

from dataset_builder.consolidate import consolidate

_OUT = Path(__file__).resolve().parent / "out"
_PARQUET = _OUT / "scientist_world_traces.parquet"
_CKPT = _OUT / "consolidated.jsonl"


def _load_done() -> set[str]:
    done = set()
    if _CKPT.exists():
        for ln in _CKPT.read_text().splitlines():
            try:
                done.add(json.loads(ln)["run_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _one(row) -> dict:
    inputs = {k: row[k] for k in ("problem", "direction", "constraint", "target")}
    text = consolidate(inputs, list(row["idea_versions"]))
    return {"run_id": row["run_id"], "final_idea_consolidated": text,
            "n_idea_versions": row["n_idea_versions"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset("parquet", data_files=str(_PARQUET), split="train")
    done = _load_done()
    todo = [r for r in ds
            if r["run_id"] and r["run_id"] not in done and r["n_idea_versions"] >= 2]
    if args.limit:
        todo = todo[: args.limit]
    print(f"total runs: {len(ds)} | already done: {len(done)} | "
          f"to consolidate (multi-version): {len(todo)} | workers: {args.workers}",
          flush=True)

    n_ok = n_fail = 0
    with _CKPT.open("a") as ck, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, r): r["run_id"] for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"run_id": futs[fut], "final_idea_consolidated": None,
                       "error": str(e)[:120]}
            ck.write(json.dumps(rec) + "\n")
            ck.flush()
            if rec.get("final_idea_consolidated"):
                n_ok += 1
            else:
                n_fail += 1
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}  ok={n_ok} fail={n_fail}", flush=True)
    print(f"DONE: ok={n_ok} fail={n_fail} | checkpoint={_CKPT}", flush=True)


if __name__ == "__main__":
    main()
