"""Write the two training sets of the published configuration to JSONL.

    python export_data.py [--data PARQUET] [--out-dir DIR]

No Tinker access is needed. The sets are built by the same data_prep calls,
with the same arguments, as the published training commands (README.md):

  sft_train.jsonl   what `sft.py --exclude-heldout --val-frac 0` trains on, one
                    line per example: {conversation, run_id, task_id, mode,
                    has_reasoning, augmented, test_score}
  grpo_train.jsonl  what `grpo_offline.py --sources cross_model,generalization
                    --weights cross_model=1.0,generalization=0.5
                    --max-samples 1500` trains on, one line per sample:
                    {conversation, advantage, reward, source, group_id,
                    task_id, mode}

The files show what a given snapshot of the traces dataset selects, and can be
kept as a frozen copy. `sft.py --mixture sft_train.jsonl --val-frac 0` trains
on exactly the examples of the SFT file.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data_prep import (HELDOUT_PAPERS, add_data_argument, grpo_dataset,
                       sft_examples)

_OUT = Path(__file__).resolve().parent / "out"

# Data selection of the published GRPO run.
GRPO_SOURCES = ("cross_model", "generalization")
GRPO_WEIGHTS = {"cross_model": 1.0, "generalization": 0.5}
GRPO_MAX_SAMPLES = 1500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(_OUT),
                    help="directory for sft_train.jsonl and grpo_train.jsonl")
    add_data_argument(ap)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft = sft_examples(source=args.data, exclude_papers=HELDOUT_PAPERS,
                       n_single_backfill=None, include_rephrased=True)
    sp = out_dir / "sft_train.jsonl"
    with sp.open("w") as f:
        for e in sft:
            f.write(json.dumps({
                "conversation": e["conversation"],
                "run_id": e["run_id"],
                "task_id": e["task_id"],
                "mode": e["mode"],
                "has_reasoning": e["has_reasoning"],
                "augmented": e.get("augmented", False),
                "test_score": e["test_score"],
            }) + "\n")
    print(f"wrote {sp}  ({len(sft)} examples, "
          f"{len({e['task_id'] for e in sft})} papers | "
          f"with <think>: {sum(1 for e in sft if e['has_reasoning'])} | "
          f"reworded: {sum(1 for e in sft if e.get('augmented'))} | "
          f"by mode {dict(Counter(e['mode'] for e in sft))})")

    grpo = grpo_dataset(source=args.data, sources=GRPO_SOURCES,
                        weights=GRPO_WEIGHTS, max_samples=GRPO_MAX_SAMPLES)
    gp = out_dir / "grpo_train.jsonl"
    with gp.open("w") as f:
        for e in grpo:
            f.write(json.dumps({
                "conversation": e["conversation"],
                "advantage": e["advantage"],
                "reward": e["reward"],
                "source": e["source"],
                "group_id": e["group_id"],
                "task_id": e["task_id"],
                "mode": e["mode"],
            }) + "\n")
    print(f"wrote {gp}  ({len(grpo)} samples, "
          f"{len({e['task_id'] for e in grpo})} papers | "
          f"by source {dict(Counter(e['source'] for e in grpo))})")


if __name__ == "__main__":
    main()
