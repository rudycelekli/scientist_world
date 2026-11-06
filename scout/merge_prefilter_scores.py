#!/usr/bin/env python3
"""Merge prefilter scores from JSONL into paper_index.json.

Reads arxiv_prefilter/scores.jsonl and adds a "prefilter_score" field
to each matching entry in paper_index.json. Deduplicates by taking the
latest score (last entry in JSONL wins).

The field is an annotation of the index only: select_for_analysis.py reads
scores.jsonl directly, and no later stage depends on this step.

Usage:
    python merge_prefilter_scores.py --data-dir /path/to/data
    python merge_prefilter_scores.py --data-dir /path/to/data --dry-run
"""

import argparse
import json
import os
from pathlib import Path


def resolve_data_dir(cli_value) -> Path:
    """Return the data directory: --data-dir, else $SCOUT_DATA_DIR, else ./data."""
    return Path(cli_value or os.environ.get("SCOUT_DATA_DIR") or "data").resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Merge prefilter scores into paper_index.json")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Data directory (default: $SCOUT_DATA_DIR, else ./data)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    index_path = data_dir / "paper_index.json"
    scores_path = data_dir / "arxiv_prefilter" / "scores.jsonl"

    if not index_path.exists():
        print(f"Error: {index_path} not found")
        return
    if not scores_path.exists():
        print(f"Error: {scores_path} not found")
        return

    # Load scores (last entry wins for dedup)
    scores = {}
    for line in scores_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            scores[entry["id"]] = entry["score"]
        except (json.JSONDecodeError, KeyError):
            continue

    print(f"Scores loaded: {len(scores)} (from {scores_path.name})")

    # Load index
    index = json.loads(index_path.read_text())
    print(f"Papers in index: {len(index)}")

    # Merge
    added = updated = unchanged = missing = 0
    for aid, score in scores.items():
        if aid not in index:
            missing += 1
            continue
        old = index[aid].get("prefilter_score")
        if old == score:
            unchanged += 1
        elif old is None:
            added += 1
            index[aid]["prefilter_score"] = score
        else:
            updated += 1
            index[aid]["prefilter_score"] = score

    print(f"  Added: {added}  |  Updated: {updated}  |  "
          f"Unchanged: {unchanged}  |  Not in index: {missing}")

    if args.dry_run:
        print("\nDry run — no changes written.")
        return

    if added == 0 and updated == 0:
        print("\nNothing to write.")
        return

    # Write to a temporary file and rename, so an interrupted run cannot
    # leave a truncated index behind.
    tmp = index_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2))
    tmp.rename(index_path)
    print(f"\nWritten to {index_path}")


if __name__ == "__main__":
    main()
