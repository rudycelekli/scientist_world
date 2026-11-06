#!/usr/bin/env python3
"""Choose which prefiltered papers get the structured analysis.

Reads the two prefilter outputs
    <data-dir>/arxiv_prefilter/scores.jsonl        quality score, 0-100
    <data-dir>/arxiv_categories/categories.jsonl   prefilter paper type
and writes a JSON list of arXiv IDs for `analyze_papers.py --ids-file`.

The rule is explicit and deterministic. No filter is applied unless its
option is given:
  1. Every scored paper is ranked by score, highest first. Papers with the
     same score are ordered by arXiv ID.
  2. --top-fraction F keeps the first floor(F * N) of the N scored papers.
  3. --min-score S drops papers that score below S.
  4. --exclude-type T [T ...] drops papers whose prefilter type is one of T.
     Papers without a type label are kept.
The output keeps the ranking order. When a paper has several rows in a JSONL
file the last one wins, as in merge_prefilter_scores.py.

Usage:
    python select_for_analysis.py --data-dir /path/to/data \\
        --top-fraction 0.25 --min-score 75 --exclude-type theoretical \\
        --out /path/to/data/analysis_ids.json

    python select_for_analysis.py --self-test    # checks the rule on synthetic rows
"""

import argparse
import contextlib
import io
import json
import math
import os
import tempfile
from fractions import Fraction
from pathlib import Path

PAPER_TYPES = (
    "metric_method",
    "benchmark_dataset",
    "analysis_findings",
    "theoretical",
    "survey_position",
)


def resolve_data_dir(cli_value) -> Path:
    """Return the data directory: --data-dir, else $SCOUT_DATA_DIR, else ./data."""
    return Path(cli_value or os.environ.get("SCOUT_DATA_DIR") or "data").resolve()


def _read_jsonl(path: Path):
    """Yield the JSON objects of a JSONL file, skipping blank and malformed lines."""
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def load_scores(path: Path) -> dict[str, float]:
    """Latest quality score per arXiv ID from the quality prefilter output."""
    scores = {}
    for row in _read_jsonl(path):
        aid, score = row.get("id"), row.get("score")
        if not isinstance(aid, str) or isinstance(score, bool):
            continue
        if isinstance(score, (int, float)):
            scores[aid] = score
    return scores


def load_types(path: Path) -> dict[str, str]:
    """Latest valid paper type per arXiv ID from the type prefilter output."""
    types = {}
    for row in _read_jsonl(path):
        aid, paper_type = row.get("id"), row.get("paper_type")
        if isinstance(aid, str) and paper_type in PAPER_TYPES:
            types[aid] = paper_type
    return types


def select_ids(scores: dict, types: dict | None = None, *, top_fraction=None,
               min_score=None, exclude_types=()) -> tuple[list[str], dict]:
    """Apply the selection rule. Returns (ordered IDs, counts after each step)."""
    ranked = sorted(scores, key=lambda aid: (-scores[aid], aid))
    counts = {"scored": len(ranked)}

    if top_fraction is not None:
        # Exact decimal arithmetic, so that 0.29 of 100 papers is 29, not 28.
        fraction = (top_fraction if isinstance(top_fraction, Fraction)
                    else Fraction(str(top_fraction)))
        ranked = ranked[:math.floor(fraction * len(ranked))]
        counts["after_top_fraction"] = len(ranked)

    if min_score is not None:
        ranked = [aid for aid in ranked if scores[aid] >= min_score]
        counts["after_min_score"] = len(ranked)

    if exclude_types:
        types = types or {}
        excluded = set(exclude_types)
        counts["without_type_label"] = sum(1 for aid in ranked if aid not in types)
        ranked = [aid for aid in ranked if types.get(aid) not in excluded]
        counts["after_exclude_type"] = len(ranked)

    counts["selected"] = len(ranked)
    return ranked, counts


def _fail(message: str):
    raise SystemExit(f"self-test FAILED: {message}")


def self_test() -> None:
    """Check the selection rule end to end on synthetic prefilter rows."""
    score_rows = [
        {"id": "2501.00001", "score": 90},
        {"id": "2501.00002", "score": 80},
        {"id": "2501.00003", "score": 80},
        {"id": "2501.00004", "score": 75},
        {"id": "2501.00005", "score": 60},
        {"id": "2501.00006", "score": 40},
        {"id": "2501.00007", "score": 20},
        {"id": "2501.00008", "score": 10},
        {"id": "2501.00009"},                   # no score: ignored
        {"id": "2501.00008", "score": 95},      # later row wins
    ]
    type_rows = [
        {"id": "2501.00001", "paper_type": "metric_method"},
        {"id": "2501.00002", "paper_type": "metric_method"},
        {"id": "2501.00002", "paper_type": "theoretical"},      # later row wins
        {"id": "2501.00003", "paper_type": "metric_method"},
        {"id": "2501.00004", "paper_type": "benchmark_dataset"},
        {"id": "2501.00005", "paper_type": "not_a_type"},       # invalid label: ignored
        {"id": "2501.00006", "paper_type": "survey_position"},
    ]
    ranking = ["2501.00008", "2501.00001", "2501.00002", "2501.00003",
               "2501.00004", "2501.00005", "2501.00006", "2501.00007"]

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "arxiv_prefilter").mkdir()
        (data_dir / "arxiv_categories").mkdir()
        scores_path = data_dir / "arxiv_prefilter" / "scores.jsonl"
        types_path = data_dir / "arxiv_categories" / "categories.jsonl"
        scores_path.write_text(
            "\n".join(json.dumps(r) for r in score_rows[:4])
            + "\nthis line is not JSON\n\n"
            + "\n".join(json.dumps(r) for r in score_rows[4:]) + "\n")
        types_path.write_text("\n".join(json.dumps(r) for r in type_rows) + "\n")

        scores = load_scores(scores_path)
        types = load_types(types_path)
        if len(scores) != 8 or scores["2501.00008"] != 95:
            _fail(f"load_scores: {scores}")
        if types.get("2501.00002") != "theoretical" or "2501.00005" in types:
            _fail(f"load_types: {types}")

        cases = [
            ({}, ranking),
            ({"top_fraction": Fraction("0.5")}, ranking[:4]),
            ({"top_fraction": Fraction("0.3")}, ranking[:2]),
            ({"min_score": 75}, ranking[:5]),
            ({"exclude_types": ["theoretical"]},
             [aid for aid in ranking if aid != "2501.00002"]),
            ({"top_fraction": Fraction("0.75"), "min_score": 75,
              "exclude_types": ["theoretical"]},
             ["2501.00008", "2501.00001", "2501.00003", "2501.00004"]),
        ]
        for kwargs, expected in cases:
            got, _ = select_ids(scores, types, **kwargs)
            if got != expected:
                _fail(f"select_ids({kwargs}) returned {got}, expected {expected}")

        hundred = {f"2501.{i:05d}": i for i in range(100)}
        got, counts = select_ids(hundred, top_fraction=0.29)
        if len(got) != 29 or got[0] != "2501.00099" or counts["after_top_fraction"] != 29:
            _fail(f"top fraction 0.29 of 100 kept {len(got)} papers")

        out_path = data_dir / "out" / "analysis_ids.json"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--data-dir", str(data_dir), "--top-fraction", "0.75",
                  "--min-score", "75", "--exclude-type", "theoretical",
                  "--out", str(out_path)])
        written = json.loads(out_path.read_text())
        if written != cases[-1][1]:
            _fail(f"command line wrote {written}, expected {cases[-1][1]}")

    print("self-test passed")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Write the list of arXiv IDs to analyse from the prefilter outputs.")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory (default: $SCOUT_DATA_DIR, else ./data)")
    parser.add_argument("--top-fraction", type=Fraction, default=None, metavar="F",
                        help="Keep the top fraction F of all scored papers by quality "
                             "score, 0 < F <= 1 (e.g. 0.25). Default: keep all.")
    parser.add_argument("--min-score", type=float, default=None, metavar="S",
                        help="Drop papers whose quality score is below S. Default: no minimum.")
    parser.add_argument("--exclude-type", nargs="+", default=[], choices=PAPER_TYPES,
                        metavar="TYPE",
                        help="Drop papers whose prefilter type is one of these "
                             f"({', '.join(PAPER_TYPES)}). Papers without a type "
                             "label are kept. Default: exclude nothing.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON file: ordered list of arXiv IDs")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the built-in check on synthetic rows and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return

    if args.out is None:
        parser.error("--out is required")
    if args.top_fraction is not None and not (0 < args.top_fraction <= 1):
        parser.error("--top-fraction must satisfy 0 < F <= 1")

    data_dir = resolve_data_dir(args.data_dir)
    scores_path = data_dir / "arxiv_prefilter" / "scores.jsonl"
    types_path = data_dir / "arxiv_categories" / "categories.jsonl"

    if not scores_path.exists():
        parser.error(f"quality prefilter output not found: {scores_path} "
                     "(run prefilter_papers.py first)")
    scores = load_scores(scores_path)

    types = {}
    if args.exclude_type:
        if not types_path.exists():
            parser.error(f"type prefilter output not found: {types_path} "
                         "(run categorize_papers.py first, or drop --exclude-type)")
        types = load_types(types_path)

    ids, counts = select_ids(scores, types, top_fraction=args.top_fraction,
                             min_score=args.min_score,
                             exclude_types=args.exclude_type)

    print(f"Scored papers: {counts['scored']}")
    if "after_top_fraction" in counts:
        print(f"  Top fraction {float(args.top_fraction):g}: {counts['after_top_fraction']}")
    if "after_min_score" in counts:
        print(f"  Score >= {args.min_score:g}: {counts['after_min_score']}")
    if "after_exclude_type" in counts:
        print(f"  Type not in ({', '.join(args.exclude_type)}): "
              f"{counts['after_exclude_type']}  "
              f"(kept without a type label: {counts['without_type_label']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ids, indent=2) + "\n")
    print(f"Selected: {counts['selected']} papers. Saved to {args.out}")


if __name__ == "__main__":
    main()
