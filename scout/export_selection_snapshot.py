"""Export the global selection: filter, rank and balance all analysed papers.

Walks every <analysis-dir>/<id>/ folder that holds a finished analysis
(result.json and .done), applies the selection filters, ranks the remaining
papers, balances the top --count across categories, and writes the ordered
list of selected arXiv IDs and a statistics file. The helper functions come
from select_papers.py.

The filter thresholds are constants in build_candidates() and equal the
defaults of select_papers.py:
    paper quality >= 5
    aggregated reproduction difficulty <= 6 (overall score weighted 5.0,
        each of the four sub-scores weighted 1.25)
    no reproduction sub-score above 5
    paper type metric_method
    no paid resources required
    problem novelty >= 5
Papers are ranked by the geometric mean of quality/10 and 1 - difficulty/10.

Usage:
    python export_selection_snapshot.py --data-dir /path/to/data \\
        --count 1500 --temperature 1.2 \\
        --ids-out top1500.json --stats-out top1500_stats.json
"""

import argparse
import json
import importlib.util
import os
from collections import Counter
from pathlib import Path


def resolve_data_dir(cli_value) -> Path:
    """Return the data directory: --data-dir, else $SCOUT_DATA_DIR, else ./data."""
    return Path(cli_value or os.environ.get("SCOUT_DATA_DIR") or "data").resolve()


def load_select_module(repo_dir: Path):
    spec = importlib.util.spec_from_file_location("select_papers", repo_dir / "select_papers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_candidates(sp, analysis_dir: Path, paper_index: dict, temperature: float, count: int):
    skip = Counter()
    candidates = []
    analyzed_done = 0

    for paper_dir in sorted(analysis_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        done_path = paper_dir / ".done"
        result_path = paper_dir / "result.json"
        if done_path.exists() and result_path.exists():
            analyzed_done += 1
        if (not done_path.exists()) or (not result_path.exists()):
            skip["no_analysis"] += 1
            continue
        try:
            result = json.loads(result_path.read_text())
        except Exception:
            skip["no_analysis"] += 1
            continue

        q_score = ((result.get("paper_quality") or {}).get("score"))
        if q_score is None:
            skip["no_analysis"] += 1
            continue
        if q_score < 5:
            skip["quality"] += 1
            continue

        scores = result.get("reproduction_scores") or {}
        overall_score = ((scores.get("overall") or {}).get("score"))
        if overall_score is None:
            skip["no_analysis"] += 1
            continue
        r_score = sp._aggregate_repro_score(scores, 5.0, 1.25)
        if r_score is None:
            skip["no_analysis"] += 1
            continue
        if r_score > 6:
            skip["repro"] += 1
            continue

        sub_values = []
        sub_dict = {}
        for key in sp.REPRO_SUBSCORES:
            value = ((scores.get(key) or {}).get("score"))
            if value is not None:
                sub_values.append(value)
                sub_dict[key] = value
        if any(v > 5 for v in sub_values):
            skip["subscore"] += 1
            continue
        if sum(1 for v in sub_values if v >= 7) > 2:
            skip["subscore"] += 1
            continue

        paper_type = ((result.get("paper_type") or {}).get("type") or "")
        if paper_type not in ("metric_method",):
            skip["type"] += 1
            continue

        if sp._requires_paid_resources(result):
            skip["cost"] += 1
            continue

        pn_score = ((result.get("problem_novelty") or {}).get("score"))
        if pn_score is not None and pn_score < 5:
            skip["problem_novelty"] += 1
            continue

        taxonomy = result.get("taxonomy") or {}
        topics = sp._topics_from_taxonomy(taxonomy)
        if paper_type:
            topics.insert(0, f"paper-type-{sp._slugify(paper_type)}")

        candidates.append(
            {
                "arxiv_id": paper_dir.name,
                "quality_score": q_score,
                "repro_score": r_score,
                "overall_repro_score": overall_score,
                "problem_novelty": pn_score,
                "sub_scores": sub_dict,
                "title": paper_index.get(paper_dir.name, {}).get("title", ""),
                "topics": topics,
                "selection_category": sp._selection_category(result),
            }
        )

    q_raw = [c["quality_score"] for c in candidates]
    r_raw = [c["repro_score"] for c in candidates]
    q_norm = sp.normalize_quality(q_raw)
    e_norm = sp.normalize_easiness(r_raw)
    combined = sp.rank_geometric(q_norm, e_norm, 0.5, 0.5)
    for i, candidate in enumerate(candidates):
        candidate["q_norm"] = q_norm[i]
        candidate["e_norm"] = e_norm[i]
        candidate["combined"] = combined[i]

    candidates.sort(key=lambda c: (-c["combined"], c["arxiv_id"]))
    selected, category_selected, category_pool_counts, category_raw_targets = (
        sp._select_with_category_temperature(candidates, count, temperature)
    )

    return {
        "analyzed_done": analyzed_done,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "category_count": len(category_pool_counts),
        "rows": [
            {
                "category": cat,
                "before": int(pool),
                "after": int(category_selected.get(cat, 0)),
            }
            for cat, pool in sorted(category_pool_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "skip": dict(skip),
        "selected": selected,
    }


def main():
    parser = argparse.ArgumentParser(description="Export a ranked Scout selection snapshot.")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory (default: $SCOUT_DATA_DIR, else ./data)")
    parser.add_argument("--analysis-dir", type=Path, default=None,
                        help="Directory of analysis results (default: <data-dir>/arxiv_analysis)")
    parser.add_argument("--paper-index", type=Path, default=None,
                        help="Paper index with titles; optional "
                             "(default: <data-dir>/paper_index.json)")
    parser.add_argument("--count", type=int, default=1500,
                        help="Number of papers to select (default: 1500)")
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="Temperature of the category balancing "
                             "(default: 1.2; <= 0 disables balancing)")
    parser.add_argument("--ids-out", type=Path, required=True,
                        help="Output JSON file: ordered list of the selected arXiv IDs")
    parser.add_argument("--stats-out", type=Path, required=True,
                        help="Output JSON file: funnel and per-category statistics")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    analysis_dir = args.analysis_dir or data_dir / "arxiv_analysis"
    paper_index_path = args.paper_index or data_dir / "paper_index.json"
    if not analysis_dir.is_dir():
        parser.error(f"analysis directory not found: {analysis_dir}")

    repo_dir = Path(__file__).resolve().parent
    sp = load_select_module(repo_dir)
    paper_index = json.loads(paper_index_path.read_text()) if paper_index_path.exists() else {}

    snapshot = build_candidates(sp, analysis_dir, paper_index, args.temperature, args.count)
    selected_ids = [c["arxiv_id"] for c in snapshot["selected"]]

    args.ids_out.write_text(json.dumps(selected_ids, indent=2) + "\n")

    stats = {
        "temperature": args.temperature,
        "analyzed_done": snapshot["analyzed_done"],
        "candidate_count": snapshot["candidate_count"],
        "selected_count": snapshot["selected_count"],
        "category_count": snapshot["category_count"],
        "rows": snapshot["rows"],
        "skip": snapshot["skip"],
        "top10_ids": selected_ids[:10],
        "bottom10_ids": selected_ids[-10:],
    }
    args.stats_out.write_text(json.dumps(stats, indent=2) + "\n")

    print(
        json.dumps(
            {
                "ids_out": str(args.ids_out),
                "stats_out": str(args.stats_out),
                "selected_count": len(selected_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
