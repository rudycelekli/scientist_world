"""Submit analyzed papers to Swarm for reproduction.

Submits papers to the Swarm queue with GitHub topics derived from their
taxonomy. The papers come from an ordered ID list (the selection snapshot),
from explicit IDs, or from one date's arXiv listing filtered by overall
reproduction difficulty score. Without --submit nothing is sent.

Usage:
    # Dry run: show the first 300 papers of a selection snapshot
    python submit_to_swarm.py --ids-json top1500.json --count 300

    # Submit them (papers Swarm already knows are skipped)
    python submit_to_swarm.py --ids-json top1500.json --count 300 --submit

    # Submit specific papers by ID (still adds taxonomy topics)
    python submit_to_swarm.py --ids 2601.16175 2603.01025 --submit

    # One date's listing, filtered by overall reproduction score
    python submit_to_swarm.py --date 2025-12-01 --category cs.CV --min-score 2 --max-score 3 --count 10 --submit
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


# Swarm's default dashboard address (dashboard_port in the Swarm config).
DEFAULT_SWARM_URL = "http://localhost:39100"
CATEGORY_MERGE_MAP_PATH = Path(__file__).with_name("category_merge_map.json")


def resolve_data_dir(cli_value) -> Path:
    """Return the data directory: --data-dir, else $SCOUT_DATA_DIR, else ./data."""
    return Path(cli_value or os.environ.get("SCOUT_DATA_DIR") or "data").resolve()


def _slugify(text: str) -> str:
    """Convert text to a GitHub-safe topic slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:35] if slug else ""


def _load_category_merge_spec() -> tuple[dict[str, str], list[tuple[re.Pattern[str], str]], str]:
    if not CATEGORY_MERGE_MAP_PATH.exists():
        return {}, [], "uncategorized"

    try:
        payload = json.loads(CATEGORY_MERGE_MAP_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}, [], "uncategorized"

    fallback = payload.get("fallback_category", "uncategorized")
    groups = payload.get("canonical_groups", {})
    aliases = {}
    if isinstance(groups, dict):
        for canonical, values in groups.items():
            if not isinstance(canonical, str) or not canonical.strip():
                continue
            canonical = canonical.strip()
            aliases[canonical] = canonical
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str) and value.strip():
                    aliases[value.strip()] = canonical

    compiled_rules = []
    for rule in payload.get("keyword_rules", []):
        if not isinstance(rule, dict):
            continue
        bucket = rule.get("bucket")
        patterns = rule.get("patterns", [])
        if not isinstance(bucket, str) or not bucket.strip() or not isinstance(patterns, list):
            continue
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            try:
                compiled_rules.append((re.compile(pattern.strip()), bucket.strip()))
            except re.error:
                continue

    return aliases, compiled_rules, fallback


CATEGORY_MERGE_ALIASES, CATEGORY_MERGE_RULES, CATEGORY_MERGE_FALLBACK = (
    _load_category_merge_spec()
)


def _canonicalize_selection_category(value: str) -> str:
    exact = CATEGORY_MERGE_ALIASES.get(value)
    if exact:
        return exact

    lower = value.lower()
    for pattern, bucket in CATEGORY_MERGE_RULES:
        if pattern.search(lower):
            return bucket

    return CATEGORY_MERGE_FALLBACK or value


def _selection_category(taxonomy: dict) -> str:
    primary = taxonomy.get("primary", {}) if isinstance(taxonomy, dict) else {}
    for key in ("category", "area", "method", "domain"):
        value = primary.get(key)
        if isinstance(value, str) and value.strip():
            return _canonicalize_selection_category(value.strip())
    return "uncategorized"


def _topics_from_taxonomy(taxonomy: dict, selection_category: Optional[str] = None) -> list[str]:
    """Extract GitHub topics from the paper's taxonomy section."""
    topics = []
    primary = taxonomy.get("primary", {})
    for key in ("domain", "area", "category", "method"):
        val = primary.get(key)
        if val:
            slug = _slugify(val)
            if slug:
                topics.append(f"{key}-{slug}")

    for sec in taxonomy.get("secondary", []):
        for key in ("domain", "area", "category", "method"):
            val = sec.get(key)
            if val:
                slug = _slugify(val)
                if slug:
                    topics.append(f"secondary-{key}-{slug}")

    if selection_category:
        slug = _slugify(selection_category)
        if slug:
            topics.append(f"selection-category-{slug}")

    return topics


def _get_swarm_papers(swarm_url: str) -> set[str]:
    """Fetch already-submitted paper IDs from the Swarm dashboard.

    Exits with an error if Swarm cannot be reached.
    """
    url = f"{swarm_url.rstrip('/')}/api/papers"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            papers = json.loads(resp.read().decode())
        return {p["arxiv_id"] for p in papers}
    except Exception as exc:
        sys.exit(f"Error: cannot list the papers known to Swarm at {url}: {exc}")


def _submit_to_swarm(arxiv_id: str, date: str, topics: list[str],
                     swarm_url: str, workflow: Optional[str] = None) -> tuple[bool, str]:
    """Submit a paper to Swarm. Returns (success, message)."""
    payload = {
        "arxiv_id": arxiv_id,
        "date": date,
        "topics": topics,
    }
    if workflow:
        payload["workflow"] = workflow
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{swarm_url.rstrip('/')}/api/papers",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
        return True, result.get("status", "ok")
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
            return False, err.get("error", f"HTTP {e.code}")
        except Exception:
            return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


def _load_ids_json(path: Path) -> list[str]:
    """Load ordered arXiv IDs from a JSON snapshot file."""
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        ids = payload
    elif isinstance(payload, dict):
        for key in ("ids", "arxiv_ids", "selected_ids"):
            value = payload.get(key)
            if isinstance(value, list):
                ids = value
                break
        else:
            raise ValueError("JSON must be a list of IDs or contain ids/arxiv_ids/selected_ids")
    else:
        raise ValueError("JSON must be a list or object")

    extracted = []
    for item in ids:
        if isinstance(item, str) and item.strip():
            extracted.append(item.strip())
            continue
        if isinstance(item, dict):
            value = item.get("arxiv_id") or item.get("id")
            if isinstance(value, str) and value.strip():
                extracted.append(value.strip())
                continue
        raise ValueError("JSON entries must be ID strings or objects with id/arxiv_id")
    return extracted


def main():
    parser = argparse.ArgumentParser(
        description="Submit analyzed papers to Swarm for reproduction")
    parser.add_argument("--date", default=None,
                        help="Listing date (YYYY-MM-DD). Required unless --ids or "
                             "--ids-json is used.")
    parser.add_argument("--ids", "--arxiv-id", nargs="+", default=None,
                        help="Specific arXiv IDs to submit (skips date/score filtering)")
    parser.add_argument("--ids-json", type=Path, default=None,
                        help="JSON file containing an ordered list of arXiv IDs to submit; "
                             "the first --count IDs are used")
    parser.add_argument("--category", default=None,
                        help="arXiv category (e.g. cs.AI, cs.CV). "
                             "If omitted, uses all categories for the date.")
    parser.add_argument("--min-score", type=int, default=1,
                        help="Minimum overall reproduction score (inclusive, default 1)")
    parser.add_argument("--max-score", type=int, default=5,
                        help="Maximum overall reproduction score (inclusive, default 5)")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of papers to submit (default 10). With --ids-json: "
                             "length of the list prefix to consider. Ignored with --ids.")
    parser.add_argument("--paper-type", default=None,
                        help="Filter by paper type (e.g. metric_method, "
                             "benchmark_dataset, analysis_findings, survey_position)")
    parser.add_argument("--no-cost", action="store_true",
                        help="Date mode only: keep only papers whose analysis says no "
                             "paid resources are required (paid_resources.required is false)")
    parser.add_argument("--submit", action="store_true",
                        help="Actually submit (without this flag, dry run only)")
    parser.add_argument("--workflow", type=str, default="standard",
                        help="Workflow name (default: standard)")
    parser.add_argument("--swarm-url", default=DEFAULT_SWARM_URL,
                        help=f"Swarm dashboard URL (default: {DEFAULT_SWARM_URL})")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory (default: $SCOUT_DATA_DIR, else ./data)")
    parser.add_argument("--analysis-dir", type=Path, default=None,
                        help="Analysis directory (default: <data-dir>/arxiv_analysis)")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)

    # Load paper index for titles
    paper_index = {}
    paper_index_path = data_dir / "paper_index.json"
    if paper_index_path.exists():
        try:
            paper_index = json.loads(paper_index_path.read_text())
        except Exception:
            pass

    analysis_dir = args.analysis_dir or data_dir / "arxiv_analysis"

    direct_ids = None
    if args.ids_json:
        try:
            direct_ids = _load_ids_json(args.ids_json)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Error: failed to load --ids-json {args.ids_json}: {exc}")
            sys.exit(1)
        direct_ids = direct_ids[:args.count]
    elif args.ids:
        direct_ids = list(args.ids)

    # --ids / --ids-json mode: submit specific papers, skip date/score filtering
    if direct_ids:
        candidates = []
        for arxiv_id in direct_ids:
            topics = []
            score = None
            result_path = analysis_dir / arxiv_id / "result.json"
            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text())
                    taxonomy = result.get("taxonomy", {})
                    selection_category = _selection_category(taxonomy)
                    topics = _topics_from_taxonomy(taxonomy, selection_category)
                    paper_type = result.get("paper_type", {}).get("type", "")
                    if paper_type:
                        topics.insert(0, f"paper-type-{_slugify(paper_type)}")
                    score = result.get("reproduction_scores", {}).get("overall", {}).get("score")
                except Exception:
                    pass
            title = paper_index.get(arxiv_id, {}).get("title", "")
            candidates.append({
                "arxiv_id": arxiv_id,
                "score": score,
                "topics": topics,
                "title": title,
            })
        source = f"from {args.ids_json}" if args.ids_json else "by explicit IDs"
        print(f"Direct submit: {len(candidates)} papers ({source})")

    else:
        # Date-based mode: collect from listings and filter
        if not args.date:
            print("Error: --date is required unless --ids or --ids-json is used")
            sys.exit(1)

        listing_dir = data_dir / "arxiv_listings" / args.date
        if not listing_dir.is_dir():
            print(f"Error: no listings found for date {args.date} at {listing_dir}")
            sys.exit(1)

        paper_ids = set()
        if args.category:
            listing_file = listing_dir / f"{args.category}.json"
            if not listing_file.exists():
                print(f"Error: no listing for {args.category} on {args.date}")
                sys.exit(1)
            paper_ids.update(json.loads(listing_file.read_text()))
        else:
            for f in listing_dir.glob("*.json"):
                paper_ids.update(json.loads(f.read_text()))

        print(f"Date {args.date}: {len(paper_ids)} papers in listings"
              + (f" ({args.category})" if args.category else " (all categories)"))

        candidates = []
        skipped_no_analysis = 0
        skipped_score = 0
        skipped_type = 0
        skipped_cost = 0

        for arxiv_id in paper_ids:
            result_path = analysis_dir / arxiv_id / "result.json"
            # result.json is copied from the template when an analysis starts;
            # .done marks one that finished.
            if not result_path.exists() or not (analysis_dir / arxiv_id / ".done").exists():
                skipped_no_analysis += 1
                continue

            try:
                result = json.loads(result_path.read_text())
            except (json.JSONDecodeError, OSError):
                skipped_no_analysis += 1
                continue

            scores = result.get("reproduction_scores", {})
            overall = scores.get("overall", {})
            score = overall.get("score")
            if score is None:
                skipped_no_analysis += 1
                continue

            if not (args.min_score <= score <= args.max_score):
                skipped_score += 1
                continue

            # Filter by paper type
            paper_type = result.get("paper_type", {}).get("type", "")
            if args.paper_type and paper_type != args.paper_type:
                skipped_type += 1
                continue

            # Filter by paid resources (API fees, paid data, licences);
            # compute is not counted as a paid resource.
            paid = result.get("paid_resources")
            if args.no_cost and isinstance(paid, dict) and paid.get("required"):
                skipped_cost += 1
                continue

            taxonomy = result.get("taxonomy", {})
            selection_category = _selection_category(taxonomy)
            topics = _topics_from_taxonomy(taxonomy, selection_category)
            if paper_type:
                topics.insert(0, f"paper-type-{_slugify(paper_type)}")
            if args.no_cost:
                topics.append("no-cost")
            title = paper_index.get(arxiv_id, {}).get("title", "")

            candidates.append({
                "arxiv_id": arxiv_id,
                "score": score,
                "topics": topics,
                "title": title,
                "taxonomy": taxonomy,
            })

        # Sort by score ascending (easiest first)
        candidates.sort(key=lambda c: (c["score"], c["arxiv_id"]))

        type_msg = f" | Skipped (wrong type): {skipped_type}" if args.paper_type else ""
        cost_msg = f" | Skipped (has cost): {skipped_cost}" if args.no_cost else ""
        total_analyzed = len(candidates) + skipped_score + skipped_type + skipped_cost
        print(f"  Analyzed: {total_analyzed} | "
              f"In score range [{args.min_score}-{args.max_score}]: {total_analyzed - skipped_score} | "
              f"No analysis: {skipped_no_analysis}{type_msg}{cost_msg} | "
              f"Final: {len(candidates)}")

    # Check for duplicates already in swarm
    already_submitted = set()
    if args.submit:
        already_submitted = _get_swarm_papers(args.swarm_url)
        if already_submitted:
            print(f"  Already in swarm: {len(already_submitted)} papers")

    # Select up to --count papers, skipping duplicates
    selected = []
    for c in candidates:
        if c["arxiv_id"] in already_submitted:
            continue
        selected.append(c)
        if not direct_ids and len(selected) >= args.count:
            break

    if not selected:
        print("\nNo papers to submit.")
        sys.exit(0)

    # Display
    label = f"score {args.min_score}-{args.max_score}" if args.date else "by ID"
    print(f"\n{'SUBMITTING' if args.submit else 'DRY RUN'}: "
          f"{len(selected)} papers ({label})\n")
    print(f"{'ID':<15} {'Score':>5}  {'Topics'}")
    print("-" * 80)
    for c in selected:
        topic_str = ", ".join(c["topics"][:6])
        if len(c["topics"]) > 6:
            topic_str += f" (+{len(c['topics'])-6} more)"
        title_short = c["title"][:40] + "..." if len(c["title"]) > 40 else c["title"]
        score_str = f"{c['score']:>5}" if c["score"] is not None else f"{'—':>5}"
        print(f"{c['arxiv_id']:<15} {score_str}  {topic_str}")
        if title_short:
            print(f"{'':>22}{title_short}")

    if not args.submit:
        print(f"\nDry run complete. Add --submit to actually submit.")
        return

    # Submit
    print()
    submitted = 0
    failed = 0
    for c in selected:
        ok, msg = _submit_to_swarm(c["arxiv_id"], args.date or "", c["topics"],
                                   args.swarm_url, args.workflow)
        if ok:
            print(f"  Submitted {c['arxiv_id']} ({msg})")
            submitted += 1
        else:
            print(f"  FAILED   {c['arxiv_id']}: {msg}")
            failed += 1

    print(f"\nDone: {submitted} submitted, {failed} failed.")


if __name__ == "__main__":
    main()
