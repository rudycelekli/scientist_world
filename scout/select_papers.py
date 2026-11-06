"""Smart paper selection and ranking for Swarm submission.

Filters papers by quality score, reproduction difficulty, and subscore
constraints, then ranks them using fixed 0-10 score scaling that
balances paper quality against reproduction feasibility.

Ranking methods:
  weighted_sum   Linear combination: w_q * quality_norm + w_r * easiness_norm
  geometric      Cobb-Douglas utility (default): quality^w_q * easiness^w_r
                 Hyperbolic iso-utility curves naturally balance both dimensions.
                 A paper needs BOTH good quality AND feasible reproduction.
  pareto         Pareto front ranking with geometric tiebreak within each front.
                 Papers not dominated in both dimensions rank highest.

This command line works on the listings of one date. The selection over all
analysed papers is made by export_selection_snapshot.py, which reuses the
functions defined here.

Usage:
    # Dry run: show ranked papers with default filters
    python select_papers.py --date 2026-03-01

    # Custom filters
    python select_papers.py --date 2026-03-01 --min-quality 7 --max-repro 4

    # Submit top 10 to a running Swarm
    python select_papers.py --date 2026-03-01 --count 10 --submit \\
        --swarm-url http://localhost:39100

    # Different ranking method, heavier quality weight
    python select_papers.py --date 2026-03-01 --method weighted_sum --quality-weight 0.7
"""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Pattern

# Swarm's default dashboard address (dashboard_port in the Swarm config).
DEFAULT_SWARM_URL = "http://localhost:39100"
CATEGORY_MERGE_MAP_PATH = Path(__file__).with_name("category_merge_map.json")

REPRO_SUBSCORES = [
    "coding_difficulty",
    "compute_difficulty",
    "speed_difficulty",
    "data_model_difficulty",
]

SUB_SHORT = {
    "coding_difficulty": "Code",
    "compute_difficulty": "Compute",
    "speed_difficulty": "Speed",
    "data_model_difficulty": "Data",
}


def resolve_data_dir(cli_value) -> Path:
    """Return the data directory: --data-dir, else $SCOUT_DATA_DIR, else ./data."""
    return Path(cli_value or os.environ.get("SCOUT_DATA_DIR") or "data").resolve()


def _parse_paid_amount(raw) -> float:
    """Parse a dollar-like string into a float. Returns 0 on failure."""
    m = re.search(r"(?:\d[\d,]*(?:\.\d+)?|\.\d+)", str(raw or "$0"))
    return float(m.group().replace(",", "")) if m else 0.0


def _parse_paid_cost(result: dict) -> float:
    """Extract numeric paid cost from API + other fields. Compute is ignored."""
    paid = result.get("paid_resources", {})
    if isinstance(paid, dict) and (
        "api_cost_estimate" in paid or "other_cost_estimate" in paid
    ):
        return (
            _parse_paid_amount(paid.get("api_cost_estimate", "$0"))
            + _parse_paid_amount(paid.get("other_cost_estimate", "$0"))
        )

    # Backward compatibility with older analysis schema.
    hardware_cost = result.get("hardware_cost", {})
    cost = hardware_cost.get("total_estimated_cost", {})
    total = 0.0
    for key in ("api", "other"):
        raw = str(cost.get(key, "$0"))
        m = re.search(r"(?:\d[\d,]*(?:\.\d+)?|\.\d+)", raw)
        if m:
            total += float(m.group().replace(",", ""))
    return total


def _requires_paid_resources(result: dict) -> bool:
    """True if the paper requires real-money spend beyond cluster compute."""
    paid = result.get("paid_resources")
    if isinstance(paid, dict) and paid.get("required") is not None:
        return bool(paid.get("required"))

    # Backward compatibility with older analysis schema.
    hardware_cost = result.get("hardware_cost", {})
    legacy_paid = hardware_cost.get("paid_resources")
    if isinstance(legacy_paid, dict) and legacy_paid.get("required") is not None:
        return bool(legacy_paid.get("required"))
    return _parse_paid_cost(result) > 0


def _aggregate_repro_score(scores: dict, overall_weight: float, subscore_weight: float):
    """Weighted 0-10 reproduction score using overall + four subscores."""
    weighted = []

    overall = (scores.get("overall") or {}).get("score")
    if overall is not None:
        weighted.append((overall, overall_weight))

    for key in REPRO_SUBSCORES:
        value = (scores.get(key) or {}).get("score")
        if value is not None:
            weighted.append((value, subscore_weight))

    if not weighted:
        return None

    total_weight = sum(weight for _, weight in weighted)
    return sum(value * weight for value, weight in weighted) / total_weight


# ── Fixed 0-10 scaling ────────────────────────────────────────────────

def normalize_quality(values: list[int]) -> list[float]:
    """Map 0-10 quality scores to [0, 1] using fixed absolute scaling."""
    return [min(max(v, 0), 10) / 10.0 for v in values]


def normalize_easiness(values: list[int]) -> list[float]:
    """Map 0-10 reproduction difficulty to [0, 1] easiness."""
    return [1.0 - (min(max(v, 0), 10) / 10.0) for v in values]


# ── Ranking methods ────────────────────────────────────────────────────

def rank_weighted_sum(q_norm, e_norm, w_q, w_r):
    """Weighted sum: w_q * quality + w_r * easiness."""
    total = w_q + w_r
    wq, wr = w_q / total, w_r / total
    return [wq * q + wr * e for q, e in zip(q_norm, e_norm)]


def rank_geometric(q_norm, e_norm, w_q, w_r):
    """Cobb-Douglas utility: quality^alpha * easiness^beta.

    Produces hyperbolic iso-utility curves that naturally prefer papers
    strong in BOTH dimensions over papers extreme in only one.
    """
    total = w_q + w_r
    alpha, beta = w_q / total, w_r / total
    eps = 1e-6
    return [
        max(q, eps) ** alpha * max(e, eps) ** beta
        for q, e in zip(q_norm, e_norm)
    ]


def rank_pareto(q_norm, e_norm, w_q, w_r):
    """Pareto front ranking, then geometric tiebreak within each front.

    Papers on the first Pareto front (not dominated in both quality and
    easiness by any other paper) get the highest scores.
    """
    n = len(q_norm)
    if n == 0:
        return []

    remaining = set(range(n))
    fronts = []
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if (q_norm[j] >= q_norm[i] and e_norm[j] >= e_norm[i]
                        and (q_norm[j] > q_norm[i] or e_norm[j] > e_norm[i])):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        fronts.append(front)
        remaining -= set(front)

    geo = rank_geometric(q_norm, e_norm, w_q, w_r)
    scores = [0.0] * n
    for front_idx, front in enumerate(fronts):
        base = (len(fronts) - front_idx) * 10
        for i in front:
            scores[i] = base + geo[i]
    return scores


RANKERS = {
    "weighted_sum": rank_weighted_sum,
    "geometric": rank_geometric,
    "pareto": rank_pareto,
}


# ── Swarm helpers ──────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:35] if slug else ""


def _topics_from_taxonomy(taxonomy: dict) -> list[str]:
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
    return topics


def _selection_category(result: dict) -> str:
    """Category key used for tempered final-set balancing."""
    taxonomy = result.get("taxonomy", {})
    primary = taxonomy.get("primary", {}) if isinstance(taxonomy, dict) else {}
    for key in ("category", "area", "method", "domain"):
        value = primary.get(key)
        if isinstance(value, str) and value.strip():
            return _canonicalize_selection_category(value.strip())
    return "uncategorized"


def _load_category_merge_spec() -> tuple[dict[str, str], list[tuple[Pattern[str], str]], str]:
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


def _allocate_category_targets(candidates: list[dict], count: int, temperature: float):
    """Allocate top-N slots across categories with temperature-smoothed shares."""
    pool_counts = Counter(c["selection_category"] for c in candidates)
    if not pool_counts or count <= 0:
        return {}, pool_counts, {}

    if temperature <= 0:
        return {}, pool_counts, {}

    exponent = 1.0 / temperature
    weights = {cat: pool_counts[cat] ** exponent for cat in pool_counts}
    weight_total = sum(weights.values())
    raw_targets = {
        cat: count * weights[cat] / weight_total
        for cat in pool_counts
    }

    targets = {
        cat: min(pool_counts[cat], math.floor(raw_targets[cat]))
        for cat in pool_counts
    }

    remaining = count - sum(targets.values())
    order = sorted(
        pool_counts,
        key=lambda cat: (
            raw_targets[cat] - math.floor(raw_targets[cat]),
            weights[cat],
            pool_counts[cat],
            cat,
        ),
        reverse=True,
    )

    while remaining > 0:
        progressed = False
        for cat in order:
            if remaining <= 0:
                break
            if targets[cat] >= pool_counts[cat]:
                continue
            targets[cat] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    return targets, pool_counts, raw_targets


def _select_with_category_temperature(candidates: list[dict], count: int, temperature: float):
    """Select top-N with category balancing while preserving within-category order."""
    if count >= len(candidates) or temperature <= 0:
        return candidates[:count], {}, Counter(), {}

    buckets = defaultdict(list)
    for candidate in candidates:
        buckets[candidate["selection_category"]].append(candidate)

    targets, pool_counts, raw_targets = _allocate_category_targets(
        candidates, count, temperature
    )

    selected = []
    selected_ids = set()
    category_selected = Counter()

    for cat, quota in targets.items():
        for candidate in buckets[cat][:quota]:
            selected.append(candidate)
            selected_ids.add(candidate["arxiv_id"])
            category_selected[cat] += 1

    if len(selected) < count:
        for candidate in candidates:
            if candidate["arxiv_id"] in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate["arxiv_id"])
            category_selected[candidate["selection_category"]] += 1
            if len(selected) >= count:
                break

    selected.sort(key=lambda c: c["combined"], reverse=True)
    return selected, category_selected, pool_counts, raw_targets


def _get_swarm_papers(swarm_url: str) -> set[str]:
    """IDs of the papers Swarm already knows. Exits if Swarm cannot be reached."""
    url = f"{swarm_url.rstrip('/')}/api/papers"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            papers = json.loads(resp.read().decode())
        return {p["arxiv_id"] for p in papers}
    except Exception as exc:
        sys.exit(f"Error: cannot list the papers known to Swarm at {url}: {exc}")


def _submit_to_swarm(arxiv_id, date, topics, swarm_url):
    payload = {"arxiv_id": arxiv_id, "date": date, "topics": topics}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{swarm_url.rstrip('/')}/api/papers",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
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


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Smart paper selection and ranking for Swarm submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data selection
    parser.add_argument("--date", required=True, help="Listing date (YYYY-MM-DD)")
    parser.add_argument("--category", nargs="*", default=None,
                        help="arXiv categories (e.g. cs.AI cs.CV). All if omitted.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Data directory (default: $SCOUT_DATA_DIR, else ./data)")

    # Quality filters
    parser.add_argument("--min-quality", type=int, default=5,
                        help="Min quality score inclusive (default: 5)")
    parser.add_argument("--max-quality", type=int, default=None,
                        help="Max quality score inclusive (default: no limit)")

    # Reproduction overall score filters
    parser.add_argument("--min-repro", type=float, default=None,
                        help="Min aggregated reproduction score inclusive (default: no limit)")
    parser.add_argument("--max-repro", type=float, default=6,
                        help="Max aggregated reproduction score inclusive (default: 6)")

    # Subscore constraints
    parser.add_argument("--subscore-cap", type=int, default=5,
                        help="Reject if ANY subscore exceeds this (default: 5). "
                             "Set to 10 to disable.")
    parser.add_argument("--subscore-high-threshold", type=int, default=7,
                        help="Threshold for counting 'high' subscores (default: 7)")
    parser.add_argument("--subscore-max-high", type=int, default=2,
                        help="Max subscores allowed >= threshold (default: 2)")

    # Ranking
    parser.add_argument("--method", choices=list(RANKERS.keys()), default="geometric",
                        help="Ranking method (default: geometric)")
    parser.add_argument("--quality-weight", type=float, default=0.5,
                        help="Weight for quality dimension (default: 0.5)")
    parser.add_argument("--repro-weight", type=float, default=0.5,
                        help="Weight for reproduction easiness dimension (default: 0.5)")
    parser.add_argument("--overall-repro-weight", type=float, default=5.0,
                        help="Weight for overall reproduction score inside aggregated repro "
                             "(default: 5.0)")
    parser.add_argument("--subscore-repro-weight", type=float, default=1.25,
                        help="Weight for each reproduction subscore inside aggregated repro "
                             "(default: 1.25)")
    parser.add_argument("--category-temperature", type=float, default=1.2,
                        help="Temperature for category-balanced final selection "
                             "(default: 1.2, <=0 disables)")
    parser.add_argument("--exclude-category", nargs="*", default=None,
                        help="Canonical selection categories to exclude after "
                             "category consolidation (e.g. Theory Multimodal)")

    # Selection
    parser.add_argument("--count", type=int, default=10,
                        help="Number of papers to select (default: 10)")

    # Extra filters
    parser.add_argument("--paper-type", default="metric_method",
                        help="Filter by paper type (default: metric_method). "
                             "Pass empty string to disable.")
    parser.add_argument("--require-code", action="store_true",
                        help="Only include papers with code available")
    parser.add_argument("--min-problem-novelty", type=int, default=5,
                        help="Minimum problem novelty score (default: 5)")
    parser.add_argument("--allow-cost", action="store_true",
                        help="Also keep papers that require paid resources (API fees, "
                             "paid data, licences). By default they are rejected; "
                             "compute is never counted as a paid resource.")

    # Submission
    parser.add_argument("--submit", action="store_true",
                        help="Actually submit to Swarm (default: dry run)")
    parser.add_argument("--swarm-url", default=DEFAULT_SWARM_URL,
                        help=f"Swarm dashboard URL (default: {DEFAULT_SWARM_URL})")

    args = parser.parse_args()
    no_cost = not args.allow_cost
    data_dir = resolve_data_dir(args.data_dir)
    excluded_categories = {
        _canonicalize_selection_category(cat.strip())
        for cat in (args.exclude_category or [])
        if isinstance(cat, str) and cat.strip()
    }

    # ── Load paper IDs from listings ───────────────────────────────────
    listing_dir = data_dir / "arxiv_listings" / args.date
    if not listing_dir.is_dir():
        print(f"Error: no listings for {args.date} at {listing_dir}")
        sys.exit(1)

    paper_ids = set()
    if args.category:
        for cat in args.category:
            f = listing_dir / f"{cat}.json"
            if f.exists():
                paper_ids.update(json.loads(f.read_text()))
            else:
                print(f"Warning: no listing for {cat} on {args.date}")
    else:
        for f in listing_dir.glob("*.json"):
            paper_ids.update(json.loads(f.read_text()))

    cats = ", ".join(args.category) if args.category else "all"
    print(f"Date {args.date} ({cats}): {len(paper_ids)} papers in listings")

    # ── Load paper index ───────────────────────────────────────────────
    paper_index = {}
    idx_path = data_dir / "paper_index.json"
    if idx_path.exists():
        paper_index = json.loads(idx_path.read_text())

    # ── Load analyses and filter ───────────────────────────────────────
    analysis_dir = data_dir / "arxiv_analysis"
    skip = {"no_analysis": 0, "quality": 0, "repro": 0,
            "subscore": 0, "type": 0, "code": 0, "cost": 0,
            "problem_novelty": 0, "selection_category": 0}
    candidates = []

    for arxiv_id in sorted(paper_ids):
        result_path = analysis_dir / arxiv_id / "result.json"
        done_path = analysis_dir / arxiv_id / ".done"
        if not result_path.exists() or not done_path.exists():
            skip["no_analysis"] += 1
            continue

        try:
            result = json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            skip["no_analysis"] += 1
            continue

        # Quality score
        quality = result.get("paper_quality", {})
        q_score = quality.get("score")
        if q_score is None:
            skip["no_analysis"] += 1
            continue
        if q_score < args.min_quality:
            skip["quality"] += 1
            continue
        if args.max_quality is not None and q_score > args.max_quality:
            skip["quality"] += 1
            continue

        # Aggregated reproduction score
        scores = result.get("reproduction_scores", {})
        overall = scores.get("overall", {})
        overall_score = overall.get("score")
        if overall_score is None:
            skip["no_analysis"] += 1
            continue
        r_score = _aggregate_repro_score(
            scores,
            args.overall_repro_weight,
            args.subscore_repro_weight,
        )
        if r_score is None:
            skip["no_analysis"] += 1
            continue
        if args.min_repro is not None and r_score < args.min_repro:
            skip["repro"] += 1
            continue
        if args.max_repro is not None and r_score > args.max_repro:
            skip["repro"] += 1
            continue

        # Subscore constraints
        sub_values = []
        sub_dict = {}
        for key in REPRO_SUBSCORES:
            sv = scores.get(key, {}).get("score")
            if sv is not None:
                sub_values.append(sv)
                sub_dict[key] = sv

        if any(v > args.subscore_cap for v in sub_values):
            skip["subscore"] += 1
            continue

        high_count = sum(1 for v in sub_values
                         if v >= args.subscore_high_threshold)
        if high_count > args.subscore_max_high:
            skip["subscore"] += 1
            continue

        # Paper type filter
        paper_type = result.get("paper_type", {}).get("type", "")
        if args.paper_type and paper_type != args.paper_type:
            skip["type"] += 1
            continue

        # Code filter
        code_status = result.get("code_availability", {}).get("status", "")
        if args.require_code and "official" not in code_status and "community" not in code_status:
            skip["code"] += 1
            continue

        # Cost filter
        if no_cost and _requires_paid_resources(result):
            skip["cost"] += 1
            continue

        # Problem novelty filter
        pn_score = result.get("problem_novelty", {}).get("score")
        if pn_score is not None and args.min_problem_novelty is not None:
            if pn_score < args.min_problem_novelty:
                skip["problem_novelty"] += 1
                continue

        taxonomy = result.get("taxonomy", {})
        selection_category = _selection_category(result)
        if selection_category in excluded_categories:
            skip["selection_category"] += 1
            continue

        topics = _topics_from_taxonomy(taxonomy)
        if paper_type:
            topics.insert(0, f"paper-type-{_slugify(paper_type)}")

        candidates.append({
            "arxiv_id": arxiv_id,
            "quality_score": q_score,
            "repro_score": r_score,
            "overall_repro_score": overall_score,
            "problem_novelty": pn_score,
            "sub_scores": sub_dict,
            "title": paper_index.get(arxiv_id, {}).get("title", ""),
            "topics": topics,
            "selection_category": selection_category,
        })

    # Print filter summary
    analyzed = len(paper_ids) - skip["no_analysis"]
    print(f"  Analyzed: {analyzed} | No analysis: {skip['no_analysis']}")
    parts = [f"quality: {skip['quality']}", f"repro: {skip['repro']}",
             f"subscore: {skip['subscore']}"]
    if args.paper_type:
        parts.append(f"type: {skip['type']}")
    if args.require_code:
        parts.append(f"code: {skip['code']}")
    if no_cost:
        parts.append(f"cost: {skip['cost']}")
    if args.min_problem_novelty is not None:
        parts.append(f"problem_novelty: {skip['problem_novelty']}")
    if excluded_categories:
        parts.append(f"category: {skip['selection_category']}")
    print(f"  Filtered out — {', '.join(parts)}")
    print(f"  Candidates: {len(candidates)}")

    if not candidates:
        print("\nNo papers match the criteria.")
        sys.exit(0)

    # ── Fixed 0-10 scaling ────────────────────────────────────────────
    q_raw = [c["quality_score"] for c in candidates]
    r_raw = [c["repro_score"] for c in candidates]

    q_norm = normalize_quality(q_raw)
    e_norm = normalize_easiness(r_raw)

    # ── Rank ───────────────────────────────────────────────────────────
    rank_fn = RANKERS[args.method]
    combined = rank_fn(q_norm, e_norm, args.quality_weight, args.repro_weight)

    for i, c in enumerate(candidates):
        c["q_norm"] = q_norm[i]
        c["e_norm"] = e_norm[i]
        c["combined"] = combined[i]

    candidates.sort(key=lambda c: c["combined"], reverse=True)

    # ── Select top N, then deduplicate ────────────────────────────────
    # Pick top --count first, THEN remove already-submitted.
    # This way re-running with the same --count only retries failed ones,
    # not backfill with new papers.
    top_n, category_selected, category_pool_counts, category_raw_targets = (
        _select_with_category_temperature(
            candidates, args.count, args.category_temperature
        )
    )

    if args.category_temperature > 0 and category_pool_counts:
        print(f"  Category-balanced selection: temperature={args.category_temperature:.2f}")
        print("  Selected quota by category:")
        for cat, pool_count in sorted(
            category_pool_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            raw_target = category_raw_targets.get(cat, 0.0)
            actual = category_selected.get(cat, 0)
            print(f"    {cat}: pool={pool_count}, target≈{raw_target:.2f}, selected={actual}")

    already_submitted = set()
    if args.submit:
        already_submitted = _get_swarm_papers(args.swarm_url)
        if already_submitted:
            in_top = sum(1 for c in top_n if c["arxiv_id"] in already_submitted)
            print(f"  Already in swarm: {in_top}/{len(top_n)} of selected papers")

    to_submit = [c for c in top_n if c["arxiv_id"] not in already_submitted]
    # For dry run, show full ranked list
    display_list = candidates if not args.submit else to_submit

    if not display_list:
        print("\nAll selected papers are already in swarm.")
        sys.exit(0)

    # ── Display ────────────────────────────────────────────────────────
    if args.submit:
        print(f"\nSUBMITTING: {len(to_submit)} papers "
              f"({len(top_n) - len(to_submit)} already in swarm, "
              f"method={args.method}, fixed_scale=0-10, "
              f"w_q={args.quality_weight}, w_r={args.repro_weight})\n")
    else:
        print(f"\nDRY RUN: {len(candidates)} papers ranked "
              f"(method={args.method}, fixed_scale=0-10, "
              f"w_q={args.quality_weight}, w_r={args.repro_weight})\n")

    # Header
    sub_hdr = "  ".join(f"{SUB_SHORT[k]:>4}" for k in REPRO_SUBSCORES)
    score_w = 8 if args.method == "pareto" else 6
    print(f"{'#':<4} {'ID':<14} {'Qual':>4} {'Repr':>5} {'PN':>4}  "
          f"{sub_hdr}  {'Q_n':>5} {'E_n':>5} {'Score':>{score_w}}  Title")
    print("-" * (125 if args.method != "pareto" else 127))

    for rank, c in enumerate(display_list, 1):
        subs = c["sub_scores"]
        sub_str = "  ".join(
            f"{subs.get(k, '-'):>4}" if isinstance(subs.get(k), int)
            else f"{'—':>4}"
            for k in REPRO_SUBSCORES
        )
        title_short = (c["title"][:38] + "...") if len(c["title"]) > 38 else c["title"]
        if args.method == "pareto":
            score_str = f"{c['combined']:>8.2f}"
        else:
            score_str = f"{c['combined']:>6.4f}"
        pn_str = f"{c['problem_novelty']:>4}" if c.get("problem_novelty") is not None else f"{'—':>4}"
        repro_str = f"{c['repro_score']:>5.2f}"
        print(f"{rank:<4} {c['arxiv_id']:<14} {c['quality_score']:>4} {repro_str} {pn_str}  "
              f"{sub_str}  {c['q_norm']:>5.3f} {c['e_norm']:>5.3f} {score_str}  "
              f"{title_short}")

    if not args.submit:
        print(f"\nDry run. Add --submit to send to Swarm.")
        return

    # ── Submit ─────────────────────────────────────────────────────────
    print()
    submitted = failed = 0
    for c in to_submit:
        ok, msg = _submit_to_swarm(c["arxiv_id"], args.date, c["topics"],
                                   args.swarm_url)
        if ok:
            print(f"  Submitted {c['arxiv_id']} ({msg})")
            submitted += 1
        else:
            print(f"  FAILED    {c['arxiv_id']}: {msg}")
            failed += 1

    print(f"\nDone: {submitted} submitted, {failed} failed.")


if __name__ == "__main__":
    main()
