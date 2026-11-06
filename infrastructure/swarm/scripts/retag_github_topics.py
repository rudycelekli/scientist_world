#!/usr/bin/env python3
"""Repair GitHub repo metadata for Swarm papers from persisted state.

Swarm sets a repository's description, homepage and topics after every
push. Those GitHub API calls are best-effort, so a repository can end up
with stale or missing metadata. This script rebuilds it from
``SWARM_DIR/papers/<arxiv_id>/meta.json`` and the paper's workspace, using
the same rules as the scheduler.

Run it from an environment where ``swarm`` is installed, on the host that
holds the Swarm state. It needs ``github_org`` in the Swarm config and a
GitHub token in ``~/.git-credentials``. The daemon does not have to run.

    python infrastructure/swarm/scripts/retag_github_topics.py                 # all papers
    python infrastructure/swarm/scripts/retag_github_topics.py 2512.01152      # selected papers
    python infrastructure/swarm/scripts/retag_github_topics.py --missing-only  # only repos without topics
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
from pathlib import Path

from swarm.config import PAPERS_DIR, load_config
from swarm.paper import resolve_paper
from swarm.scheduler import (
    _MILESTONE_ORDER,
    _iteration_completion_topics,
    _overview_domain_topics,
)
from swarm.workspace import _get_github_token, _github_api, _sanitize_github_topics


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _milestone_topic(milestone: str | None) -> list[str]:
    if not milestone or milestone == "none":
        return []
    return [f"milestone-{_slugify(milestone)}"]


def _meta_pathspec_ids(ids: list[str] | None) -> list[str]:
    if ids:
        return ids
    return sorted(p.parent.name for p in PAPERS_DIR.glob("*/meta.json"))


def _repair_one(arxiv_id: str, *, missing_only: bool, verbose: bool) -> tuple[bool, str]:
    cfg = load_config()
    if not cfg.github_org:
        return False, "github_org is not configured"

    meta_path = PAPERS_DIR / arxiv_id / "meta.json"
    if not meta_path.exists():
        return False, "meta.json not found"

    meta = json.loads(meta_path.read_text())
    paper = resolve_paper(arxiv_id, cfg.scout_data_dir)
    full_name = f"{cfg.github_org}/{paper.repo_name()}"
    token = _get_github_token()
    if not token:
        return False, "no GitHub token found"

    status, current = _github_api("GET", f"/repos/{full_name}/topics", token)
    if status != 200 or not current:
        return False, f"failed to fetch current topics (status={status})"

    current_topics = current.get("names", [])
    if missing_only and current_topics:
        return True, f"skipped (already has {len(current_topics)} topics)"

    title = paper.title or paper.arxiv_id
    description = f"{meta.get('milestone', 'none')}: {title}"[:350]
    homepage = f"https://arxiv.org/abs/{paper.arxiv_id}"
    status, resp = _github_api(
        "PATCH",
        f"/repos/{full_name}",
        token,
        {"description": description, "homepage": homepage},
    )
    if status != 200:
        msg = resp.get("message", "") if resp else "unknown"
        return False, f"repo metadata update failed: {msg}"

    workspace = Path(meta.get("workspace_path") or "")
    desired_topics: list[str] = []
    desired_topics.extend(_milestone_topic(meta.get("milestone")))
    if workspace.is_dir():
        desired_topics.extend(_overview_domain_topics(workspace))
    if meta.get("date"):
        desired_topics.append(f"date-{meta['date']}")
    desired_topics.extend(meta.get("topics") or [])
    workflow = meta.get("workflow")
    workflow_step = int(meta.get("workflow_step", 0) or 0)
    workflow_steps = cfg.workflows.get(workflow, []) if workflow else []
    has_next_iteration = bool(workflow and workflow_step + 1 < len(workflow_steps))
    desired_topics.extend(
        _iteration_completion_topics(
            int(meta.get("iteration", 1) or 1),
            status=str(meta.get("status", "")),
            has_next_iteration=has_next_iteration,
        )
    )

    milestone_topics = {
        f"milestone-{_slugify(name)}"
        for name in list(_MILESTONE_ORDER) + ["not_possible"]
    }
    for topic in current_topics:
        if topic in milestone_topics:
            continue
        if topic not in desired_topics:
            desired_topics.append(topic)

    sanitized_topics, rewritten, dropped = _sanitize_github_topics(desired_topics)
    payload = {"names": sanitized_topics[:20]}
    status, resp = _github_api("PUT", f"/repos/{full_name}/topics", token, payload)
    if status != 200:
        msg = resp.get("message", "") if resp else "unknown"
        return False, f"topic update failed: {msg}"

    details = [f"set {len(payload['names'])} topics"]
    if rewritten and verbose:
        details.append(f"rewritten={rewritten}")
    if dropped and verbose:
        details.append(f"dropped={dropped}")
    return True, ", ".join(details)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair GitHub topics for Swarm repos")
    parser.add_argument("arxiv_ids", nargs="*", help="Specific arXiv IDs to repair")
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only repair repos whose current GitHub topics list is empty",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print rewritten/dropped topic details",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel GitHub requests to use during repair",
    )
    args = parser.parse_args()

    ids = _meta_pathspec_ids(args.arxiv_ids)
    if not ids:
        print("No paper metadata found.")
        return 0

    failures = 0
    max_workers = max(1, min(args.workers, len(ids)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _repair_one,
                arxiv_id,
                missing_only=args.missing_only,
                verbose=args.verbose,
            ): arxiv_id
            for arxiv_id in ids
        }
        for future in as_completed(futures):
            arxiv_id = futures[future]
            try:
                ok, msg = future.result()
            except Exception as e:
                ok, msg = False, f"unexpected error: {e}"
            prefix = "OK" if ok else "FAIL"
            print(f"{prefix} {arxiv_id}: {msg}", flush=True)
            if not ok:
                failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
