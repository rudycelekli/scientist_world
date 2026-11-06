#!/usr/bin/env python3
"""Fetch all arXiv papers for a given day and category.

Uses the arXiv Atom API (stdlib only, no external dependencies).
Papers are selected by original submission date (v1) and by category. The
arXiv ``cat:`` query also matches cross-listed papers, so a listing is not
limited to papers whose primary category is the requested one.
A global paper index deduplicates across dates and categories.

Known issue: Withdrawn/unavailable papers are not filtered here. The arXiv
Atom API has no reliable field indicating withdrawal status. A HEAD request
to the PDF URL could detect 404s, but adds ~3s per paper due to rate limiting.
These papers are caught by the PDF download (HTTP 404) and skipped by the
later stages. The OAI-PMH arXivRaw format has a source_type field that
reliably indicates withdrawal, but requires a different API.

Usage:
    python fetch_daily_arxiv.py                          # last weekday, cs.LG
    python fetch_daily_arxiv.py --date 2025-06-09        # specific date
    python fetch_daily_arxiv.py --category cs.AI cs.CV   # other categories
    python fetch_daily_arxiv.py --data-dir /path/to/data # explicit data directory
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
PAGE_SIZE = 500
NS_ATOM = "{http://www.w3.org/2005/Atom}"
NS_ARXIV = "{http://arxiv.org/schemas/atom}"
NS_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


def resolve_data_dir(cli_value) -> Path:
    """Return the data directory: --data-dir, else $SCOUT_DATA_DIR, else ./data."""
    return Path(cli_value or os.environ.get("SCOUT_DATA_DIR") or "data").resolve()


def default_date() -> str:
    """Most recent weekday before today (arXiv has no weekend announcements)."""
    d = datetime.now() - timedelta(days=1)
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


# ── Index management ─────────────────────────────────────────────────────────

def load_index(index_path: Path) -> dict[str, dict]:
    """Load the global paper index: {arxiv_id: {full metadata}}."""
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}


def save_index(index: dict[str, dict], index_path: Path) -> None:
    """Atomically write the index (write tmp then rename)."""
    tmp = index_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(sorted(index.items())), indent=2, ensure_ascii=False))
    tmp.rename(index_path)


# ── arXiv API ─────────────────────────────────────────────────────────────────

def fetch_page(query: str, start: int, max_results: int,
               max_retries: int = 5) -> ET.Element:
    """Fetch one page of results from the arXiv API with retry on rate limit."""
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    })
    url = f"{ARXIV_API}?{params}"
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url) as resp:
                return ET.fromstring(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Rate limited (429), waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def parse_entry(entry: ET.Element) -> dict | None:
    """Parse one Atom entry into a paper dict. Returns None for empty/error entries."""
    raw_id = entry.findtext(f"{NS_ATOM}id", "")
    if "/abs/" not in raw_id:
        return None

    # arxiv ID: strip URL prefix and version suffix
    full_id = raw_id.split("/abs/")[-1]
    base_id = full_id.rsplit("v", 1)[0] if "v" in full_id else full_id

    # Authors (with optional affiliations)
    authors = []
    for a in entry.findall(f"{NS_ATOM}author"):
        name = a.findtext(f"{NS_ATOM}name", "")
        affs = [e.text for e in a.findall(f"{NS_ARXIV}affiliation") if e.text]
        rec = {"name": name}
        if affs:
            rec["affiliations"] = affs
        authors.append(rec)

    # Categories
    cats = [c.get("term") for c in entry.findall(f"{NS_ATOM}category") if c.get("term")]
    primary = ""
    pc = entry.find(f"{NS_ARXIV}primary_category")
    if pc is not None:
        primary = pc.get("term", "")

    title = " ".join((entry.findtext(f"{NS_ATOM}title", "") or "").split())
    abstract = (entry.findtext(f"{NS_ATOM}summary", "") or "").strip()

    return {
        "arxiv_id": base_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "published": entry.findtext(f"{NS_ATOM}published", ""),   # v1 date
        "updated": entry.findtext(f"{NS_ATOM}updated", ""),       # latest version date
        "primary_category": primary,
        "categories": cats,
        "doi": entry.findtext(f"{NS_ARXIV}doi", None),
        "journal_ref": entry.findtext(f"{NS_ARXIV}journal_ref", None),
        "comment": entry.findtext(f"{NS_ARXIV}comment", None),
        "links": {
            "abs": f"https://arxiv.org/abs/{base_id}",
            "pdf": f"https://arxiv.org/pdf/{base_id}",
        },
    }


def fetch_papers(date_str: str, category: str) -> list[dict]:
    """Fetch all papers submitted on `date_str` under `category`."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    date_from = d.strftime("%Y%m%d0000")
    date_to = d.strftime("%Y%m%d2359")
    query = f"cat:{category} AND submittedDate:[{date_from} TO {date_to}]"

    # First page
    root = fetch_page(query, 0, PAGE_SIZE)
    total = int(root.findtext(f"{NS_OPENSEARCH}totalResults", "0"))
    print(f"  Total results reported by API: {total}")

    papers = []
    for e in root.findall(f"{NS_ATOM}entry"):
        p = parse_entry(e)
        if p:
            papers.append(p)

    # Remaining pages
    start = PAGE_SIZE
    while start < total:
        time.sleep(3)  # arXiv rate limit: 1 request per 3 seconds
        print(f"  Fetching results {start}–{start + PAGE_SIZE}...")
        root = fetch_page(query, start, PAGE_SIZE)
        for e in root.findall(f"{NS_ATOM}entry"):
            p = parse_entry(e)
            if p:
                papers.append(p)
        start += PAGE_SIZE

    return papers


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch arXiv papers by day and category.")
    parser.add_argument("--date", default=None,
                        help="YYYY-MM-DD (default: last weekday)")
    parser.add_argument("--category", nargs="+", default=["cs.LG"],
                        help="arXiv categories (default: cs.LG)")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory (default: $SCOUT_DATA_DIR, else ./data)")
    args = parser.parse_args()

    date_str = args.date or default_date()

    base = resolve_data_dir(args.data_dir)
    index_path = base / "paper_index.json"
    out_dir = base / "arxiv_listings" / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load global index once
    index = load_index(index_path)
    index_modified = False

    for category in args.category:
        out_file = out_dir / f"{category}.json"

        # Skip if this date/category was already fetched
        if out_file.exists():
            existing = json.loads(out_file.read_text())
            print(f"Already fetched: {date_str}/{category} ({len(existing)} papers). Skipping.")
            continue

        # Fetch from arXiv
        print(f"Fetching: category={category}  date={date_str}")
        all_papers = fetch_papers(date_str, category)
        print(f"  API returned {len(all_papers)} papers")

        # Save listing with ALL IDs (complete record of this category/date)
        all_ids = [p["arxiv_id"] for p in all_papers]
        out_file.write_text(json.dumps(all_ids, indent=2))

        # Add only new papers to the global index
        source = f"{date_str}/{category}"
        new_papers = [p for p in all_papers if p["arxiv_id"] not in index]
        for p in new_papers:
            entry = dict(p, fetched_from=source)
            del entry["arxiv_id"]  # key is already the arxiv_id
            index[p["arxiv_id"]] = entry
        index_modified = True

        print(f"  Total: {len(all_papers)}  |  New to index: {len(new_papers)}  |  Already indexed: {len(all_papers) - len(new_papers)}")
        print(f"  Saved → {out_file}")

    # Save index once at the end
    if index_modified:
        save_index(index, index_path)
        print(f"Index total: {len(index)} unique papers")


if __name__ == "__main__":
    main()
