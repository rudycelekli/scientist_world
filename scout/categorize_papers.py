#!/usr/bin/env python3
"""Type prefilter: cheap first-pass paper_type label from markdown excerpts.

Mirrors prefilter_papers.py, but predicts only the primary paper_type from
the first 15,000 characters of each markdown file. This is intended as a cheap
routing stage before the expensive structured analysis. It uses the same kind
of OpenAI-compatible endpoint as prefilter_papers.py.
Results are appended to <data-dir>/arxiv_categories/categories.jsonl.

Usage:
    python categorize_papers.py --all --api-url http://HOST:8000/v1 --model MODEL --workers 64
    python categorize_papers.py --ids 2506.07330 2506.07334 --api-url http://HOST:8000/v1 --model MODEL
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from openai import OpenAI


EXCERPT_CHARS = 15000
PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "prefilter_category.md"
STRUCTURED_SYSTEM_PROMPT_FILE = (
    Path(__file__).resolve().parent / "prompts" / "prefilter_category_structured.md"
)
REPAIR_PROMPT_FILE = (
    Path(__file__).resolve().parent / "prompts" / "prefilter_category_repair.md"
)
LOCK_TIMEOUT = 600
ALLOWED_TYPES = [
    "metric_method",
    "benchmark_dataset",
    "analysis_findings",
    "theoretical",
    "survey_position",
]
CATEGORY_JSON_SCHEMA = {
    "name": "paper_category",
    "schema": {
        "type": "object",
        "properties": {
            "paper_type": {
                "type": "string",
                "enum": ALLOWED_TYPES,
            },
        },
        "required": ["paper_type"],
        "additionalProperties": False,
    },
}
CATEGORY_PATTERN = re.compile(
    r"\[\[PAPER_TYPE:\s*("
    + "|".join(re.escape(p) for p in ALLOWED_TYPES)
    + r")\s*\]\]",
    re.IGNORECASE,
)
CATEGORY_FALLBACK1 = re.compile(
    r"(?:paper[_ ]type|type|category)\s*[:=]\s*("
    + "|".join(re.escape(p) for p in ALLOWED_TYPES)
    + r")",
    re.IGNORECASE,
)


def resolve_data_dir(cli_value) -> Path:
    """Return the data directory: --data-dir, else $SCOUT_DATA_DIR, else ./data."""
    return Path(cli_value or os.environ.get("SCOUT_DATA_DIR") or "data").resolve()


def default_date() -> str:
    d = datetime.now() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def shard_filter(ids: list[str], shard_idx: int, num_shards: int) -> list[str]:
    return [
        aid for aid in ids
        if int(hashlib.md5(aid.encode()).hexdigest(), 16) % num_shards == shard_idx
    ]


def parse_category(text: str):
    matches = list(CATEGORY_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1).lower(), "bracket_tag"

    matches = list(CATEGORY_FALLBACK1.finditer(text))
    if matches:
        return matches[-1].group(1).lower(), "category_label"

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if lines and lines[-1] in ALLOWED_TYPES:
        return lines[-1], "final_line_label"

    return None, None


def parse_structured_category(text: str):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    paper_type = data.get("paper_type")
    if paper_type not in ALLOWED_TYPES:
        return None
    return paper_type


def request_structured_category(prompt: str, structured_system_prompt: str,
                                client: OpenAI, model: str,
                                max_tokens: int, temperature: float):
    budgets = [max_tokens]
    retry_budget = max(max_tokens * 2, 128)
    if retry_budget != max_tokens:
        budgets.append(retry_budget)

    last_raw = None
    last_error = None

    for budget in budgets:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": structured_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=budget,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": CATEGORY_JSON_SCHEMA,
                },
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            last_error = f"structured API error (max_tokens={budget}): {e}"
            continue

        paper_type = parse_structured_category(raw)
        if paper_type:
            return paper_type, raw, "json_schema"

        last_raw = raw
        last_error = f"structured parse failed (max_tokens={budget})"

    return None, last_raw, last_error


def repair_category(raw_text: str, repair_prompt: str,
                    client: OpenAI, model: str):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": repair_prompt + "\n\nCategorization reasoning:\n" + raw_text,
            }],
            max_tokens=32,
            temperature=0.0,
        )
        repair_raw = resp.choices[0].message.content.strip()
    except Exception as e:
        return None, None, f"repair API error: {e}"

    paper_type, parse_mode = parse_category(repair_raw)
    if paper_type is None:
        return None, repair_raw, "repair parse failed"

    return paper_type, repair_raw, f"repair_{parse_mode}"


def try_lock(category_dir: Path, aid: str) -> bool:
    lock_dir = category_dir / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{aid}.lock"

    if lock_path.exists():
        try:
            age = time.time() - lock_path.stat().st_mtime
            if age < LOCK_TIMEOUT:
                return False
            lock_path.unlink()
        except OSError:
            return False

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock(category_dir: Path, aid: str):
    try:
        (category_dir / "locks" / f"{aid}.lock").unlink(missing_ok=True)
    except OSError:
        pass


def load_existing(categories_path: Path) -> dict[str, str]:
    results = {}
    if not categories_path.exists():
        return results
    for line in categories_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            paper_type = entry["paper_type"]
            if paper_type in ALLOWED_TYPES:
                results[entry["id"]] = paper_type
        except (json.JSONDecodeError, KeyError):
            continue
    return results


class ClientPool:
    def __init__(self, api_urls: list[str], api_key: str = "unused"):
        self._clients = [OpenAI(base_url=url, api_key=api_key) for url in api_urls]
        self._idx = 0
        self._lock = threading.Lock()

    def next(self) -> OpenAI:
        with self._lock:
            client = self._clients[self._idx % len(self._clients)]
            self._idx += 1
            return client


def categorize_paper(arxiv_id: str, md_dir: Path, prompt_template: str,
                     structured_system_prompt: str, repair_prompt: str,
                     pool: ClientPool, model: str, max_tokens: int,
                     temperature: float, category_dir: Path | None = None):
    if category_dir and not try_lock(category_dir, arxiv_id):
        return "skip"

    md_path = md_dir / arxiv_id / f"{arxiv_id}.md"
    if not md_path.exists():
        return None

    excerpt = md_path.read_text()[:EXCERPT_CHARS]
    full_prompt = prompt_template + excerpt
    client = pool.next()

    paper_type, raw, parse_mode = request_structured_category(
        full_prompt, structured_system_prompt, client, model, max_tokens, temperature
    )
    if paper_type:
        return {
            "id": arxiv_id,
            "paper_type": paper_type,
            "model": model,
            "timestamp": int(time.time()),
            "parse_mode": parse_mode,
            "raw": raw,
        }

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [{arxiv_id}] API error: {e}")
        return None

    paper_type, parse_mode = parse_category(raw)
    repair_raw = None
    if not paper_type:
        paper_type, repair_raw, parse_mode = repair_category(
            raw, repair_prompt, client, model
        )
    if not paper_type:
        print(f"  [{arxiv_id}] Could not parse category from: {raw[:120]}")
        return None

    return {
        "id": arxiv_id,
        "paper_type": paper_type,
        "model": model,
        "timestamp": int(time.time()),
        "parse_mode": parse_mode,
        "raw": raw,
        "repair_raw": repair_raw,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Label papers with a primary paper_type through an "
                    "OpenAI-compatible endpoint."
    )
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: last weekday)")
    parser.add_argument("--category", nargs="+", default=["cs.LG"],
                        help="arXiv categories (default: cs.LG)")
    parser.add_argument("--all", action="store_true",
                        help="Process all papers with markdown")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Specific arxiv IDs")
    parser.add_argument("--shard", default=None, metavar="K/N",
                        help="Deterministic sharding: process shard K of N")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Data directory (default: $SCOUT_DATA_DIR, else ./data)")

    parser.add_argument("--api-url", nargs="+", required=True,
                        help="Base URL(s) of an OpenAI-compatible endpoint "
                             "(e.g. http://HOST:8000/v1). Several URLs are used round-robin.")
    parser.add_argument("--api-key", default=None,
                        help="API key for the endpoint (default: $SCOUT_API_KEY, else "
                             "'unused' for servers without authentication). Prefer the "
                             "environment variable: command-line arguments are visible "
                             "in the process list.")
    parser.add_argument("--model", required=True,
                        help="Model name as served by the endpoint")

    parser.add_argument("--max-tokens", type=int, default=5000,
                        help="Max tokens for model response (default: 5000)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel API calls (default: 16)")

    args = parser.parse_args()

    shard_idx = shard_total = None
    if args.shard:
        try:
            k, n = args.shard.split("/")
            shard_idx, shard_total = int(k), int(n)
            assert 0 <= shard_idx < shard_total
        except (ValueError, AssertionError):
            parser.error("--shard must be K/N where 0 <= K < N")

    data_dir = resolve_data_dir(args.data_dir)
    md_dir = data_dir / "arxiv_mds"
    category_dir = data_dir / "arxiv_categories"
    category_dir.mkdir(parents=True, exist_ok=True)
    categories_path = category_dir / "categories.jsonl"

    if args.all:
        target_ids = sorted(
            d.name for d in md_dir.iterdir()
            if d.is_dir() and (d / f"{d.name}.md").exists()
        )
        if shard_total:
            target_ids = shard_filter(target_ids, shard_idx, shard_total)
            print(f"All papers with markdown: shard {shard_idx}/{shard_total} → {len(target_ids)}")
        else:
            print(f"All papers with markdown: {len(target_ids)}")
    elif args.ids:
        target_ids = args.ids
    else:
        date_str = args.date or default_date()
        listing_dir = data_dir / "arxiv_listings"
        target_ids = []
        for cat in args.category:
            f = listing_dir / date_str / f"{cat}.json"
            if f.exists():
                ids = json.loads(f.read_text())
                print(f"Listing: {date_str}/{cat} — {len(ids)} papers")
                target_ids.extend(ids)
            else:
                print(f"Listing not found: {date_str}/{cat} — skipping")
        target_ids = list(dict.fromkeys(target_ids))

    if not target_ids:
        print("No papers found.")
        return

    already_done = load_existing(categories_path)
    todo = []
    no_md = 0
    for aid in target_ids:
        if aid in already_done:
            continue
        if not (md_dir / aid / f"{aid}.md").exists():
            no_md += 1
            continue
        todo.append(aid)

    if args.limit:
        todo = todo[:args.limit]

    print(f"  Already categorized: {len(already_done)}  |  No markdown: {no_md}  |  "
          f"To process: {len(todo)}")

    if not todo:
        print("Nothing to process.")
        return

    prompt_template = PROMPT_FILE.read_text()
    structured_system_prompt = STRUCTURED_SYSTEM_PROMPT_FILE.read_text()
    repair_prompt = REPAIR_PROMPT_FILE.read_text()

    api_key = args.api_key or os.environ.get("SCOUT_API_KEY") or "unused"
    pool = ClientPool(args.api_url, api_key)
    print(f"\nEndpoints: {len(args.api_url)}  |  Workers: {args.workers}  |  "
          f"Model: {args.model}")

    write_lock = threading.Lock()
    categories_file = open(categories_path, "a")

    def write_result(result):
        line = json.dumps({
            "id": result["id"],
            "paper_type": result["paper_type"],
            "model": result["model"],
            "timestamp": result["timestamp"],
            "parse_mode": result["parse_mode"],
            "raw": result["raw"],
            "repair_raw": result.get("repair_raw"),
        })
        with write_lock:
            categories_file.write(line + "\n")
            categories_file.flush()

    t_start = time.time()
    success = failed = skipped = 0
    done_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                categorize_paper, aid, md_dir, prompt_template,
                structured_system_prompt, repair_prompt, pool, args.model,
                args.max_tokens, args.temperature, category_dir
            ): aid
            for aid in todo
        }
        for future in concurrent.futures.as_completed(futures):
            aid = futures[future]
            done_count += 1
            try:
                result = future.result()
                if result == "skip":
                    skipped += 1
                elif result:
                    write_result(result)
                    release_lock(category_dir, aid)
                    success += 1
                else:
                    release_lock(category_dir, aid)
                    failed += 1
            except Exception as e:
                print(f"  [{aid}] Exception: {e}")
                release_lock(category_dir, aid)
                failed += 1

            if done_count % 100 == 0:
                elapsed = time.time() - t_start
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (len(todo) - done_count) / rate if rate > 0 else 0
                print(f"  Progress: {done_count}/{len(todo)}  |  "
                      f"OK: {success}  Skip: {skipped}  Fail: {failed}  |  "
                      f"{rate:.1f} papers/s  |  ETA: {eta:.0f}s")

    categories_file.close()
    elapsed = time.time() - t_start
    rate = success / elapsed if elapsed > 0 else 0
    print(f"\nDone. Categorized: {success}  |  Failed: {failed}  |  "
          f"Time: {elapsed:.0f}s  |  Rate: {rate:.1f} papers/s")


if __name__ == "__main__":
    main()
