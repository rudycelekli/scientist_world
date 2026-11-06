#!/usr/bin/env python3
"""Quality prefilter: score papers 0-100 from the start of their markdown.

Fast triage pass over paper markdowns. Reads the first 15,000 characters
(abstract and introduction) and asks a model behind an OpenAI-compatible
chat-completions endpoint for a research-quality score from 0 to 100.
The endpoint must support JSON-schema structured output
(``response_format: json_schema``); plain-text parsing is the fallback.
Results are appended to <data-dir>/arxiv_prefilter/scores.jsonl.

Supports several endpoints (round-robin), parallel workers, sharding for
multi-process runs, and an --all mode like convert_pdf2md.py.

Usage:
    # Single endpoint
    python prefilter_papers.py --date 2026-03-01 --category cs.CL cs.CV cs.AI \\
        --api-url http://HOST:8000/v1 --model MODEL --workers 32

    # Several endpoints (round-robin load balancing)
    python prefilter_papers.py --all \\
        --api-url http://HOST1:8000/v1 http://HOST2:8000/v1 \\
        --model MODEL --workers 64

    # With sharding (one process per shard)
    python prefilter_papers.py --all --shard 0/4 --api-url http://HOST1:8000/v1 --model MODEL
    python prefilter_papers.py --all --shard 1/4 --api-url http://HOST2:8000/v1 --model MODEL
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

# ── Constants ──────────────────────────────────────────────────────────

EXCERPT_CHARS = 15000
PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "prefilter.md"
STRUCTURED_SYSTEM_PROMPT_FILE = (
    Path(__file__).resolve().parent / "prompts" / "prefilter_structured.md"
)
REPAIR_PROMPT_FILE = (
    Path(__file__).resolve().parent / "prompts" / "prefilter_repair.md"
)
SCORE_JSON_SCHEMA = {
    "name": "paper_score",
    "schema": {
        "type": "object",
        "properties": {
            "explanation": {"type": "string", "maxLength": 1600},
            "score": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["explanation", "score"],
        "additionalProperties": False,
    },
}
SCORE_PATTERN = re.compile(r"\[\[SCORE:\s*(\d{1,3})\s*\]\]", re.IGNORECASE)
SCORE_FALLBACK1 = re.compile(
    r"(?:final|overall)?\s*score\s*[:=]\s*(\d{1,3})(?:\s*/\s*100)?",
    re.IGNORECASE,
)
SCORE_FALLBACK2 = re.compile(r"^\s*(\d{1,3})(?:\s*/\s*100)?\s*$")
LOCK_TIMEOUT = 600  # 10 min — prefilter API calls are fast


# ── Helpers ────────────────────────────────────────────────────────────

def resolve_data_dir(cli_value) -> Path:
    """Return the data directory: --data-dir, else $SCOUT_DATA_DIR, else ./data."""
    return Path(cli_value or os.environ.get("SCOUT_DATA_DIR") or "data").resolve()


def default_date() -> str:
    d = datetime.now() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def shard_filter(ids: list[str], shard_idx: int, num_shards: int) -> list[str]:
    return [aid for aid in ids
            if int(hashlib.md5(aid.encode()).hexdigest(), 16) % num_shards == shard_idx]


def parse_score(text: str):
    """Extract an explicitly stated score from the model response.

    We allow brief explanations, but we only trust a clearly marked final
    score. This avoids mis-parsing numbered bullets or incidental integers
    from the explanation body.
    """
    matches = list(SCORE_PATTERN.finditer(text))
    if matches:
        val = int(matches[-1].group(1))
        if 0 <= val <= 100:
            return val, "bracket_tag"

    matches = list(SCORE_FALLBACK1.finditer(text))
    if matches:
        val = int(matches[-1].group(1))
        if 0 <= val <= 100:
            return val, "score_label"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        m = SCORE_FALLBACK2.match(lines[-1])
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val, "final_line_number"

    return None, None


def parse_structured_score(text: str):
    """Extract score and explanation from a structured JSON response."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, None

    if not isinstance(data, dict):
        return None, None

    score = data.get("score")
    explanation = data.get("explanation", "")
    if not isinstance(score, int) or not (0 <= score <= 100):
        return None, None
    if not isinstance(explanation, str):
        explanation = str(explanation)
    return score, explanation


def request_structured_score(prompt: str, structured_system_prompt: str,
                             client: OpenAI, model: str,
                             max_tokens: int, temperature: float):
    """Ask for a JSON object containing explanation and score."""
    budgets = [max_tokens]
    retry_budget = max(max_tokens * 2, 384)
    if retry_budget != max_tokens:
        budgets.append(retry_budget)

    last_error = None
    last_raw = None

    for budget in budgets:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": structured_system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=budget,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": SCORE_JSON_SCHEMA,
                },
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            last_error = f"structured API error (max_tokens={budget}): {e}"
            continue

        score, explanation = parse_structured_score(raw)
        if score is not None:
            return score, explanation, raw, "json_schema"

        last_raw = raw
        last_error = f"structured parse failed (max_tokens={budget})"

    return None, None, last_raw, last_error


def repair_score(raw_text: str, repair_prompt: str, client: OpenAI, model: str):
    """Ask the model to convert its reasoning into one explicit score tag."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": repair_prompt + "\n\nJudge reasoning:\n" + raw_text,
            }],
            max_tokens=32,
            temperature=0.0,
        )
        repair_raw = resp.choices[0].message.content.strip()
    except Exception as e:
        return None, None, f"repair API error: {e}"

    score, parse_mode = parse_score(repair_raw)
    if score is None:
        return None, repair_raw, "repair parse failed"

    return score, repair_raw, f"repair_{parse_mode}"


def try_lock(prefilter_dir: Path, aid: str) -> bool:
    """Atomically claim a paper for scoring."""
    lock_dir = prefilter_dir / "locks"
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


def release_lock(prefilter_dir: Path, aid: str):
    """Remove lock file."""
    try:
        (prefilter_dir / "locks" / f"{aid}.lock").unlink(missing_ok=True)
    except OSError:
        pass


def load_scored(scores_path: Path) -> dict[str, int]:
    """Load already-scored paper IDs from JSONL."""
    scores = {}
    if scores_path.exists():
        for line in scores_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                scores[entry["id"]] = entry["score"]
            except (json.JSONDecodeError, KeyError):
                continue
    return scores


# ── Load-balanced client pool ──────────────────────────────────────────

class ClientPool:
    """Round-robin pool of OpenAI clients for load balancing."""

    def __init__(self, api_urls: list[str], api_key: str = "unused"):
        self._clients = [
            OpenAI(base_url=url, api_key=api_key)
            for url in api_urls
        ]
        self._idx = 0
        self._lock = threading.Lock()

    def next(self) -> OpenAI:
        with self._lock:
            client = self._clients[self._idx % len(self._clients)]
            self._idx += 1
            return client


# ── Score one paper ────────────────────────────────────────────────────

def score_paper(arxiv_id: str, md_dir: Path, prompt_template: str,
                structured_system_prompt: str, repair_prompt: str,
                pool: ClientPool, model: str,
                max_tokens: int, temperature: float,
                prefilter_dir: Path = None):
    """Score a single paper. Returns result dict, 'skip', or None on failure."""
    # Atomic lock — prevents other processes from scoring the same paper
    if prefilter_dir and not try_lock(prefilter_dir, arxiv_id):
        return "skip"

    md_path = md_dir / arxiv_id / f"{arxiv_id}.md"
    if not md_path.exists():
        return None

    excerpt = md_path.read_text()[:EXCERPT_CHARS]
    full_prompt = prompt_template + excerpt

    client = pool.next()
    score, explanation, raw, parse_mode = request_structured_score(
        full_prompt, structured_system_prompt, client, model,
        max_tokens, temperature
    )
    repair_raw = None
    if score is not None:
        return {
            "id": arxiv_id,
            "score": score,
            "model": model,
            "timestamp": int(time.time()),
            "parse_mode": parse_mode,
            "explanation": explanation,
            "raw": raw,
            "repair_raw": repair_raw,
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

    score, parse_mode = parse_score(raw)
    explanation = raw
    if score is None:
        score, repair_raw, parse_mode = repair_score(
            raw, repair_prompt, client, model
        )
    if score is None:
        print(f"  [{arxiv_id}] Could not parse score from: {raw[:100]}")
        return None

    return {
        "id": arxiv_id,
        "score": score,
        "model": model,
        "timestamp": int(time.time()),
        "parse_mode": parse_mode,
        "explanation": explanation,
        "raw": raw,
        "repair_raw": repair_raw,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score papers 0-100 for research quality through an "
                    "OpenAI-compatible endpoint.")

    # Data selection
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: last weekday)")
    parser.add_argument("--category", nargs="+", default=["cs.LG"],
                        help="arXiv categories (default: cs.LG)")
    parser.add_argument("--all", action="store_true",
                        help="Process ALL papers with markdown (ignores --date/--category)")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Specific arxiv IDs (ignores --date/--category)")
    parser.add_argument("--shard", default=None, metavar="K/N",
                        help="Deterministic sharding: process shard K of N (e.g. 0/4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Data directory (default: $SCOUT_DATA_DIR, else ./data)")

    # API
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

    # Generation
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens for LLM response (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")

    # Parallelism
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel API calls (default: 16)")

    args = parser.parse_args()

    # Parse shard
    shard_idx = shard_total = None
    if args.shard:
        try:
            k, n = args.shard.split("/")
            shard_idx, shard_total = int(k), int(n)
            assert 0 <= shard_idx < shard_total
        except (ValueError, AssertionError):
            parser.error("--shard must be K/N where 0 <= K < N (e.g. 0/4)")

    data_dir = resolve_data_dir(args.data_dir)
    md_dir = data_dir / "arxiv_mds"
    prefilter_dir = data_dir / "arxiv_prefilter"
    prefilter_dir.mkdir(parents=True, exist_ok=True)
    scores_path = prefilter_dir / "scores.jsonl"

    # ── Collect paper IDs ──────────────────────────────────────────────
    if args.all:
        # All papers with markdown
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

    # Filter: must have markdown, must not already be scored
    already_scored = load_scored(scores_path)
    todo = []
    no_md = 0
    for aid in target_ids:
        if aid in already_scored:
            continue
        if not (md_dir / aid / f"{aid}.md").exists():
            no_md += 1
            continue
        todo.append(aid)

    if args.limit:
        todo = todo[:args.limit]

    print(f"  Already scored: {len(already_scored)}  |  No markdown: {no_md}  |  "
          f"To process: {len(todo)}")

    if not todo:
        print("Nothing to process.")
        return

    # ── Load prompt template ───────────────────────────────────────────
    prompt_template = PROMPT_FILE.read_text()
    structured_system_prompt = STRUCTURED_SYSTEM_PROMPT_FILE.read_text()
    repair_prompt = REPAIR_PROMPT_FILE.read_text()

    # ── Set up client pool ─────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("SCOUT_API_KEY") or "unused"
    pool = ClientPool(args.api_url, api_key)
    print(f"\nEndpoints: {len(args.api_url)}  |  Workers: {args.workers}  |  "
          f"Model: {args.model}")

    # ── Process papers ─────────────────────────────────────────────────
    write_lock = threading.Lock()
    scores_file = open(scores_path, "a")

    def write_result(result):
        line = json.dumps({
            "id": result["id"],
            "score": result["score"],
            "model": result["model"],
            "timestamp": result["timestamp"],
            "parse_mode": result["parse_mode"],
            "explanation": result["explanation"],
            "raw": result["raw"],
            "repair_raw": result["repair_raw"],
        })
        with write_lock:
            scores_file.write(line + "\n")
            scores_file.flush()

    t_start = time.time()
    success = failed = skipped = 0
    done_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                score_paper, aid, md_dir, prompt_template,
                structured_system_prompt, repair_prompt,
                pool, args.model, args.max_tokens, args.temperature,
                prefilter_dir
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
                    release_lock(prefilter_dir, aid)
                    success += 1
                else:
                    release_lock(prefilter_dir, aid)
                    failed += 1
            except Exception as e:
                print(f"  [{aid}] Exception: {e}")
                release_lock(prefilter_dir, aid)
                failed += 1

            if done_count % 100 == 0:
                elapsed = time.time() - t_start
                rate = done_count / elapsed
                eta = (len(todo) - done_count) / rate if rate > 0 else 0
                print(f"  Progress: {done_count}/{len(todo)}  |  "
                      f"OK: {success}  Skip: {skipped}  Fail: {failed}  |  "
                      f"{rate:.1f} papers/s  |  ETA: {eta:.0f}s")

    scores_file.close()
    elapsed = time.time() - t_start
    rate = success / elapsed if elapsed > 0 else 0
    print(f"\nDone. Scored: {success}  |  Failed: {failed}  |  "
          f"Time: {elapsed:.0f}s  |  Rate: {rate:.1f} papers/s")


if __name__ == "__main__":
    main()
