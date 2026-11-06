#!/usr/bin/env python3
"""Convert locally stored arXiv PDFs to markdown using MinerU.

Run on a machine with a GPU. No network access is needed once the MinerU
models are downloaded.
Reads PDFs from <data-dir>/arxiv_pdfs/, writes markdown to <data-dir>/arxiv_mds/.
Skips papers that already have markdown output.
Uses MinerU's native batch processing (directory input).

MinerU outputs to <output>/<id>/<backend>/; this script flattens it to <output>/<id>/.

Requires the `mineru` command line tool (see requirements.txt).

Usage:
    # One day's listing, or explicit IDs
    python convert_pdf2md.py --date 2025-06-09 --category cs.AI --batch-size 80
    python convert_pdf2md.py --ids 2506.07330 2506.07334

    # Mass processing: convert ALL unprocessed PDFs in the folder
    python convert_pdf2md.py --all --batch-size 80 --data-dir $DATA_DIR

    # Multi-node: launch on N nodes simultaneously (lock-file based, no conflicts)
    python convert_pdf2md.py --all --batch-size 80 --data-dir $DATA_DIR   # node 1
    python convert_pdf2md.py --all --batch-size 80 --data-dir $DATA_DIR   # node 2

    # Deterministic sharding (alternative to lock files, zero coordination)
    python convert_pdf2md.py --all --batch-size 80 --shard 0/4 --data-dir $DATA_DIR
    python convert_pdf2md.py --all --batch-size 80 --shard 1/4 --data-dir $DATA_DIR
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path


def resolve_data_dir(cli_value) -> Path:
    """Return the data directory: --data-dir, else $SCOUT_DATA_DIR, else ./data."""
    return Path(cli_value or os.environ.get("SCOUT_DATA_DIR") or "data").resolve()


def default_date() -> str:
    """Most recent weekday before today (matches fetch_daily_arxiv.py)."""
    d = datetime.now() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


LOCK_FILE = ".lock"
LOCK_TIMEOUT = 7200  # 2 hours — stale lock threshold


def is_converted(md_dir: Path, aid: str) -> bool:
    """Check if a paper already has markdown output."""
    md_file = md_dir / aid / f"{aid}.md"
    return md_file.exists() and md_file.stat().st_size > 0


def try_lock(md_dir: Path, aid: str, lock_timeout: int = LOCK_TIMEOUT) -> bool:
    """Atomically claim a paper for processing via lock file.

    Returns True if we acquired the lock, False if already done or locked.
    Uses O_CREAT|O_EXCL for atomic creation on shared filesystems (GPFS/NFS).
    """
    if is_converted(md_dir, aid):
        return False

    paper_dir = md_dir / aid
    paper_dir.mkdir(parents=True, exist_ok=True)
    lock_path = paper_dir / LOCK_FILE

    # Check for stale lock
    if lock_path.exists():
        try:
            age = time.time() - lock_path.stat().st_mtime
            if age < lock_timeout:
                return False  # Another worker owns it
            lock_path.unlink()  # Stale — reclaim
        except OSError:
            return False

    # Atomic create
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        info = f"{socket.gethostname()}:{os.getpid()}:{time.time():.0f}\n"
        os.write(fd, info.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False  # Another worker beat us


def release_lock(md_dir: Path, aid: str) -> None:
    """Remove lock file for a paper."""
    lock_path = md_dir / aid / LOCK_FILE
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def shard_filter(ids: list[str], shard_idx: int, num_shards: int) -> list[str]:
    """Deterministically partition IDs by consistent hash."""
    return [aid for aid in ids
            if int(hashlib.md5(aid.encode()).hexdigest(), 16) % num_shards == shard_idx]


def cleanup_mineru_artifacts(paper_dir: Path, aid: str,
                             pdf_dir: Path | None = None) -> None:
    """Remove MinerU intermediate artifacts, keeping only .md, images/, and <id>.pdf symlink."""
    keep = {f"{aid}.md", "images"}
    for item in list(paper_dir.iterdir()):
        if item.name not in keep:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    # Replace _origin.pdf copy with a symlink to the original PDF
    if pdf_dir is not None:
        link = paper_dir / f"{aid}.pdf"
        target = pdf_dir / f"{aid}.pdf"
        if not link.exists() and target.exists():
            os.symlink(target.resolve(), link)


def flatten_mineru_output(md_dir: Path, arxiv_ids: list[str],
                          pdf_dir: Path | None = None) -> tuple[int, int]:
    """Flatten MinerU output from <id>/<backend>/ to <id>/.

    Returns (converted, failed) counts.
    """
    converted = 0
    failed = 0
    for aid in arxiv_ids:
        paper_dir = md_dir / aid
        if not paper_dir.exists():
            print(f"  [{aid}] FAILED: no output directory")
            failed += 1
            continue

        md_files = list(paper_dir.rglob("*.md"))
        if not md_files:
            print(f"  [{aid}] FAILED: no .md file produced — cleaning up")
            shutil.rmtree(paper_dir)
            failed += 1
            continue

        md_file = md_files[0]
        backend_dir = md_file.parent

        # Already flat — just clean up artifacts
        if backend_dir == paper_dir:
            cleanup_mineru_artifacts(paper_dir, aid, pdf_dir)
            converted += 1
            continue

        # Move backend dir contents up to paper_dir
        for item in backend_dir.iterdir():
            dest = paper_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))

        # Remove the now-empty backend dir
        for subdir in list(paper_dir.iterdir()):
            if subdir.is_dir() and not any(subdir.iterdir()):
                subdir.rmdir()

        cleanup_mineru_artifacts(paper_dir, aid, pdf_dir)
        converted += 1

    return converted, failed


def _run_batch(batch: list[str], pdf_dir: Path, md_dir: Path) -> tuple[int, int]:
    """Run MinerU on a batch of papers. Returns (converted, failed)."""
    with tempfile.TemporaryDirectory() as tmp_input:
        for aid in batch:
            src = pdf_dir / f"{aid}.pdf"
            if src.exists():
                os.symlink(src, Path(tmp_input) / f"{aid}.pdf")

        result = subprocess.run(
            ["mineru", "-p", tmp_input, "-o", str(md_dir)],
        )
        if result.returncode != 0:
            print(f"  MinerU exited with code {result.returncode}")

    converted, failed = flatten_mineru_output(md_dir, batch, pdf_dir)
    print(f"  Converted: {converted}  |  Failed: {failed}")
    return converted, failed


def main():
    parser = argparse.ArgumentParser(description="Convert local PDFs to markdown via MinerU.")
    parser.add_argument("--date", default=None,
                        help="YYYY-MM-DD (default: last weekday)")
    parser.add_argument("--category", nargs="+", default=["cs.LG"],
                        help="arXiv categories (default: cs.LG)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of papers to convert")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Papers per MinerU batch to avoid OOM (default: 10)")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Specific arxiv IDs to convert (ignores --date/--category)")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory (default: $SCOUT_DATA_DIR, else ./data)")
    parser.add_argument("--cleanup-existing", action="store_true",
                        help="Clean up MinerU artifacts from all existing output folders, then exit")
    parser.add_argument("--all", action="store_true",
                        help="Process ALL unprocessed PDFs (ignores --date/--category). "
                             "Safe to run from multiple processes simultaneously.")
    parser.add_argument("--shard", default=None, metavar="K/N",
                        help="Deterministic sharding: process shard K of N "
                             "(e.g. --shard 0/4). Alternative to lock-based dedup.")
    parser.add_argument("--lock-timeout", type=int, default=LOCK_TIMEOUT,
                        help=f"Stale lock timeout in seconds (default: {LOCK_TIMEOUT})")
    args = parser.parse_args()

    shard_idx = shard_total = None
    if args.shard:
        try:
            k, n = args.shard.split("/")
            shard_idx, shard_total = int(k), int(n)
            assert 0 <= shard_idx < shard_total
        except (ValueError, AssertionError):
            parser.error("--shard must be K/N where 0 <= K < N (e.g. 0/4)")

    base = resolve_data_dir(args.data_dir)
    pdf_dir = base / "arxiv_pdfs"
    md_dir = base / "arxiv_mds"
    md_dir.mkdir(parents=True, exist_ok=True)

    if args.cleanup_existing:
        cleaned = 0
        for paper_dir in sorted(md_dir.iterdir()):
            if not paper_dir.is_dir():
                continue
            aid = paper_dir.name
            if (paper_dir / f"{aid}.md").exists():
                cleanup_mineru_artifacts(paper_dir, aid, pdf_dir)
                cleaned += 1
        print(f"Cleaned up {cleaned} existing folders.")
        return

    # ── Determine which papers to convert ─────────────────────────────
    use_locks = False

    if args.all:
        # --all mode: scan entire pdf_dir
        all_pdfs = sorted(f.stem for f in pdf_dir.glob("*.pdf"))
        if shard_total:
            all_pdfs = shard_filter(all_pdfs, shard_idx, shard_total)
            print(f"All PDFs: shard {shard_idx}/{shard_total} → {len(all_pdfs)} papers")
        else:
            print(f"All PDFs: {len(all_pdfs)} papers")
            use_locks = True  # enable lock-based dedup for concurrent processes

        already_done = sum(1 for aid in all_pdfs if is_converted(md_dir, aid))
        target_ids = [aid for aid in all_pdfs if not is_converted(md_dir, aid)]
        print(f"  Already done: {already_done}  |  Remaining: {len(target_ids)}")

        if args.limit:
            target_ids = target_ids[:args.limit]
            print(f"  Limited to: {len(target_ids)}")

    elif args.ids:
        target_ids = args.ids
    else:
        date_str = args.date or default_date()
        target_ids = []
        for category in args.category:
            listing_file = base / "arxiv_listings" / date_str / f"{category}.json"
            if not listing_file.exists():
                print(f"Listing not found: {date_str}/{category} — skipping.")
                continue
            ids = json.loads(listing_file.read_text())
            print(f"Listing: {date_str}/{category}  ({len(ids)} papers)")
            target_ids.extend(ids)
        target_ids = list(dict.fromkeys(target_ids))  # deduplicate
        if not target_ids:
            print("No listings found.")
            return

    if not args.all:
        # Filter: must have PDF downloaded, must not already have markdown
        todo = []
        skipped = 0
        already_done = 0
        for aid in target_ids:
            pdf_path = pdf_dir / f"{aid}.pdf"
            if not pdf_path.exists():
                skipped += 1
            elif is_converted(md_dir, aid):
                already_done += 1
            else:
                todo.append(aid)

        if args.limit:
            todo = todo[:args.limit]

        print(f"Total: {len(target_ids)}  |  To convert: {len(todo)}  |  "
              f"Already done: {already_done}  |  Skipped: {skipped}")
    else:
        todo = target_ids

    if not todo:
        print("Nothing to convert.")
        return

    # ── Process in batches ─────────────────────────────────────────────
    batch_size = args.batch_size
    t_start = time.time()
    converted_total, failed_total, skipped_lock = 0, 0, 0

    if use_locks:
        # Lock-based mode: claim papers dynamically per batch.
        # Keeps looping through todo, claiming what's available.
        # Other processes running simultaneously will claim different papers.
        pos = 0
        batch_num = 0
        while pos < len(todo):
            claimed = []
            while len(claimed) < batch_size and pos < len(todo):
                aid = todo[pos]
                pos += 1
                # Re-check done (another process may have finished it)
                if is_converted(md_dir, aid):
                    skipped_lock += 1
                    continue
                if try_lock(md_dir, aid, args.lock_timeout):
                    claimed.append(aid)
                else:
                    skipped_lock += 1

            if not claimed:
                continue

            batch_num += 1
            print(f"\nBatch {batch_num} ({len(claimed)} papers, "
                  f"skipped {skipped_lock} locked/done)")

            converted, failed = _run_batch(claimed, pdf_dir, md_dir)
            converted_total += converted
            failed_total += failed

            # Release every lock: converted papers are recognised by their .md
            # file, failed ones become available for another worker to retry.
            for aid in claimed:
                release_lock(md_dir, aid)
    else:
        # Simple mode: no locks needed (date-based or sharded)
        num_batches = (len(todo) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            batch = todo[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            print(f"\nBatch {batch_idx + 1}/{num_batches} ({len(batch)} papers)")
            converted, failed = _run_batch(batch, pdf_dir, md_dir)
            converted_total += converted
            failed_total += failed

    total_elapsed = time.time() - t_start
    avg = total_elapsed / converted_total if converted_total else 0
    print(f"\nDone. Converted: {converted_total}  |  Failed: {failed_total}"
          + (f"  |  Skipped (other workers): {skipped_lock}" if skipped_lock else ""))
    print(f"Total time: {total_elapsed:.1f}s  |  Avg per paper: {avg:.1f}s")


if __name__ == "__main__":
    main()
