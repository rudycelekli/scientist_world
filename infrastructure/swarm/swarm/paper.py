"""Paper resolution — map arxiv_id to file paths.

Given an arxiv_id and the Scout data directory, resolve paths to
the paper's markdown, images and PDF.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PaperInfo:
    """Resolved file paths for a paper.

    Attributes:
        arxiv_id: The arxiv identifier (e.g. "2512.01152").
        title: Paper title from paper_index.json (empty if unavailable).
        paper_md_path: Path to the paper's markdown file.
        paper_images_dir: Path to the paper's images directory.
        paper_pdf_path: Path to the paper's PDF file (may not exist).
    """
    arxiv_id: str
    title: str
    paper_md_path: str
    paper_images_dir: str
    paper_pdf_path: str

    def repo_name(self) -> str:
        """Generate a GitHub repo name: ``{arxiv_id}-{slugified-title}``.

        Truncates at a word boundary if the result exceeds 100 characters.
        """
        slug = _slugify(self.title) if self.title else ""
        if slug:
            name = f"{self.arxiv_id}-{slug}"
        else:
            name = self.arxiv_id
        if len(name) > 100:
            name = name[:100].rsplit("-", 1)[0]
        return name


def _slugify(text: str) -> str:
    """Convert text to a GitHub-safe repo name slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s]+", "-", text.strip())
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


_paper_index_cache: dict | None = None
_paper_index_cache_path: str | None = None


def _load_paper_index(index_path: Path) -> dict:
    """Load paper_index.json with caching (the index can be very large)."""
    global _paper_index_cache, _paper_index_cache_path
    path_str = str(index_path)
    if _paper_index_cache is not None and _paper_index_cache_path == path_str:
        return _paper_index_cache
    try:
        _paper_index_cache = json.loads(index_path.read_text())
        _paper_index_cache_path = path_str
    except (json.JSONDecodeError, OSError):
        _paper_index_cache = {}
        _paper_index_cache_path = path_str
    return _paper_index_cache


def resolve_paper(arxiv_id: str, data_dir: str) -> PaperInfo:
    """Resolve an arxiv_id to file paths under the Scout data directory.

    The markdown file is required. The PDF is optional — it may not have
    been downloaded.

    Args:
        arxiv_id: The arxiv identifier (e.g. "2512.01152").
        data_dir: Path to the Scout data/ directory.

    Returns:
        PaperInfo with resolved paths.

    Raises:
        FileNotFoundError: If the markdown file does not exist.
        ValueError: If data_dir is empty or does not exist.
    """
    if not data_dir:
        raise ValueError("data_dir must not be empty")
    base = Path(data_dir)
    if not base.is_dir():
        raise ValueError(f"data_dir does not exist: {data_dir}")

    # Title from paper_index.json (cached — the whole index is parsed)
    title = ""
    paper_index_path = base / "paper_index.json"
    if paper_index_path.exists():
        idx = _load_paper_index(paper_index_path)
        title = idx.get(arxiv_id, {}).get("title", "")

    # Markdown (required)
    md_dir = base / "arxiv_mds" / arxiv_id
    md_path = md_dir / f"{arxiv_id}.md"

    # Fallback: extract title from first markdown heading if not in index
    if not title and md_path.exists():
        try:
            for line in md_path.read_text(errors="replace").splitlines()[:20]:
                line = line.strip()
                if line.startswith("# ") and len(line) > 3:
                    title = line[2:].strip()
                    break
        except OSError:
            pass
    if not md_path.exists():
        raise FileNotFoundError(
            f"Paper markdown not found: {md_path}\n"
            f"Run Scout steps 1-3 first to fetch and convert the paper."
        )

    images_dir = md_dir / "images"

    # PDF (optional)
    pdf_path = base / "arxiv_pdfs" / f"{arxiv_id}.pdf"

    return PaperInfo(
        arxiv_id=arxiv_id,
        title=title,
        paper_md_path=str(md_path),
        paper_images_dir=str(images_dir),
        paper_pdf_path=str(pdf_path),
    )
