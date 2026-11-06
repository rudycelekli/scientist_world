"""Tests for paper resolution."""

import pytest

from swarm.paper import resolve_paper, PaperInfo


class TestResolvePaper:
    """Paper path resolution from arxiv_id."""

    def test_resolve_with_all_files(self, tmp_path):
        """Resolve when markdown, images and PDF all exist."""
        data_dir = tmp_path / "data"
        arxiv_id = "2512.01152"

        # Create expected files
        md_dir = data_dir / "arxiv_mds" / arxiv_id
        md_dir.mkdir(parents=True)
        (md_dir / f"{arxiv_id}.md").write_text("# Paper")
        (md_dir / "images").mkdir()

        pdf_dir = data_dir / "arxiv_pdfs"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / f"{arxiv_id}.pdf").write_bytes(b"%PDF")

        info = resolve_paper(arxiv_id, str(data_dir))

        assert info.arxiv_id == arxiv_id
        assert info.paper_md_path == str(md_dir / f"{arxiv_id}.md")
        assert info.paper_images_dir == str(md_dir / "images")
        assert info.paper_pdf_path == str(pdf_dir / f"{arxiv_id}.pdf")

    def test_resolve_without_optional_files(self, tmp_path):
        """Resolve when only markdown exists (the PDF is optional)."""
        data_dir = tmp_path / "data"
        arxiv_id = "2512.01152"

        md_dir = data_dir / "arxiv_mds" / arxiv_id
        md_dir.mkdir(parents=True)
        (md_dir / f"{arxiv_id}.md").write_text("# Paper")

        info = resolve_paper(arxiv_id, str(data_dir))

        assert info.arxiv_id == arxiv_id
        assert info.paper_md_path == str(md_dir / f"{arxiv_id}.md")
        assert info.paper_pdf_path == str(data_dir / "arxiv_pdfs" / f"{arxiv_id}.pdf")

    def test_resolve_missing_markdown_raises(self, tmp_path):
        """Raise FileNotFoundError when markdown doesn't exist."""
        data_dir = tmp_path / "data"
        (data_dir / "arxiv_mds").mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Paper markdown not found"):
            resolve_paper("9999.99999", str(data_dir))

    def test_resolve_empty_data_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="must not be empty"):
            resolve_paper("2512.01152", "")

    def test_resolve_nonexistent_data_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            resolve_paper("2512.01152", str(tmp_path / "nonexistent"))


class TestPaperInfo:
    """PaperInfo dataclass basics."""

    def test_dataclass_fields(self):
        info = PaperInfo(
            arxiv_id="2512.01152",
            title="My Paper Title",
            paper_md_path="/data/arxiv_mds/2512.01152/2512.01152.md",
            paper_images_dir="/data/arxiv_mds/2512.01152/images",
            paper_pdf_path="/data/arxiv_pdfs/2512.01152.pdf",
        )
        assert info.arxiv_id == "2512.01152"
        assert info.title == "My Paper Title"

    def test_repo_name(self):
        info = PaperInfo(
            arxiv_id="2512.01152",
            title="Attention Is All You Need",
            paper_md_path="", paper_images_dir="", paper_pdf_path="",
        )
        assert info.repo_name() == "2512.01152-attention-is-all-you-need"

    def test_repo_name_no_title(self):
        info = PaperInfo(
            arxiv_id="2512.01152",
            title="",
            paper_md_path="", paper_images_dir="", paper_pdf_path="",
        )
        assert info.repo_name() == "2512.01152"

    def test_repo_name_truncation(self):
        info = PaperInfo(
            arxiv_id="2512.01152",
            title="A " + "Very Long Title Word " * 10,
            paper_md_path="", paper_images_dir="", paper_pdf_path="",
        )
        name = info.repo_name()
        assert len(name) <= 100
