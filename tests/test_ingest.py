"""
tests/test_ingest.py — Tests for the PDF ingestion pipeline.

Tests:
  - Text cleaning
  - Document chunking
  - Metadata preservation through the pipeline
  - Empty page handling
  - Missing docs folder error handling
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import clean_text, chunk_documents


# ── clean_text ────────────────────────────────────────────────────────────────

def test_clean_text_collapses_multiple_spaces():
    raw = "This   has   extra   spaces"
    result = clean_text(raw)
    assert "   " not in result
    assert "This has extra spaces" == result


def test_clean_text_collapses_multiple_newlines():
    raw = "Paragraph one.\n\n\n\nParagraph two."
    result = clean_text(raw)
    assert "\n\n\n" not in result


def test_clean_text_strips_whitespace():
    raw = "   hello world   "
    assert clean_text(raw) == "hello world"


def test_clean_text_empty_string():
    assert clean_text("") == ""


def test_clean_text_preserves_paragraph_structure():
    raw = "First paragraph.\n\nSecond paragraph."
    result = clean_text(raw)
    assert "First paragraph." in result
    assert "Second paragraph." in result


# ── chunk_documents ───────────────────────────────────────────────────────────

@pytest.mark.slow
def test_chunk_documents_returns_list():
    pages = [
        {
            "text": "A" * 2000,  # Long enough to produce multiple chunks
            "source_file": "test.pdf",
            "page_number": 1,
        }
    ]
    chunks = chunk_documents(pages)
    assert isinstance(chunks, list)
    assert len(chunks) > 1


@pytest.mark.slow
def test_chunk_documents_preserves_source_file_metadata():
    pages = [
        {
            "text": "Some document content that is long enough to be chunked properly.",
            "source_file": "policy.pdf",
            "page_number": 3,
        }
    ]
    chunks = chunk_documents(pages)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["source_file"] == "policy.pdf"


@pytest.mark.slow
def test_chunk_documents_preserves_page_number_metadata():
    pages = [
        {
            "text": "Content on page five of the document.",
            "source_file": "report.pdf",
            "page_number": 5,
        }
    ]
    chunks = chunk_documents(pages)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["page_number"] == 5


@pytest.mark.slow
def test_chunk_documents_multiple_pages_preserve_metadata():
    pages = [
        {"text": "Page one content. " * 50, "source_file": "doc.pdf", "page_number": 1},
        {"text": "Page two content. " * 50, "source_file": "doc.pdf", "page_number": 2},
    ]
    chunks = chunk_documents(pages)
    page_numbers = {c.metadata["page_number"] for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers


def test_chunk_documents_empty_input():
    chunks = chunk_documents([])
    assert chunks == []


@pytest.mark.slow
def test_chunk_documents_chunk_size_respected():
    """No chunk should exceed CHUNK_SIZE + a small margin (overlap can push slightly over)."""
    from ingest import CHUNK_SIZE
    pages = [{"text": "Word " * 1000, "source_file": "x.pdf", "page_number": 1}]
    chunks = chunk_documents(pages)
    for chunk in chunks:
        # Allow 20% margin for splitter edge cases
        assert len(chunk.page_content) <= CHUNK_SIZE * 1.2


@pytest.mark.slow
def test_chunk_documents_non_empty_content():
    pages = [{"text": "Important enterprise policy details.", "source_file": "x.pdf", "page_number": 1}]
    chunks = chunk_documents(pages)
    for chunk in chunks:
        assert len(chunk.page_content.strip()) > 0


# ── load_documents (with mocked pypdf) ───────────────────────────────────────

def test_load_documents_empty_folder(tmp_path):
    from ingest import load_documents
    result = load_documents(str(tmp_path))
    assert result == []


def test_load_documents_skips_empty_pages(tmp_path):
    """Pages that extract no text should be skipped, not added as empty entries."""
    mock_page_empty = MagicMock()
    mock_page_empty.extract_text.return_value = ""

    mock_page_text = MagicMock()
    mock_page_text.extract_text.return_value = "This page has actual content."

    mock_reader = MagicMock()
    mock_reader.pages = [mock_page_empty, mock_page_text]

    # Create a fake PDF file so glob finds it
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    with patch("ingest.PdfReader", return_value=mock_reader):
        from ingest import load_documents
        result = load_documents(str(tmp_path))

    # Empty pages should be filtered out — only the page with text should remain
    assert len(result) == 1
    assert result[0]["text"] == "This page has actual content."


# ── OCR fallback and statistics tests ─────────────────────────────────────────

def test_load_documents_normal_does_not_trigger_ocr(tmp_path):
    """Pages with sufficient text should NOT invoke OCR fallback."""
    fake_pdf = tmp_path / "normal.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is a normal document page with plenty of extractable text content."
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("ingest.PdfReader", return_value=mock_reader), \
         patch("ingest.ocr_page") as mock_ocr:
        from ingest import load_documents, get_ingestion_stats
        result = load_documents(str(tmp_path))

    stats = get_ingestion_stats()
    mock_ocr.assert_not_called()
    assert len(result) == 1
    assert result[0]["source_file"] == "normal.pdf"
    assert result[0]["page_number"] == 1
    assert stats["pages_normal"] == 1
    assert stats["pages_ocr"] == 0


def test_load_documents_triggers_ocr_when_text_empty(tmp_path):
    """Pages with no text stream should automatically trigger the OCR fallback."""
    fake_pdf = tmp_path / "scanned.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""  # No text stream
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("ingest.PdfReader", return_value=mock_reader), \
         patch("ingest.ocr_page", return_value="Text successfully extracted via OCR fallback.") as mock_ocr:
        from ingest import load_documents, get_ingestion_stats
        result = load_documents(str(tmp_path))

    stats = get_ingestion_stats()
    mock_ocr.assert_called_once_with(fake_pdf, 0)
    assert len(result) == 1
    assert result[0]["text"] == "Text successfully extracted via OCR fallback."
    assert result[0]["source_file"] == "scanned.pdf"
    assert result[0]["page_number"] == 1
    assert stats["pages_ocr"] == 1
    assert stats["pages_normal"] == 0


def test_load_documents_failed_ocr_handling(tmp_path):
    """Pages where both normal text extraction and OCR fail should be counted as failed."""
    fake_pdf = tmp_path / "blank.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("ingest.PdfReader", return_value=mock_reader), \
         patch("ingest.ocr_page", return_value="") as mock_ocr:
        from ingest import load_documents, get_ingestion_stats
        result = load_documents(str(tmp_path))

    stats = get_ingestion_stats()
    mock_ocr.assert_called_once()
    assert result == []
    assert stats["pages_failed"] == 1


def test_load_documents_mixed_pages_preserve_metadata(tmp_path):
    """Mixed document with normal page and OCR page preserves correct 1-indexed page numbers."""
    fake_pdf = tmp_path / "mixed.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    mock_p1 = MagicMock()
    mock_p1.extract_text.return_value = "Page one has standard digital text in the stream."
    mock_p2 = MagicMock()
    mock_p2.extract_text.return_value = ""  # Scanned page 2
    mock_reader = MagicMock()
    mock_reader.pages = [mock_p1, mock_p2]

    def fake_ocr(path, page_idx):
        if page_idx == 1:
            return "Page two OCR extracted content."
        return ""

    with patch("ingest.PdfReader", return_value=mock_reader), \
         patch("ingest.ocr_page", side_effect=fake_ocr):
        from ingest import load_documents
        result = load_documents(str(tmp_path))

    assert len(result) == 2
    assert result[0]["page_number"] == 1
    assert result[0]["source_file"] == "mixed.pdf"
    assert "standard digital text" in result[0]["text"]

    assert result[1]["page_number"] == 2
    assert result[1]["source_file"] == "mixed.pdf"
    assert "Page two OCR extracted content." in result[1]["text"]
