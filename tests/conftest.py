"""
conftest.py — Shared pytest fixtures for the Enterprise RAG test suite.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on the import path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_pages():
    """A minimal list of page dicts as returned by ingest.load_documents()."""
    return [
        {
            "text": "The attendance policy requires a minimum of 75 percent attendance. "
                    "Students failing to meet this requirement will not be permitted to sit exams. "
                    "This rule applies to all courses without exception.",
            "source_file": "policy.pdf",
            "page_number": 1,
        },
        {
            "text": "Leave applications must be submitted at least two weeks in advance. "
                    "Emergency leave requires manager approval within 24 hours. "
                    "Annual leave entitlement is 21 days per calendar year.",
            "source_file": "policy.pdf",
            "page_number": 2,
        },
        {
            "text": "Performance reviews are conducted every six months. "
                    "The review criteria include quality of work, punctuality, and collaboration. "
                    "Results are communicated in writing within 10 business days.",
            "source_file": "report.pdf",
            "page_number": 5,
        },
    ]


@pytest.fixture
def sample_chunks(sample_pages):
    """Chunks produced from sample_pages via the real chunk_documents function."""
    from ingest import chunk_documents
    return chunk_documents(sample_pages)
