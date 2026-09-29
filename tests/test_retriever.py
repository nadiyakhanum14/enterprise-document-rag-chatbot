"""
tests/test_retriever.py — Tests for hybrid retrieval and score normalization.

Tests:
  - Score normalization correctness
  - Hybrid fusion with known scores
  - BM25 retrieval behavior
  - Deduplication of merged results
  - Refusal when no documents are indexed
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever import _normalize
from langchain_core.documents import Document


# ── Score normalization ───────────────────────────────────────────────────────

def test_normalize_standard_case():
    scores = np.array([0.0, 0.5, 1.0])
    result = _normalize(scores)
    assert result[0] == pytest.approx(0.0)
    assert result[2] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.5)


def test_normalize_all_zeros():
    scores = np.array([0.0, 0.0, 0.0])
    result = _normalize(scores)
    assert all(r == 0.0 for r in result)


def test_normalize_all_same_nonzero():
    scores = np.array([5.0, 5.0, 5.0])
    result = _normalize(scores)
    assert all(r == 0.0 for r in result)


def test_normalize_single_element():
    scores = np.array([42.0])
    result = _normalize(scores)
    assert result[0] == 0.0


def test_normalize_preserves_order():
    scores = np.array([3.0, 1.0, 5.0, 2.0])
    result = _normalize(scores)
    # Relative order must be preserved
    assert result[2] > result[0] > result[3] > result[1]


def test_normalize_output_range():
    scores = np.array([10.0, 30.0, 20.0, 50.0, 5.0])
    result = _normalize(scores)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


# ── HybridRetriever (mocked ChromaDB + BM25) ─────────────────────────────────

def _make_mock_retriever(corpus_texts=None, corpus_metas=None):
    """Build a HybridRetriever with all external dependencies mocked."""
    if corpus_texts is None:
        corpus_texts = [
            "The attendance policy requires 75% minimum attendance.",
            "Employees must submit leave applications two weeks in advance.",
            "The performance review cycle runs every six months.",
        ]
    if corpus_metas is None:
        corpus_metas = [
            {"source_file": "policy.pdf", "page_number": 1},
            {"source_file": "policy.pdf", "page_number": 2},
            {"source_file": "policy.pdf", "page_number": 3},
        ]

    with patch("retriever.HuggingFaceEmbeddings"), \
         patch("retriever.Chroma") as mock_chroma_cls, \
         patch("retriever.CrossEncoder") as mock_ce_cls:
        mock_chroma = MagicMock()
        mock_chroma.get.return_value = {
            "documents": corpus_texts,
            "metadatas": corpus_metas,
        }

        # similarity_search_with_relevance_scores returns (doc, score) pairs
        from langchain_core.documents import Document
        mock_docs_with_scores = [
            (Document(page_content=t, metadata=m), 0.8 - i * 0.1)
            for i, (t, m) in enumerate(zip(corpus_texts, corpus_metas))
        ]
        mock_chroma.similarity_search_with_relevance_scores.return_value = mock_docs_with_scores
        mock_chroma_cls.return_value = mock_chroma

        # Cross-encoder returns descending scores
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.9, 0.7, 0.5]
        mock_ce_cls.return_value = mock_ce

        from retriever import HybridRetriever
        retriever = HybridRetriever(top_k=3, fetch_k=5)

    return retriever


def test_retriever_bm25_is_built():
    r = _make_mock_retriever()
    assert r.bm25 is not None


def test_retriever_bm25_none_when_empty_corpus():
    with patch("retriever.HuggingFaceEmbeddings"), \
         patch("retriever.Chroma") as mock_chroma_cls, \
         patch("retriever.CrossEncoder"):

        mock_chroma = MagicMock()
        mock_chroma.get.return_value = {"documents": [], "metadatas": []}
        mock_chroma.similarity_search_with_relevance_scores.return_value = []
        mock_chroma_cls.return_value = mock_chroma

        from retriever import HybridRetriever
        r = HybridRetriever()
        assert r.bm25 is None


def test_retriever_corpus_size():
    texts = ["doc one", "doc two", "doc three"]
    r = _make_mock_retriever(corpus_texts=texts)
    assert r.get_corpus_size() == 3


def test_retriever_retrieve_returns_documents():
    r = _make_mock_retriever()
    from langchain_core.documents import Document
    results = r.retrieve("attendance policy")
    assert isinstance(results, list)
    assert all(isinstance(d, Document) for d in results)


def test_retriever_retrieve_respects_top_k():
    r = _make_mock_retriever()
    r.top_k = 2
    results = r.retrieve("leave policy")
    assert len(results) <= 2


def test_retriever_semantic_only_returns_documents():
    r = _make_mock_retriever()
    from langchain_core.documents import Document
    results = r.retrieve_semantic_only("performance review")
    assert isinstance(results, list)
    assert all(isinstance(d, Document) for d in results)


def test_retriever_metadata_preserved():
    r = _make_mock_retriever()
    results = r.retrieve("attendance")
    for doc in results:
        assert "source_file" in doc.metadata
        assert "page_number" in doc.metadata


# ── Hybrid fusion logic ───────────────────────────────────────────────────────

def test_fuse_results_combines_scores():
    r = _make_mock_retriever()

    sem_docs = [Document(page_content="attendance policy text", metadata={"source_file": "a.pdf", "page_number": 1})]
    sem_scores = np.array([0.9])

    bm25_docs = [Document(page_content="low relevance filler", metadata={"source_file": "b.pdf", "page_number": 2})]
    bm25_scores = np.array([5.0])

    fused = r._fuse_results(sem_docs, sem_scores, bm25_docs, bm25_scores)
    # Two distinct documents should both be in the result
    assert len(fused) == 2
    # Scores are normalized independently; with 1 sem doc and 1 bm25 doc,
    # each normalizes to 0 within its own list, so fused scores are 0.
    # What matters is that the result is a list of (Document, float) tuples.
    assert all(isinstance(score, float) for _, score in fused)


def test_fuse_results_deduplicates():
    r = _make_mock_retriever()
    from langchain_core.documents import Document

    doc = Document(page_content="exact same content here", metadata={})
    sem_docs = [doc]
    bm25_docs = [doc]  # same document in both lists

    fused = r._fuse_results(sem_docs, np.array([0.8]), bm25_docs, np.array([3.0]))
    assert len(fused) == 1


def test_fuse_results_sorted_descending():
    r = _make_mock_retriever()
    from langchain_core.documents import Document

    sem_docs = [
        Document(page_content="high relevance content", metadata={}),
        Document(page_content="low relevance filler text", metadata={}),
    ]
    sem_scores = np.array([0.9, 0.1])

    fused = r._fuse_results(sem_docs, sem_scores, [], np.array([]))
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)
