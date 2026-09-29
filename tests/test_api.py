"""
test_api.py — Unit tests for FastAPI backend (api.py).
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api
from api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Enterprise Document RAG Chatbot API" in data["name"]
    assert data["docs_url"] == "/docs"
    assert data["health_url"] == "/health"


def test_health_endpoint():
    with patch("api.get_retriever") as mock_get_ret:
        mock_ret = MagicMock()
        mock_ret.get_corpus_size.return_value = 21
        mock_get_ret.return_value = mock_ret

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "chroma_ready" in data
        assert "indexed_chunks" in data
        assert data["indexed_chunks"] == 21


def test_query_empty_question_fails():
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_query_grounded_answer():
    mock_result = {
        "answer": "Raj values speed and simplicity.\n\nSources:\n- RAGG.pdf — Page 3",
        "sources": [
            {
                "source_file": "RAGG.pdf",
                "page_number": 3,
                "snippet": "Raj values speed and simplicity over configurability…",
            }
        ],
        "retrieved_chunks": 5,
    }

    with patch("api.get_retriever") as mock_get_ret, patch("chain.ask", return_value=mock_result):
        mock_get_ret.return_value = MagicMock()
        response = client.post(
            "/query",
            json={
                "question": "What does Raj value?",
                "conversation_history": [],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What does Raj value?"
        assert "Raj values speed" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source_file"] == "RAGG.pdf"
        assert data["sources"][0]["page_number"] == 3
        assert data["retrieved_chunks"] == 5


def test_query_refusal_returns_empty_sources():
    mock_result = {
        "answer": "I don't have enough information in the provided documents to answer that.",
        "sources": [],
        "retrieved_chunks": 5,
    }

    with patch("api.get_retriever") as mock_get_ret, patch("chain.ask", return_value=mock_result):
        mock_get_ret.return_value = MagicMock()
        response = client.post(
            "/query",
            json={"question": "What is the capital of France?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "don't have enough information" in data["answer"]
        assert data["sources"] == []


def test_list_documents_endpoint():
    with patch("api.get_retriever") as mock_get_ret:
        mock_ret = MagicMock()
        mock_ret.get_corpus_size.return_value = 21
        mock_get_ret.return_value = mock_ret

        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total_documents" in data
        assert data["indexed_chunks"] == 21
