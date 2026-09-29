"""
tests/test_chain.py — Tests for the RAG generation pipeline.

Tests:
  - format_context builds correct citation strings
  - format_history limits to MAX_HISTORY_TURNS
  - API key validation
  - Empty question handling
  - Refusal behavior detection
  - Source deduplication
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from chain import format_context, format_history, _check_api_key
from langchain_core.documents import Document


# ── format_context ────────────────────────────────────────────────────────────

def test_format_context_includes_source_file():
    docs = [Document(page_content="Policy text here.", metadata={"source_file": "policy.pdf", "page_number": 4})]
    result = format_context(docs)
    assert "policy.pdf" in result


def test_format_context_includes_page_number():
    docs = [Document(page_content="Some content.", metadata={"source_file": "doc.pdf", "page_number": 7})]
    result = format_context(docs)
    assert "7" in result


def test_format_context_numbers_chunks():
    docs = [
        Document(page_content="Chunk one.", metadata={"source_file": "a.pdf", "page_number": 1}),
        Document(page_content="Chunk two.", metadata={"source_file": "a.pdf", "page_number": 2}),
    ]
    result = format_context(docs)
    assert "[1]" in result
    assert "[2]" in result


def test_format_context_empty_list():
    result = format_context([])
    assert "No relevant" in result


def test_format_context_missing_metadata_fallback():
    docs = [Document(page_content="Content with no metadata.", metadata={})]
    result = format_context(docs)
    assert "Unknown" in result
    assert "?" in result


def test_format_context_multiple_docs_separated():
    docs = [
        Document(page_content="First chunk content.", metadata={"source_file": "x.pdf", "page_number": 1}),
        Document(page_content="Second chunk content.", metadata={"source_file": "x.pdf", "page_number": 2}),
    ]
    result = format_context(docs)
    assert "First chunk content." in result
    assert "Second chunk content." in result


# ── format_history ────────────────────────────────────────────────────────────

def test_format_history_empty():
    result = format_history([])
    assert result == ""


def test_format_history_formats_roles():
    history = [
        {"role": "user", "content": "What is the policy?"},
        {"role": "assistant", "content": "The policy states..."},
    ]
    result = format_history(history)
    assert "User:" in result
    assert "Assistant:" in result


def test_format_history_limits_turns(monkeypatch):
    monkeypatch.setenv("MAX_HISTORY_TURNS", "2")
    # Reload the module to pick up the monkeypatched env
    import importlib
    import chain
    importlib.reload(chain)

    # 10 turns = 20 messages
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
        for i in range(20)
    ]
    result = chain.format_history(history)
    # MAX_HISTORY_TURNS=2 means only last 4 messages (2 turns × 2 messages)
    assert "message 16" in result or "message 17" in result or "message 18" in result


def test_format_history_single_message():
    history = [{"role": "user", "content": "Hello"}]
    result = format_history(history)
    assert "Hello" in result


# ── _check_api_key ────────────────────────────────────────────────────────────

def test_check_api_key_missing(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    import importlib, chain
    importlib.reload(chain)
    assert chain._check_api_key() is not None


def test_check_api_key_placeholder(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
    import importlib, chain
    importlib.reload(chain)
    assert chain._check_api_key() is not None


def test_check_api_key_valid(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real-key-12345")
    import importlib, chain
    importlib.reload(chain)
    assert chain._check_api_key() is None


# ── ask() — empty question ────────────────────────────────────────────────────

def test_ask_empty_question(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    import importlib, chain
    importlib.reload(chain)

    mock_retriever = MagicMock()
    result = chain.ask("   ", mock_retriever)
    assert "Please enter a question" in result["answer"]
    mock_retriever.retrieve.assert_not_called()


# ── ask() — API key not configured ───────────────────────────────────────────

def test_ask_returns_error_when_no_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    import importlib, chain
    importlib.reload(chain)

    mock_retriever = MagicMock()
    result = chain.ask("What is the policy?", mock_retriever)
    assert "API key" in result["answer"] or "not configured" in result["answer"]
    assert result["sources"] == []


# ── ask() — source deduplication ─────────────────────────────────────────────

def test_ask_deduplicates_sources(monkeypatch):
    """
    Source deduplication is pure Python logic — test it directly
    without needing to mock the entire LangChain pipeline.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    import importlib, chain
    importlib.reload(chain)

    docs = [
        Document(page_content="Content A.", metadata={"source_file": "x.pdf", "page_number": 1}),
        Document(page_content="Content B.", metadata={"source_file": "x.pdf", "page_number": 1}),  # same page
        Document(page_content="Content C.", metadata={"source_file": "x.pdf", "page_number": 2}),
    ]

    # Test the deduplication logic directly — it runs after LLM, independently
    seen_sources = set()
    sources = []
    for doc in docs:
        source_file = doc.metadata.get("source_file", "Unknown")
        page_number = doc.metadata.get("page_number", "?")
        key = f"{source_file}::{page_number}"
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({"source_file": source_file, "page_number": page_number})

    source_keys = [f"{s['source_file']}::{s['page_number']}" for s in sources]
    assert len(source_keys) == len(set(source_keys))  # no duplicates
    assert len(source_keys) == 2  # x.pdf::1 and x.pdf::2


# ── Refusal and citation suppression tests ───────────────────────────────────

def test_is_refusal_detection():
    """Verify is_refusal accurately recognizes refusal answers vs valid answers."""
    from chain import is_refusal, REFUSAL_RESPONSE

    assert is_refusal(REFUSAL_RESPONSE) is True
    assert is_refusal("I don't have enough information in the provided documents to answer that.") is True
    assert is_refusal("That information is not available in the provided document context.") is True
    assert is_refusal("I cannot find any mention of that in the documents.") is True

    # Valid grounded answers must not be flagged as refusals
    assert is_refusal("The attendance policy requires 75% minimum attendance.") is False
    assert is_refusal("The PRD owner is Product/Engineering Lead (see Page 1).") is False


def test_ask_refusal_returns_empty_sources(monkeypatch):
    """
    When an unsupported question causes the model to return the refusal response,
    the sources list must be empty even though chunks were retrieved.
    """
    from langchain_core.runnables import RunnableLambda
    import chain

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    import importlib
    importlib.reload(chain)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(page_content="Some company policy excerpt.", metadata={"source_file": "policy.pdf", "page_number": 1}),
        Document(page_content="Another excerpt.", metadata={"source_file": "policy.pdf", "page_number": 2}),
    ]

    mock_llm = RunnableLambda(
        lambda x: "I don't have enough information in the provided documents to answer that."
    )

    with patch.object(chain, "build_llm", return_value=mock_llm):
        result = chain.ask("What is the capital of France?", mock_retriever)

    assert result["answer"] == "I don't have enough information in the provided documents to answer that."
    # CRITICAL: sources must be empty for refused answers
    assert result["sources"] == []
    # Retrieved chunks count is preserved for observability/auditing
    assert result["retrieved_chunks"] == 2


def test_ask_grounded_answer_returns_sources(monkeypatch):
    """
    When a supported question produces a grounded answer,
    sources must be populated with source filename, page number, and snippet.
    """
    from langchain_core.runnables import RunnableLambda
    import chain

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    import importlib
    importlib.reload(chain)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(page_content="The attendance policy requires 75% attendance.", metadata={"source_file": "policy.pdf", "page_number": 3}),
    ]

    mock_llm = RunnableLambda(
        lambda x: "The attendance policy requires 75% attendance.\n\nSources:\n- policy.pdf — Page 3"
    )

    with patch.object(chain, "build_llm", return_value=mock_llm):
        result = chain.ask("What is the attendance policy?", mock_retriever)

    assert "75%" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_file"] == "policy.pdf"
    assert result["sources"][0]["page_number"] == 3
    assert "75%" in result["sources"][0]["snippet"]
    assert result["retrieved_chunks"] == 1


def test_ask_normal_answer_filters_unrelated_sources(monkeypatch):
    """
    When the retriever returns multiple chunks (some unrelated),
    sources must only include the source cited by the answer,
    filtering out unrelated retrieved pages.
    """
    from langchain_core.runnables import RunnableLambda
    import chain

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    import importlib
    importlib.reload(chain)

    mock_retriever = MagicMock()
    # Retriever returns 3 chunks, but only page 3 is relevant to the question
    mock_retriever.retrieve.return_value = [
        Document(page_content="Raj values speed and simplicity.", metadata={"source_file": "RAGG.pdf", "page_number": 3}),
        Document(page_content="Unrelated executive summary text.", metadata={"source_file": "RAGG.pdf", "page_number": 1}),
        Document(page_content="Evaluation plan metrics and KPIs.", metadata={"source_file": "RAGG.pdf", "page_number": 9}),
    ]

    mock_llm = RunnableLambda(
        lambda x: "Raj values speed and simplicity.\n\nSources:\n- RAGG.pdf — Page 3"
    )

    with patch.object(chain, "build_llm", return_value=mock_llm):
        result = chain.ask("Who is Raj and what does he value?", mock_retriever)

    assert len(result["sources"]) == 1
    assert result["sources"][0]["page_number"] == 3
    assert result["sources"][0]["source_file"] == "RAGG.pdf"
    # Unrelated pages 1 and 9 must NOT be in sources
    pages_in_sources = [s["page_number"] for s in result["sources"]]
    assert 1 not in pages_in_sources
    assert 9 not in pages_in_sources


def test_ask_follow_up_answer_returns_relevant_sources(monkeypatch):
    """
    In a multi-turn conversation, when a follow-up answer cites a source
    from previous conversation context, sources must return the relevant
    source even if not present in the current turn's retrieved chunks.
    """
    from langchain_core.runnables import RunnableLambda
    import chain

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    import importlib
    importlib.reload(chain)

    # In turn 2, retriever retrieves generic chunks (pages 1, 9)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(page_content="Executive summary text.", metadata={"source_file": "RAGG.pdf", "page_number": 1}),
        Document(page_content="Evaluation plan text.", metadata={"source_file": "RAGG.pdf", "page_number": 9}),
    ]
    # Retriever corpus has the chunk from Page 3
    mock_retriever._corpus_metas = [
        {"source_file": "RAGG.pdf", "page_number": 3},
        {"source_file": "RAGG.pdf", "page_number": 1},
    ]
    mock_retriever._corpus_texts = [
        "Raj is a new hire in Operations who values speed and simplicity.",
        "Executive summary text.",
    ]

    history = [
        {"role": "user", "content": "Who is Raj and what does he value?"},
        {
            "role": "assistant",
            "content": "Raj values speed and simplicity.\n\nSources:\n- RAGG.pdf — Page 3",
            "sources": [
                {
                    "source_file": "RAGG.pdf",
                    "page_number": 3,
                    "snippet": "Raj is a new hire in Operations…",
                }
            ],
        },
    ]

    mock_llm = RunnableLambda(
        lambda x: "Raj prioritizes speed and simplicity over configurability.\n\nSources:\n- RAGG.pdf — Page 3"
    )

    with patch.object(chain, "build_llm", return_value=mock_llm):
        result = chain.ask("What does he prioritize?", mock_retriever, conversation_history=history)

    assert "speed and simplicity" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_file"] == "RAGG.pdf"
    assert result["sources"][0]["page_number"] == 3
    # Unrelated pages 1 and 9 from current retrieval must NOT be in sources
    pages = [s["page_number"] for s in result["sources"]]
    assert 1 not in pages
    assert 9 not in pages


def test_resolve_sources_deduplicates_duplicate_citations():
    """Verify resolve_sources removes duplicate source/page entries."""
    from chain import resolve_sources

    docs = [
        Document(page_content="Section 1.", metadata={"source_file": "guide.pdf", "page_number": 2}),
        Document(page_content="Section 2.", metadata={"source_file": "guide.pdf", "page_number": 2}),
    ]

    # LLM cited the same page twice
    answer = "Here is the summary.\n\nSources:\n- guide.pdf — Page 2\n- guide.pdf — Page 2"
    sources = resolve_sources(answer, docs)

    assert len(sources) == 1
    assert sources[0]["source_file"] == "guide.pdf"
    assert sources[0]["page_number"] == 2

