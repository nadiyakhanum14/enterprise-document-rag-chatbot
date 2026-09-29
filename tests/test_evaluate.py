"""
tests/test_evaluate.py — Tests for the evaluation pipeline.

Tests:
  - Golden QA set loads and validates correctly
  - Refusal detection logic works
  - All required fields are present in QA set
  - NOT_IN_DOCS questions exist for refusal testing
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

GOLDEN_QA_PATH = Path(__file__).parent.parent / "evaluation" / "golden_qa_set.json"


# ── Golden QA set structure ───────────────────────────────────────────────────

def test_golden_qa_file_exists():
    assert GOLDEN_QA_PATH.exists(), "evaluation/golden_qa_set.json must exist"


def test_golden_qa_loads_as_list():
    with open(GOLDEN_QA_PATH) as f:
        data = json.load(f)
    assert isinstance(data, list)


def test_golden_qa_minimum_questions():
    with open(GOLDEN_QA_PATH) as f:
        data = json.load(f)
    assert len(data) >= 10, "Should have at least 10 evaluation questions"


def test_golden_qa_required_fields():
    with open(GOLDEN_QA_PATH) as f:
        data = json.load(f)
    required_fields = {"id", "question", "ground_truth", "answer_in_docs"}
    for item in data:
        missing = required_fields - set(item.keys())
        assert not missing, f"Question {item.get('id', '?')} missing fields: {missing}"


def test_golden_qa_has_refusal_questions():
    with open(GOLDEN_QA_PATH) as f:
        data = json.load(f)
    refusal_qs = [q for q in data if not q["answer_in_docs"]]
    assert len(refusal_qs) >= 1, "Must have at least one out-of-scope refusal test question"


def test_golden_qa_has_in_scope_questions():
    with open(GOLDEN_QA_PATH) as f:
        data = json.load(f)
    in_scope = [q for q in data if q["answer_in_docs"]]
    assert len(in_scope) >= 5, "Must have at least 5 answerable questions"


def test_golden_qa_unique_ids():
    with open(GOLDEN_QA_PATH) as f:
        data = json.load(f)
    ids = [q["id"] for q in data]
    assert len(ids) == len(set(ids)), "All question IDs must be unique"


def test_golden_qa_no_empty_questions():
    with open(GOLDEN_QA_PATH) as f:
        data = json.load(f)
    for item in data:
        assert item["question"].strip(), f"Question {item['id']} has empty question text"


def test_golden_qa_refusal_ground_truth_marker():
    """Out-of-scope questions should use NOT_IN_DOCS as ground truth."""
    with open(GOLDEN_QA_PATH) as f:
        data = json.load(f)
    for item in data:
        if not item["answer_in_docs"]:
            assert item["ground_truth"] == "NOT_IN_DOCS", (
                f"Question {item['id']} has answer_in_docs=False "
                f"but ground_truth is not 'NOT_IN_DOCS'"
            )


def test_golden_qa_source_and_page_metadata():
    """Verify in-scope questions have valid source and page citations."""
    with open(GOLDEN_QA_PATH) as f:
        data = json.load(f)
    for item in data:
        if item["answer_in_docs"]:
            assert item.get("expected_source") == "RAGG.pdf"
            assert isinstance(item.get("expected_page"), int)
            assert 1 <= item["expected_page"] <= 11
        else:
            assert item.get("expected_source") is None
            assert item.get("expected_page") is None


# ── Refusal detection logic ───────────────────────────────────────────────────

def test_refusal_detected_correct_phrase():
    from evaluate import evaluate_refusal_behavior

    results = [
        {
            "id": "Q001",
            "question": "What is the budget?",
            "answer": "I don't have enough information in the provided documents to answer that.",
            "answer_in_docs": False,
            "category": "not_in_docs",
        }
    ]
    report = evaluate_refusal_behavior(results)
    assert report["correct_refusals"] == 1
    assert report["refusal_rate"] == 1.0


def test_refusal_missed_when_llm_guesses():
    from evaluate import evaluate_refusal_behavior

    results = [
        {
            "id": "Q002",
            "question": "What is the CEO salary?",
            "answer": "The CEO earns approximately $500,000 per year.",  # hallucinated
            "answer_in_docs": False,
            "category": "not_in_docs",
        }
    ]
    report = evaluate_refusal_behavior(results)
    assert report["correct_refusals"] == 0
    assert report["refusal_rate"] == 0.0


def test_refusal_skips_in_scope_questions():
    from evaluate import evaluate_refusal_behavior

    results = [
        {
            "id": "Q003",
            "question": "What is the attendance policy?",
            "answer": "The attendance policy requires 75% attendance.",
            "answer_in_docs": True,
            "category": "factual",
        }
    ]
    report = evaluate_refusal_behavior(results)
    assert report["refusal_questions"] == 0


def test_refusal_mixed_results():
    from evaluate import evaluate_refusal_behavior

    results = [
        {
            "id": "Q001",
            "question": "Company CEO?",
            "answer": "I cannot find that information in the provided documents.",
            "answer_in_docs": False,
            "category": "not_in_docs",
        },
        {
            "id": "Q002",
            "question": "Budget?",
            "answer": "The budget is $1 million.",  # hallucinated - should be refused
            "answer_in_docs": False,
            "category": "not_in_docs",
        },
    ]
    report = evaluate_refusal_behavior(results)
    assert report["refusal_questions"] == 2
    assert report["correct_refusals"] == 1
    assert report["refusal_rate"] == 0.5
