"""
Unit tests for RAG generation metrics.

Covers all 4 generation metric functions:
- answer_relevance_heuristic
- answer_faithfulness_scaffold
- context_utilisation_heuristic
- answer_completeness_scaffold

Tests exercise: heuristic correctness, LLM-judge JSON parsing, edge cases
(empty inputs, malformed JSON, missing score field), and threshold pass/fail.
"""

from __future__ import annotations

import json
import pytest

from aieval360.evaluation.rag.models import RAGInput, RetrievedDocument
from aieval360.evaluation.rag.generation_metrics import (
    answer_completeness_scaffold,
    answer_faithfulness_scaffold,
    answer_relevance_heuristic,
    context_utilisation_heuristic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_docs(*contents: str) -> list[RetrievedDocument]:
    return [
        RetrievedDocument(doc_id=f"doc{i}", content=c, rank=i + 1)
        for i, c in enumerate(contents)
    ]


def judge_json(score: float, reasoning: str = "test reasoning") -> str:
    return json.dumps({"score": score, "reasoning": reasoning})


# ---------------------------------------------------------------------------
# answer_relevance_heuristic
# ---------------------------------------------------------------------------

class TestAnswerRelevanceHeuristic:
    def test_exact_overlap(self):
        inp = RAGInput(
            query="what is machine learning",
            generated_answer="machine learning is a subfield of AI",
        )
        out = answer_relevance_heuristic(inp)
        assert out.metric_name == "answer_relevance_heuristic"
        assert out.value is not None
        assert 0.0 <= out.value <= 1.0

    def test_no_overlap_gives_low_score(self):
        inp = RAGInput(
            query="quantum physics",
            generated_answer="the weather is sunny today",
        )
        out = answer_relevance_heuristic(inp)
        assert out.value is not None
        assert out.value < 0.2

    def test_full_overlap_gives_one(self):
        # Identical text → Jaccard = 1.0
        text = "machine learning model training"
        inp = RAGInput(query=text, generated_answer=text)
        out = answer_relevance_heuristic(inp)
        assert out.value == pytest.approx(1.0)

    def test_empty_query_returns_error(self):
        inp = RAGInput(query="", generated_answer="some answer")
        out = answer_relevance_heuristic(inp)
        assert out.value is None
        assert out.error is not None

    def test_empty_answer_returns_zero(self):
        inp = RAGInput(query="what is AI", generated_answer="")
        out = answer_relevance_heuristic(inp)
        assert out.value == pytest.approx(0.0)

    def test_none_answer_returns_zero(self):
        inp = RAGInput(query="what is AI", generated_answer=None)
        out = answer_relevance_heuristic(inp)
        assert out.value == pytest.approx(0.0)

    def test_threshold_pass(self):
        inp = RAGInput(query="machine learning", generated_answer="machine learning models")
        out = answer_relevance_heuristic(inp, threshold=0.1)
        assert out.passed is True

    def test_threshold_fail(self):
        inp = RAGInput(query="quantum", generated_answer="sunny day")
        out = answer_relevance_heuristic(inp, threshold=0.9)
        assert out.passed is False

    def test_details_populated(self):
        inp = RAGInput(query="machine learning", generated_answer="learning machines")
        out = answer_relevance_heuristic(inp)
        assert "query_tokens" in out.details
        assert "answer_tokens" in out.details
        assert "intersection" in out.details

    def test_case_insensitive(self):
        inp1 = RAGInput(query="Machine Learning", generated_answer="MACHINE LEARNING IS GREAT")
        inp2 = RAGInput(query="machine learning", generated_answer="machine learning is great")
        out1 = answer_relevance_heuristic(inp1)
        out2 = answer_relevance_heuristic(inp2)
        assert out1.value == pytest.approx(out2.value)


# ---------------------------------------------------------------------------
# answer_faithfulness_scaffold
# ---------------------------------------------------------------------------

class TestAnswerFaithfulnessScaffold:
    def test_valid_judge_response(self):
        inp = RAGInput(
            query="What is RAG?",
            generated_answer="RAG uses retrieved docs",
            judge_response=judge_json(0.9, "All claims grounded in context."),
        )
        out = answer_faithfulness_scaffold(inp)
        assert out.metric_name == "answer_faithfulness"
        assert out.value == pytest.approx(0.9)
        assert out.reasoning == "All claims grounded in context."

    def test_low_score_propagated(self):
        inp = RAGInput(
            query="q", generated_answer="a",
            judge_response=judge_json(0.2, "Claims not supported."),
        )
        out = answer_faithfulness_scaffold(inp)
        assert out.value == pytest.approx(0.2)

    def test_no_judge_response_returns_error(self):
        inp = RAGInput(query="q", generated_answer="a")
        out = answer_faithfulness_scaffold(inp)
        assert out.value is None
        assert out.error is not None

    def test_empty_judge_response_returns_error(self):
        inp = RAGInput(query="q", generated_answer="a", judge_response="")
        out = answer_faithfulness_scaffold(inp)
        assert out.value is None
        assert out.error is not None

    def test_malformed_json_returns_error(self):
        inp = RAGInput(query="q", generated_answer="a", judge_response="{not json}")
        out = answer_faithfulness_scaffold(inp)
        assert out.value is None
        assert out.error is not None

    def test_missing_score_field_returns_error(self):
        inp = RAGInput(
            query="q", generated_answer="a",
            judge_response=json.dumps({"reasoning": "no score here"}),
        )
        out = answer_faithfulness_scaffold(inp)
        assert out.value is None
        assert out.error is not None

    def test_non_numeric_score_returns_error(self):
        inp = RAGInput(
            query="q", generated_answer="a",
            judge_response=json.dumps({"score": "high", "reasoning": "good"}),
        )
        out = answer_faithfulness_scaffold(inp)
        assert out.value is None
        assert out.error is not None

    def test_threshold_pass(self):
        inp = RAGInput(
            query="q", generated_answer="a",
            judge_response=judge_json(0.85),
        )
        out = answer_faithfulness_scaffold(inp, threshold=0.7)
        assert out.passed is True

    def test_threshold_fail(self):
        inp = RAGInput(
            query="q", generated_answer="a",
            judge_response=judge_json(0.5),
        )
        out = answer_faithfulness_scaffold(inp, threshold=0.7)
        assert out.passed is False

    def test_details_populated(self):
        inp = RAGInput(
            query="q", generated_answer="a",
            judge_response=judge_json(0.8),
        )
        out = answer_faithfulness_scaffold(inp)
        assert "judge_response_length" in out.details
        assert out.details["judge_response_length"] > 0


# ---------------------------------------------------------------------------
# context_utilisation_heuristic
# ---------------------------------------------------------------------------

class TestContextUtilisationHeuristic:
    def test_answer_uses_context_words(self):
        inp = RAGInput(
            query="What is transfer learning?",
            retrieved_docs=make_docs("Transfer learning is a machine learning technique"),
            generated_answer="Transfer learning is used in machine learning",
        )
        out = context_utilisation_heuristic(inp)
        assert out.metric_name == "context_utilisation_heuristic"
        assert out.value is not None
        assert out.value > 0.0

    def test_answer_ignores_context(self):
        inp = RAGInput(
            query="q",
            retrieved_docs=make_docs("quantum physics nuclear energy"),
            generated_answer="sunny weather today is great",
        )
        out = context_utilisation_heuristic(inp)
        assert out.value is not None
        assert out.value < 0.1

    def test_no_docs_returns_error(self):
        inp = RAGInput(query="q", generated_answer="some answer")
        out = context_utilisation_heuristic(inp)
        assert out.value is None
        assert out.error is not None

    def test_empty_answer_returns_zero(self):
        inp = RAGInput(
            query="q",
            retrieved_docs=make_docs("context content here"),
            generated_answer="",
        )
        out = context_utilisation_heuristic(inp)
        assert out.value == pytest.approx(0.0)

    def test_none_answer_returns_zero(self):
        inp = RAGInput(
            query="q",
            retrieved_docs=make_docs("context content here"),
            generated_answer=None,
        )
        out = context_utilisation_heuristic(inp)
        assert out.value == pytest.approx(0.0)

    def test_threshold_pass(self):
        inp = RAGInput(
            query="q",
            retrieved_docs=make_docs("machine learning deep learning neural networks"),
            generated_answer="machine learning and deep learning are AI techniques",
        )
        out = context_utilisation_heuristic(inp, threshold=0.1)
        assert out.passed is True

    def test_threshold_fail(self):
        inp = RAGInput(
            query="q",
            retrieved_docs=make_docs("quantum physics"),
            generated_answer="sunny day",
            # no overlap
        )
        out = context_utilisation_heuristic(inp, threshold=0.5)
        assert out.passed is False

    def test_multiple_docs_combined(self):
        inp = RAGInput(
            query="q",
            retrieved_docs=make_docs("alpha beta", "gamma delta"),
            generated_answer="alpha gamma",
        )
        out = context_utilisation_heuristic(inp)
        # context tokens: {alpha, beta, gamma, delta} = 4
        # answer tokens: {alpha, gamma} = 2 → overlap = 2
        # value = 2/4 = 0.5
        assert out.value == pytest.approx(0.5)

    def test_details_populated(self):
        inp = RAGInput(
            query="q",
            retrieved_docs=make_docs("hello world"),
            generated_answer="hello",
        )
        out = context_utilisation_heuristic(inp)
        assert "context_tokens" in out.details
        assert "answer_tokens" in out.details
        assert "num_docs" in out.details


# ---------------------------------------------------------------------------
# answer_completeness_scaffold
# ---------------------------------------------------------------------------

class TestAnswerCompletenessScaffold:
    def test_valid_judge_response(self):
        inp = RAGInput(
            query="Explain RAG",
            generated_answer="RAG retrieves docs then generates an answer",
            judge_response=judge_json(0.75, "Covers retrieval but misses generation detail."),
        )
        out = answer_completeness_scaffold(inp)
        assert out.metric_name == "answer_completeness"
        assert out.value == pytest.approx(0.75)
        assert "generation detail" in (out.reasoning or "")

    def test_no_judge_response_returns_error(self):
        inp = RAGInput(query="q", generated_answer="a")
        out = answer_completeness_scaffold(inp)
        assert out.value is None
        assert out.error is not None

    def test_malformed_json_returns_error(self):
        inp = RAGInput(
            query="q", generated_answer="a",
            judge_response="<not json>",
        )
        out = answer_completeness_scaffold(inp)
        assert out.value is None
        assert out.error is not None

    def test_zero_score(self):
        inp = RAGInput(
            query="q", generated_answer="a",
            judge_response=judge_json(0.0, "Answer says nothing relevant."),
        )
        out = answer_completeness_scaffold(inp)
        assert out.value == pytest.approx(0.0)

    def test_full_score(self):
        inp = RAGInput(
            query="q", generated_answer="a",
            judge_response=judge_json(1.0, "Completely answered."),
        )
        out = answer_completeness_scaffold(inp)
        assert out.value == pytest.approx(1.0)

    def test_threshold_pass(self):
        inp = RAGInput(
            query="q", generated_answer="a",
            judge_response=judge_json(0.8),
        )
        out = answer_completeness_scaffold(inp, threshold=0.6)
        assert out.passed is True

    def test_threshold_fail(self):
        inp = RAGInput(
            query="q", generated_answer="a",
            judge_response=judge_json(0.4),
        )
        out = answer_completeness_scaffold(inp, threshold=0.6)
        assert out.passed is False

    def test_details_populated(self):
        inp = RAGInput(
            query="q", generated_answer="a",
            judge_response=judge_json(0.7),
        )
        out = answer_completeness_scaffold(inp)
        assert "judge_response_length" in out.details
