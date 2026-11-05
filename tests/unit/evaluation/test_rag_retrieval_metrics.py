"""
Unit tests for RAG retrieval metrics.

Covers all 7 retrieval metric functions:
- precision_at_k
- recall_at_k
- mean_reciprocal_rank
- ndcg_at_k
- hit_at_k
- context_precision  (alias)
- context_recall     (alias)

Tests exercise: normal cases, edge cases (empty inputs, no relevant docs,
all relevant), mathematical correctness, and threshold pass/fail logic.
"""

from __future__ import annotations

import math
import pytest

from aieval360.evaluation.rag.models import RAGInput, RetrievedDocument
from aieval360.evaluation.rag.retrieval_metrics import (
    context_precision,
    context_recall,
    hit_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_docs(*doc_ids: str) -> list[RetrievedDocument]:
    """Build a ranked list of RetrievedDocument objects."""
    return [
        RetrievedDocument(doc_id=d, content=f"content of {d}", score=1.0 - i * 0.1, rank=i + 1)
        for i, d in enumerate(doc_ids)
    ]


def make_input(
    retrieved: list[str],
    relevant: set[str],
    query: str = "test query",
) -> RAGInput:
    return RAGInput(
        query=query,
        retrieved_docs=make_docs(*retrieved),
        relevant_doc_ids=relevant,
    )


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_relevant(self):
        inp = make_input(["a", "b", "c"], {"a", "b", "c"})
        out = precision_at_k(inp, k=3)
        assert out.value == pytest.approx(1.0)
        assert out.metric_name == "precision_at_k"

    def test_none_relevant_in_top_k(self):
        inp = make_input(["a", "b", "c"], {"d", "e"})
        out = precision_at_k(inp, k=3)
        assert out.value == pytest.approx(0.0)

    def test_half_relevant(self):
        inp = make_input(["a", "b", "c", "d"], {"a", "c"})
        out = precision_at_k(inp, k=4)
        assert out.value == pytest.approx(0.5)

    def test_k_larger_than_retrieved(self):
        # k=10 but only 3 docs — denominator is still k
        inp = make_input(["a", "b", "c"], {"a", "b"})
        out = precision_at_k(inp, k=10)
        assert out.value == pytest.approx(2 / 10)

    def test_empty_retrieved_docs(self):
        inp = make_input([], {"a"})
        out = precision_at_k(inp, k=5)
        assert out.value == pytest.approx(0.0)

    def test_empty_relevant_ids_returns_none(self):
        inp = make_input(["a", "b"], set())
        out = precision_at_k(inp)
        assert out.value is None
        assert out.error is not None

    def test_threshold_pass(self):
        inp = make_input(["a", "b", "c"], {"a", "b", "c"})
        out = precision_at_k(inp, k=3, threshold=0.9)
        assert out.passed is True

    def test_threshold_fail(self):
        inp = make_input(["a", "b", "c"], {"a"})
        out = precision_at_k(inp, k=3, threshold=0.9)
        assert out.passed is False

    def test_no_threshold_passed_is_none(self):
        inp = make_input(["a"], {"a"})
        out = precision_at_k(inp, k=1)
        assert out.threshold is None
        assert out.passed is None

    def test_details_populated(self):
        inp = make_input(["a", "b", "c"], {"a"})
        out = precision_at_k(inp, k=3)
        assert "hits" in out.details
        assert out.details["hits"] == 1
        assert out.details["k"] == 3


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_all_relevant_found(self):
        inp = make_input(["a", "b", "c"], {"a", "b", "c"})
        out = recall_at_k(inp, k=3)
        assert out.value == pytest.approx(1.0)

    def test_no_relevant_found(self):
        inp = make_input(["x", "y", "z"], {"a", "b"})
        out = recall_at_k(inp, k=3)
        assert out.value == pytest.approx(0.0)

    def test_partial_recall(self):
        # 2 of 4 relevant found in top-3
        inp = make_input(["a", "b", "x"], {"a", "b", "c", "d"})
        out = recall_at_k(inp, k=3)
        assert out.value == pytest.approx(0.5)

    def test_empty_retrieved(self):
        inp = make_input([], {"a"})
        out = recall_at_k(inp, k=5)
        assert out.value == pytest.approx(0.0)

    def test_empty_relevant_returns_none(self):
        inp = make_input(["a"], set())
        out = recall_at_k(inp)
        assert out.value is None
        assert out.error is not None

    def test_k_less_than_all_relevant(self):
        inp = make_input(["a", "b", "c", "d", "e"], {"a", "b", "c", "d", "e"})
        out = recall_at_k(inp, k=3)
        assert out.value == pytest.approx(3 / 5)

    def test_details_populated(self):
        inp = make_input(["a", "b"], {"a", "b", "c"})
        out = recall_at_k(inp, k=2)
        assert out.details["n_relevant"] == 3
        assert out.details["hits"] == 2


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------

class TestMeanReciprocalRank:
    def test_first_doc_relevant(self):
        inp = make_input(["a", "b", "c"], {"a"})
        out = mean_reciprocal_rank(inp)
        assert out.value == pytest.approx(1.0)

    def test_second_doc_relevant(self):
        inp = make_input(["x", "a", "b"], {"a"})
        out = mean_reciprocal_rank(inp)
        assert out.value == pytest.approx(0.5)

    def test_third_doc_relevant(self):
        inp = make_input(["x", "y", "a"], {"a"})
        out = mean_reciprocal_rank(inp)
        assert out.value == pytest.approx(1 / 3)

    def test_no_relevant_found(self):
        inp = make_input(["x", "y", "z"], {"a"})
        out = mean_reciprocal_rank(inp)
        assert out.value == pytest.approx(0.0)

    def test_empty_retrieved(self):
        inp = make_input([], {"a"})
        out = mean_reciprocal_rank(inp)
        assert out.value == pytest.approx(0.0)

    def test_empty_relevant_returns_none(self):
        inp = make_input(["a"], set())
        out = mean_reciprocal_rank(inp)
        assert out.value is None

    def test_multiple_relevant_uses_first(self):
        # First relevant is at rank 2; rank 1 is not relevant
        inp = make_input(["x", "a", "b"], {"a", "b"})
        out = mean_reciprocal_rank(inp)
        assert out.value == pytest.approx(0.5)

    def test_details_first_rank(self):
        inp = make_input(["x", "a"], {"a"})
        out = mean_reciprocal_rank(inp)
        assert out.details["first_relevant_rank"] == 2


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

class TestNdcgAtK:
    def test_perfect_ranking(self):
        # All relevant docs at the top → NDCG = 1.0
        inp = make_input(["a", "b", "c", "x", "y"], {"a", "b", "c"})
        out = ndcg_at_k(inp, k=5)
        assert out.value == pytest.approx(1.0)

    def test_zero_ndcg_no_relevant_in_top_k(self):
        inp = make_input(["x", "y", "z"], {"a", "b"})
        out = ndcg_at_k(inp, k=3)
        assert out.value == pytest.approx(0.0)

    def test_dcg_decreases_with_rank(self):
        # doc at rank 1 → gains log2(2)=1; doc at rank 3 → gains log2(4)=2 (less)
        # Only first doc is relevant
        inp_rank1 = make_input(["a", "x", "y"], {"a"})
        inp_rank3 = make_input(["x", "y", "a"], {"a"})
        out1 = ndcg_at_k(inp_rank1, k=3)
        out3 = ndcg_at_k(inp_rank3, k=3)
        assert out1.value > out3.value

    def test_empty_retrieved(self):
        inp = make_input([], {"a"})
        out = ndcg_at_k(inp, k=5)
        assert out.value == pytest.approx(0.0)

    def test_empty_relevant_returns_none(self):
        inp = make_input(["a"], set())
        out = ndcg_at_k(inp)
        assert out.value is None

    def test_partial_ndcg_value_in_range(self):
        inp = make_input(["x", "a", "b"], {"a", "b"})
        out = ndcg_at_k(inp, k=3)
        # Must be between 0 and 1
        assert 0.0 <= out.value <= 1.0

    def test_details_contain_dcg_idcg(self):
        inp = make_input(["a", "b"], {"a", "b"})
        out = ndcg_at_k(inp, k=2)
        assert "dcg" in out.details
        assert "idcg" in out.details
        assert out.details["idcg"] > 0

    def test_ndcg_known_value(self):
        # Retrieved: [a(rel), x, b(rel)], k=3
        # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0 + 0.5 = 1.5
        # Ideal: [a(rel), b(rel), x], DCG_ideal = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
        # NDCG = 1.5 / 1.6309 ≈ 0.9196
        inp = make_input(["a", "x", "b"], {"a", "b"})
        out = ndcg_at_k(inp, k=3)
        dcg = 1.0 / math.log2(2) + 0 + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = dcg / idcg
        assert out.value == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------

class TestHitAtK:
    def test_hit_first_doc(self):
        inp = make_input(["a", "b", "c"], {"a"})
        out = hit_at_k(inp, k=3)
        assert out.value == pytest.approx(1.0)

    def test_no_hit(self):
        inp = make_input(["x", "y", "z"], {"a"})
        out = hit_at_k(inp, k=3)
        assert out.value == pytest.approx(0.0)

    def test_hit_last_in_k(self):
        inp = make_input(["x", "y", "a"], {"a"})
        out = hit_at_k(inp, k=3)
        assert out.value == pytest.approx(1.0)

    def test_no_hit_just_outside_k(self):
        # doc at rank 4 when k=3
        inp = make_input(["x", "y", "z", "a"], {"a"})
        out = hit_at_k(inp, k=3)
        assert out.value == pytest.approx(0.0)

    def test_empty_retrieved(self):
        inp = make_input([], {"a"})
        out = hit_at_k(inp, k=5)
        assert out.value == pytest.approx(0.0)

    def test_empty_relevant_returns_none(self):
        inp = make_input(["a"], set())
        out = hit_at_k(inp)
        assert out.value is None

    def test_threshold_at_1_passes_on_hit(self):
        inp = make_input(["a"], {"a"})
        out = hit_at_k(inp, k=1, threshold=1.0)
        assert out.passed is True

    def test_threshold_at_1_fails_on_miss(self):
        inp = make_input(["x"], {"a"})
        out = hit_at_k(inp, k=1, threshold=1.0)
        assert out.passed is False


# ---------------------------------------------------------------------------
# context_precision / context_recall (aliases)
# ---------------------------------------------------------------------------

class TestContextAliases:
    def test_context_precision_name(self):
        inp = make_input(["a", "b"], {"a"})
        out = context_precision(inp, k=2)
        assert out.metric_name == "context_precision"

    def test_context_recall_name(self):
        inp = make_input(["a", "b"], {"a"})
        out = context_recall(inp, k=2)
        assert out.metric_name == "context_recall"

    def test_context_precision_value_matches_precision_at_k(self):
        inp = make_input(["a", "b", "x"], {"a", "b"})
        cp = context_precision(inp, k=3)
        pk = precision_at_k(inp, k=3)
        assert cp.value == pytest.approx(pk.value)

    def test_context_recall_value_matches_recall_at_k(self):
        inp = make_input(["a", "b", "x"], {"a", "b", "c"})
        cr = context_recall(inp, k=3)
        rk = recall_at_k(inp, k=3)
        assert cr.value == pytest.approx(rk.value)

    def test_context_precision_threshold(self):
        inp = make_input(["a", "b"], {"a", "b"})
        out = context_precision(inp, k=2, threshold=0.8)
        assert out.passed is True

    def test_context_recall_threshold(self):
        inp = make_input(["a"], {"a", "b"})
        out = context_recall(inp, k=1, threshold=0.9)
        assert out.passed is False
