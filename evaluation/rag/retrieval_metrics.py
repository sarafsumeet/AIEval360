"""
Retrieval quality metrics for RAG evaluation.

All metrics are DETERMINISTIC — they require only the ranked retrieved document
list and the set of known-relevant doc IDs; no LLM calls needed.

Implemented metrics
-------------------
- Precision@K        Fraction of top-K docs that are relevant
- Recall@K           Fraction of relevant docs found in top-K
- MRR               Mean Reciprocal Rank (rank of first relevant doc)
- NDCG@K            Normalised Discounted Cumulative Gain
- Hit@K             Binary: at least one relevant doc in top-K
- ContextPrecision  Alias for Precision@K (RAG-specific naming)
- ContextRecall     Alias for Recall@K  (RAG-specific naming)
"""

from __future__ import annotations

import math
from typing import Optional

from aieval360.evaluation.rag.models import RAGInput, RAGMetricOutput


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _top_k_docs(rag_input: RAGInput, k: int) -> list:
    """Return the top-K retrieved docs (already rank-ordered by position)."""
    return rag_input.retrieved_docs[:k]


def _relevance_flags(docs, relevant_ids: set[str]) -> list[int]:
    """Binary relevance vector (1 = relevant, 0 = not) for a doc list."""
    return [1 if d.doc_id in relevant_ids else 0 for d in docs]


def _make_output(
    metric_name: str,
    value: Optional[float],
    threshold: Optional[float] = None,
    details: Optional[dict] = None,
    error: Optional[str] = None,
) -> RAGMetricOutput:
    return RAGMetricOutput(
        metric_name=metric_name,
        value=value,
        threshold=threshold,
        details=details or {},
        error=error,
    )


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------

def precision_at_k(rag_input: RAGInput, k: int = 5, threshold: Optional[float] = None) -> RAGMetricOutput:
    """
    Precision@K = |relevant ∩ top-K| / K.

    Answers: "Of the K documents I showed the user, how many were actually useful?"
    """
    if not rag_input.relevant_doc_ids:
        return _make_output("precision_at_k", None, threshold, error="relevant_doc_ids is empty")
    if not rag_input.retrieved_docs:
        return _make_output("precision_at_k", 0.0, threshold, details={"k": k, "hits": 0})

    top_k = _top_k_docs(rag_input, k)
    flags = _relevance_flags(top_k, rag_input.relevant_doc_ids)
    hits = sum(flags)
    value = hits / k
    return _make_output(
        "precision_at_k", value, threshold,
        details={"k": k, "hits": hits, "retrieved": len(top_k), "relevance_flags": flags},
    )


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

def recall_at_k(rag_input: RAGInput, k: int = 5, threshold: Optional[float] = None) -> RAGMetricOutput:
    """
    Recall@K = |relevant ∩ top-K| / |relevant|.

    Answers: "What fraction of all relevant documents did I find in the top-K?"
    """
    n_relevant = len(rag_input.relevant_doc_ids)
    if n_relevant == 0:
        return _make_output("recall_at_k", None, threshold, error="relevant_doc_ids is empty")
    if not rag_input.retrieved_docs:
        return _make_output("recall_at_k", 0.0, threshold, details={"k": k, "hits": 0, "n_relevant": n_relevant})

    top_k = _top_k_docs(rag_input, k)
    flags = _relevance_flags(top_k, rag_input.relevant_doc_ids)
    hits = sum(flags)
    value = hits / n_relevant
    return _make_output(
        "recall_at_k", value, threshold,
        details={"k": k, "hits": hits, "n_relevant": n_relevant, "relevance_flags": flags},
    )


# ---------------------------------------------------------------------------
# MRR — Mean Reciprocal Rank
# ---------------------------------------------------------------------------

def mean_reciprocal_rank(rag_input: RAGInput, threshold: Optional[float] = None) -> RAGMetricOutput:
    """
    MRR = 1 / rank_of_first_relevant_doc.

    Returns 0.0 if no relevant document appears in the retrieved list.
    For a single query this is just RR (reciprocal rank); 'mean' applies
    when averaged across a query set at the run level.
    """
    if not rag_input.relevant_doc_ids:
        return _make_output("mrr", None, threshold, error="relevant_doc_ids is empty")
    if not rag_input.retrieved_docs:
        return _make_output("mrr", 0.0, threshold, details={"first_relevant_rank": None})

    for i, doc in enumerate(rag_input.retrieved_docs, start=1):
        if doc.doc_id in rag_input.relevant_doc_ids:
            return _make_output(
                "mrr", 1.0 / i, threshold,
                details={"first_relevant_rank": i, "total_retrieved": len(rag_input.retrieved_docs)},
            )
    return _make_output("mrr", 0.0, threshold, details={"first_relevant_rank": None})


# ---------------------------------------------------------------------------
# NDCG@K — Normalised Discounted Cumulative Gain
# ---------------------------------------------------------------------------

def _dcg(relevance_flags: list[int]) -> float:
    """Compute DCG for a binary relevance vector."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_flags))


def ndcg_at_k(rag_input: RAGInput, k: int = 5, threshold: Optional[float] = None) -> RAGMetricOutput:
    """
    NDCG@K = DCG@K / IDCG@K  (ideal = all relevant docs ranked first).

    Rewards ranking relevant docs higher in the list.
    """
    n_relevant = len(rag_input.relevant_doc_ids)
    if n_relevant == 0:
        return _make_output("ndcg_at_k", None, threshold, error="relevant_doc_ids is empty")
    if not rag_input.retrieved_docs:
        return _make_output("ndcg_at_k", 0.0, threshold, details={"k": k})

    top_k = _top_k_docs(rag_input, k)
    flags = _relevance_flags(top_k, rag_input.relevant_doc_ids)
    dcg = _dcg(flags)

    # Ideal: put all relevant docs at the top
    ideal_flags = [1] * min(n_relevant, k) + [0] * max(0, k - n_relevant)
    idcg = _dcg(ideal_flags)

    value = dcg / idcg if idcg > 0 else 0.0
    return _make_output(
        "ndcg_at_k", value, threshold,
        details={"k": k, "dcg": dcg, "idcg": idcg, "relevance_flags": flags},
    )


# ---------------------------------------------------------------------------
# Hit@K  (binary)
# ---------------------------------------------------------------------------

def hit_at_k(rag_input: RAGInput, k: int = 5, threshold: Optional[float] = None) -> RAGMetricOutput:
    """
    Hit@K = 1.0 if at least one relevant doc appears in top-K, else 0.0.

    A simple pass/fail gate useful as a mandatory gate in a RAG pipeline.
    """
    if not rag_input.relevant_doc_ids:
        return _make_output("hit_at_k", None, threshold, error="relevant_doc_ids is empty")
    if not rag_input.retrieved_docs:
        return _make_output("hit_at_k", 0.0, threshold, details={"k": k})

    top_k = _top_k_docs(rag_input, k)
    hit = any(d.doc_id in rag_input.relevant_doc_ids for d in top_k)
    return _make_output(
        "hit_at_k", 1.0 if hit else 0.0, threshold,
        details={"k": k, "hit": hit},
    )


# ---------------------------------------------------------------------------
# ContextPrecision / ContextRecall (RAG-specific aliases)
# ---------------------------------------------------------------------------

def context_precision(rag_input: RAGInput, k: int = 5, threshold: Optional[float] = None) -> RAGMetricOutput:
    """
    Context Precision = Precision@K renamed for RAG governance reports.

    Answers: "Were the retrieved context chunks actually relevant to the query?"
    """
    result = precision_at_k(rag_input, k=k, threshold=threshold)
    result.metric_name = "context_precision"
    return result


def context_recall(rag_input: RAGInput, k: int = 5, threshold: Optional[float] = None) -> RAGMetricOutput:
    """
    Context Recall = Recall@K renamed for RAG governance reports.

    Answers: "Did the retriever find enough of the relevant context?"
    """
    result = recall_at_k(rag_input, k=k, threshold=threshold)
    result.metric_name = "context_recall"
    return result
