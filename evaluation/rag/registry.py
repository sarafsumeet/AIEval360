"""
RAG metric registry.

Maps metric names to their callable functions and default configuration,
providing a central catalogue for all RAG evaluation metrics available
in AIEval360.

Design decisions
----------------
- Functions, not classes: RAG metrics are pure functions; the registry
  maps names to those functions and their default K / threshold values.
- Governance evidence: each entry carries metric_type and description so
  evaluation reports can include the full metric taxonomy for audit trails.
- Extendable: custom metrics can be registered via register().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class RAGMetricRegistration:
    """
    Registry entry for a single RAG metric function.

    name         Canonical metric name (matches key in registry).
    fn           Callable: (RAGInput, **kwargs) -> RAGMetricOutput.
    metric_type  Taxonomy: DETERMINISTIC | HEURISTIC | LLM_JUDGED.
    description  Human-readable description for governance reports.
    default_k    Default top-K cutoff (for K-based retrieval metrics).
    default_threshold  Default pass/fail threshold (None = informational).
    requires_judge  True if the metric needs judge_response pre-populated.
    """

    name: str
    fn: Callable[..., Any]
    metric_type: str
    description: str
    default_k: Optional[int] = None
    default_threshold: Optional[float] = None
    requires_judge: bool = False
    tags: list[str] = field(default_factory=list)


class RAGMetricRegistry:
    """
    Central catalogue of RAG evaluation metrics.

    Usage
    -----
    registry = build_default_rag_registry()
    reg = registry.get("precision_at_k")
    result = reg.fn(rag_input, k=5)
    """

    def __init__(self) -> None:
        self._entries: dict[str, RAGMetricRegistration] = {}

    def register(self, registration: RAGMetricRegistration) -> None:
        self._entries[registration.name] = registration

    def get(self, name: str) -> RAGMetricRegistration:
        if name not in self._entries:
            raise KeyError(f"RAG metric '{name}' not found in registry")
        return self._entries[name]

    def list_all(self) -> list[str]:
        return sorted(self._entries.keys())

    def list_by_type(self, metric_type: str) -> list[str]:
        return sorted(
            name for name, reg in self._entries.items()
            if reg.metric_type.upper() == metric_type.upper()
        )

    def list_retrieval(self) -> list[str]:
        return sorted(
            name for name, reg in self._entries.items()
            if "retrieval" in reg.tags
        )

    def list_generation(self) -> list[str]:
        return sorted(
            name for name, reg in self._entries.items()
            if "generation" in reg.tags
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries


def build_default_rag_registry() -> RAGMetricRegistry:
    """
    Build and return a RAGMetricRegistry populated with all built-in
    RAG metrics: 7 retrieval + 4 generation = 11 total.
    """
    from aieval360.evaluation.rag.generation_metrics import (
        answer_completeness_scaffold,
        answer_faithfulness_scaffold,
        answer_relevance_heuristic,
        context_utilisation_heuristic,
    )
    from aieval360.evaluation.rag.retrieval_metrics import (
        context_precision,
        context_recall,
        hit_at_k,
        mean_reciprocal_rank,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )

    registry = RAGMetricRegistry()

    # --- Retrieval metrics (DETERMINISTIC) --------------------------------

    registry.register(RAGMetricRegistration(
        name="precision_at_k",
        fn=precision_at_k,
        metric_type="DETERMINISTIC",
        description="Fraction of top-K retrieved documents that are relevant. "
                    "Answers: 'Of the K documents shown, how many were useful?'",
        default_k=5,
        default_threshold=0.5,
        tags=["retrieval"],
    ))

    registry.register(RAGMetricRegistration(
        name="recall_at_k",
        fn=recall_at_k,
        metric_type="DETERMINISTIC",
        description="Fraction of all relevant documents found in top-K. "
                    "Answers: 'Did the retriever surface enough of the relevant corpus?'",
        default_k=5,
        default_threshold=0.7,
        tags=["retrieval"],
    ))

    registry.register(RAGMetricRegistration(
        name="mrr",
        fn=mean_reciprocal_rank,
        metric_type="DETERMINISTIC",
        description="Mean Reciprocal Rank: 1 / rank_of_first_relevant_document. "
                    "High values mean the most relevant doc appears near the top.",
        default_k=None,
        default_threshold=0.5,
        tags=["retrieval"],
    ))

    registry.register(RAGMetricRegistration(
        name="ndcg_at_k",
        fn=ndcg_at_k,
        metric_type="DETERMINISTIC",
        description="Normalised Discounted Cumulative Gain at K. "
                    "Rewards ranking relevant documents higher in the list.",
        default_k=5,
        default_threshold=0.6,
        tags=["retrieval"],
    ))

    registry.register(RAGMetricRegistration(
        name="hit_at_k",
        fn=hit_at_k,
        metric_type="DETERMINISTIC",
        description="Binary: 1.0 if at least one relevant document appears in top-K, "
                    "else 0.0. Useful as a mandatory gate in RAG pipelines.",
        default_k=5,
        default_threshold=1.0,
        tags=["retrieval"],
    ))

    registry.register(RAGMetricRegistration(
        name="context_precision",
        fn=context_precision,
        metric_type="DETERMINISTIC",
        description="RAG-specific alias for Precision@K. "
                    "Answers: 'Were the retrieved context chunks actually relevant?'",
        default_k=5,
        default_threshold=0.5,
        tags=["retrieval"],
    ))

    registry.register(RAGMetricRegistration(
        name="context_recall",
        fn=context_recall,
        metric_type="DETERMINISTIC",
        description="RAG-specific alias for Recall@K. "
                    "Answers: 'Did the retriever find enough of the relevant context?'",
        default_k=5,
        default_threshold=0.7,
        tags=["retrieval"],
    ))

    # --- Generation metrics -----------------------------------------------

    registry.register(RAGMetricRegistration(
        name="answer_relevance_heuristic",
        fn=answer_relevance_heuristic,
        metric_type="HEURISTIC",
        description="Jaccard token overlap between the query and the generated answer. "
                    "Quick proxy for relevance; use alongside an LLM-judged metric for "
                    "governance-grade assessments.",
        default_threshold=0.1,
        tags=["generation"],
    ))

    registry.register(RAGMetricRegistration(
        name="context_utilisation_heuristic",
        fn=context_utilisation_heuristic,
        metric_type="HEURISTIC",
        description="Fraction of retrieved context tokens that appear in the answer. "
                    "Low score despite high context recall signals potential hallucination.",
        default_threshold=0.05,
        tags=["generation"],
    ))

    registry.register(RAGMetricRegistration(
        name="answer_faithfulness",
        fn=answer_faithfulness_scaffold,
        metric_type="LLM_JUDGED",
        description="LLM-judged faithfulness: are all claims in the answer grounded in "
                    "the retrieved context? Requires judge_response pre-populated on RAGInput.",
        default_threshold=0.7,
        requires_judge=True,
        tags=["generation"],
    ))

    registry.register(RAGMetricRegistration(
        name="answer_completeness",
        fn=answer_completeness_scaffold,
        metric_type="LLM_JUDGED",
        description="LLM-judged completeness: does the answer fully address all aspects "
                    "of the query given the available context? "
                    "Requires judge_response pre-populated on RAGInput.",
        default_threshold=0.6,
        requires_judge=True,
        tags=["generation"],
    ))

    return registry


# Module-level default registry — import and use directly in the pipeline
default_rag_registry: RAGMetricRegistry = build_default_rag_registry()
