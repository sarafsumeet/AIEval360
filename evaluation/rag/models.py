"""
Domain models for RAG evaluation.

Separates the retrieval layer (documents retrieved) from the generation layer
(answer produced given those documents), keeping governance evidence on both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RetrievedDocument:
    """A single document chunk returned by the retrieval step."""

    doc_id: str
    content: str
    score: float = 0.0          # similarity / BM25 / rerank score (higher = more relevant)
    rank: int = 0               # 1-based position in the ranked list
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("rank must be >= 0 (0 = unranked)")


@dataclass
class RAGInput:
    """
    Full input bundle for a single RAG evaluation example.

    Fields
    ------
    query           The user question / prompt sent to the retriever.
    retrieved_docs  Ranked list of documents the retriever returned.
    generated_answer  The answer produced by the LLM given the retrieved docs.
    ground_truth_answer  Reference answer used for answer-quality metrics.
    relevant_doc_ids  Set of doc_ids considered relevant (for retrieval metrics).
    judge_response  Pre-computed LLM-judge JSON (for grounding / relevance metrics).
    metadata        Any extra context (dataset_id, eval_run_id, etc.).
    """

    query: str
    retrieved_docs: list[RetrievedDocument] = field(default_factory=list)
    generated_answer: Optional[str] = None
    ground_truth_answer: Optional[str] = None
    relevant_doc_ids: set[str] = field(default_factory=set)
    judge_response: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGMetricOutput:
    """
    Result from a single RAG metric calculator.

    value       Numeric score (None if the metric could not be computed).
    threshold   Pass/fail gate (None = informational only).
    details     Breakdown dict for audit / explainability.
    reasoning   Human-readable explanation (populated by LLM-judged metrics).
    error       Non-None when the metric failed to compute.
    """

    metric_name: str
    value: Optional[float]
    threshold: Optional[float] = None
    details: dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None
    error: Optional[str] = None

    @property
    def passed(self) -> Optional[bool]:
        if self.value is None or self.threshold is None:
            return None
        return self.value >= self.threshold


@dataclass
class RAGEvaluationResult:
    """
    Aggregated result for a complete RAG evaluation run across N examples.

    metric_outputs  One RAGMetricOutput per metric (averaged / aggregated).
    per_example     Per-example breakdowns keyed by example index.
    total_examples  Number of examples evaluated.
    errors          Any errors encountered during the run.
    elapsed_seconds Wall-clock time for the entire run.
    """

    metric_outputs: list[RAGMetricOutput] = field(default_factory=list)
    per_example: list[dict[str, Any]] = field(default_factory=list)
    total_examples: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def has_failures(self) -> bool:
        return any(m.passed is False for m in self.metric_outputs)

    @property
    def overall_status(self) -> str:
        if self.has_failures:
            return "completed_with_failures"
        return "completed"
