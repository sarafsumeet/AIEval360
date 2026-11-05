"""
RAG Evaluation Pipeline.

Orchestrates a complete RAG evaluation run across a list of RAGInput examples,
computing both retrieval and generation metrics, and returning an aggregated
RAGEvaluationResult with per-example breakdowns for audit / governance.

Design decisions
----------------
- Stateless: the pipeline holds no mutable state between runs.
- Evidence-first: every run records which metrics were requested, their
  thresholds, and the per-example raw scores so the result can be persisted
  to an audit log without loss of detail.
- LLM-judge decoupled: the pipeline accepts an optional judge callable
  (async fn: query, context, answer → str) so LLM calls remain mockable
  in tests and swappable in production.
- Pure-Python: no sklearn / sacrebleu dependency for retrieval metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from aieval360.evaluation.rag.generation_metrics import (
    answer_completeness_scaffold,
    answer_faithfulness_scaffold,
    answer_relevance_heuristic,
    context_utilisation_heuristic,
)
from aieval360.evaluation.rag.models import (
    RAGEvaluationResult,
    RAGInput,
    RAGMetricOutput,
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

# Type alias for an async judge callable
JudgeFn = Callable[[str, str, str], Awaitable[str]]


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class RAGPipelineConfig:
    """
    Configuration for one RAG evaluation run.

    retrieval_k         Top-K cutoff for retrieval metrics (Precision, Recall,
                        NDCG, Hit).  Defaults to 5.
    retrieval_metrics   Names of retrieval metrics to compute.
    generation_metrics  Names of generation metrics to compute.
    thresholds          Dict of metric_name → pass/fail threshold.
    judge_fn            Optional async callable that produces a judge JSON
                        string given (query, context, answer).  Required for
                        LLM-judged generation metrics.
    executed_by         Identity of the person / system that triggered the run
                        (governance evidence field).
    run_label           Human-readable label for this evaluation run.
    """

    retrieval_k: int = 5
    retrieval_metrics: list[str] = field(default_factory=lambda: [
        "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "hit_at_k",
        "context_precision", "context_recall",
    ])
    generation_metrics: list[str] = field(default_factory=lambda: [
        "answer_relevance_heuristic",
        "context_utilisation_heuristic",
        "answer_faithfulness",
        "answer_completeness",
    ])
    thresholds: dict[str, float] = field(default_factory=dict)
    judge_fn: Optional[JudgeFn] = None
    executed_by: str = "system"
    run_label: str = "rag_evaluation"


# ---------------------------------------------------------------------------
# Retrieval metric dispatch
# ---------------------------------------------------------------------------

_RETRIEVAL_DISPATCH: dict[str, Any] = {
    "precision_at_k":    precision_at_k,
    "recall_at_k":       recall_at_k,
    "mrr":               mean_reciprocal_rank,
    "ndcg_at_k":         ndcg_at_k,
    "hit_at_k":          hit_at_k,
    "context_precision": context_precision,
    "context_recall":    context_recall,
}

_GENERATION_DISPATCH: dict[str, Any] = {
    "answer_relevance_heuristic":    answer_relevance_heuristic,
    "context_utilisation_heuristic": context_utilisation_heuristic,
    "answer_faithfulness":           answer_faithfulness_scaffold,
    "answer_completeness":           answer_completeness_scaffold,
}

_K_METRICS = {"precision_at_k", "recall_at_k", "ndcg_at_k", "hit_at_k",
              "context_precision", "context_recall"}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGEvaluationPipeline:
    """
    End-to-end RAG evaluation pipeline.

    Usage
    -----
    pipeline = RAGEvaluationPipeline(config)
    result   = await pipeline.run(examples)
    """

    def __init__(self, config: Optional[RAGPipelineConfig] = None) -> None:
        self.config = config or RAGPipelineConfig()

    async def run(self, examples: list[RAGInput]) -> RAGEvaluationResult:
        """
        Evaluate all examples and return aggregated results.

        Steps
        -----
        1. For each example, compute requested retrieval metrics.
        2. For each example, optionally call judge_fn then compute generation metrics.
        3. Aggregate per-metric scores (mean) across all examples.
        4. Return RAGEvaluationResult with per-example breakdowns.
        """
        start = time.monotonic()
        cfg = self.config
        per_example: list[dict[str, Any]] = []
        all_scores: dict[str, list[float]] = {}  # metric_name → list of per-example values
        errors: list[str] = []

        for idx, example in enumerate(examples):
            example_scores: dict[str, Optional[float]] = {}
            example_errors: list[str] = []

            # --- Retrieval metrics ---
            for metric_name in cfg.retrieval_metrics:
                fn = _RETRIEVAL_DISPATCH.get(metric_name)
                if fn is None:
                    example_errors.append(f"[example {idx}] unknown retrieval metric: {metric_name}")
                    continue
                threshold = cfg.thresholds.get(metric_name)
                try:
                    if metric_name in _K_METRICS:
                        out: RAGMetricOutput = fn(example, k=cfg.retrieval_k, threshold=threshold)
                    else:
                        out = fn(example, threshold=threshold)
                    example_scores[metric_name] = out.value
                    if out.value is not None:
                        all_scores.setdefault(metric_name, []).append(out.value)
                    if out.error:
                        example_errors.append(f"[example {idx}] {metric_name}: {out.error}")
                except Exception as exc:  # noqa: BLE001
                    example_errors.append(f"[example {idx}] {metric_name} raised: {exc}")

            # --- Judge call (if needed) ---
            if cfg.judge_fn is not None:
                llm_judged = {"answer_faithfulness", "answer_completeness"}
                needs_judge = any(m in llm_judged for m in cfg.generation_metrics)
                if needs_judge and example.query and example.generated_answer:
                    context_text = " ".join(d.content for d in example.retrieved_docs)
                    try:
                        example.judge_response = await cfg.judge_fn(
                            example.query, context_text, example.generated_answer
                        )
                    except Exception as exc:  # noqa: BLE001
                        example_errors.append(f"[example {idx}] judge_fn raised: {exc}")

            # --- Generation metrics ---
            for metric_name in cfg.generation_metrics:
                fn = _GENERATION_DISPATCH.get(metric_name)
                if fn is None:
                    example_errors.append(f"[example {idx}] unknown generation metric: {metric_name}")
                    continue
                threshold = cfg.thresholds.get(metric_name)
                try:
                    out = fn(example, threshold=threshold)
                    example_scores[metric_name] = out.value
                    if out.value is not None:
                        all_scores.setdefault(metric_name, []).append(out.value)
                    if out.error:
                        example_errors.append(f"[example {idx}] {metric_name}: {out.error}")
                except Exception as exc:  # noqa: BLE001
                    example_errors.append(f"[example {idx}] {metric_name} raised: {exc}")

            per_example.append({"index": idx, "scores": example_scores, "errors": example_errors})
            errors.extend(example_errors)

        # --- Aggregate: mean per metric ---
        aggregated: list[RAGMetricOutput] = []
        all_metric_names = list(cfg.retrieval_metrics) + list(cfg.generation_metrics)
        for metric_name in all_metric_names:
            scores = all_scores.get(metric_name, [])
            if scores:
                mean_val = sum(scores) / len(scores)
            else:
                mean_val = None
            aggregated.append(RAGMetricOutput(
                metric_name=metric_name,
                value=mean_val,
                threshold=cfg.thresholds.get(metric_name),
                details={"n_examples": len(scores), "scores": scores},
            ))

        return RAGEvaluationResult(
            metric_outputs=aggregated,
            per_example=per_example,
            total_examples=len(examples),
            errors=errors,
            elapsed_seconds=time.monotonic() - start,
        )
