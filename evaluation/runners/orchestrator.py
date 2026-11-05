"""
Evaluation orchestrator — coordinates a single evaluation run.

The EvaluationOrchestrator:
1. Accepts a run specification (which metrics, which dataset rows, which provider)
2. Dispatches completion requests to the provider adapter
3. Calls each metric calculator with the results
4. Returns a list of MetricOutput records ready for persistence

Evidence guarantees (Supplementary Recommendation #2):
- Every EvaluationSpec records provider_name, model_name, model_version
- Every run records a dataset_id and row count for lineage
- Mandatory gate failures are surfaced immediately in the return value

The orchestrator does NOT write to the database; callers persist the results.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from aieval360.providers.base.interface import BaseProviderAdapter, CompletionRequest

from ..metrics.base import MetricCalculator, MetricInput, MetricOutput, MetricRegistry


@dataclass
class EvaluationSpec:
    """
    Specification for a single evaluation run.

    Fields map directly to EvaluationRun evidence fields in the DB model.
    """

    # Identity
    name: str
    evaluation_category: str               # e.g. "llm_benchmark", "classification"
    organisation_id: str
    project_id: str
    ai_system_id: str
    dataset_id: str

    # Provider evidence (required by governance model)
    provider_name: str
    model_name: str
    model_version: str = "unknown"
    judge_model: Optional[str] = None      # for LLM-judged metrics

    # Dataset rows — list of dicts with at minimum 'input' (and usually 'reference')
    dataset_rows: list[dict[str, Any]] = field(default_factory=list)

    # Metrics to run: list of (metric_name, threshold)
    metrics: list[tuple[str, Optional[float]]] = field(default_factory=list)

    # Governance
    executed_by: str = "system"
    is_reproducible: bool = True
    reproducibility_notes: str = ""
    metric_definitions_version: str = "v1.0.0"

    # Prompt template to wrap each input row
    prompt_template: str = "{input}"

    # Concurrency (default: sequential to keep tests simple)
    max_concurrent: int = 1


@dataclass
class RunResult:
    """Aggregated result of a completed evaluation run."""

    spec: EvaluationSpec
    metric_outputs: list[MetricOutput]
    total_rows: int
    completed_rows: int
    failed_rows: int
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)

    @property
    def mandatory_gate_failures(self) -> list[MetricOutput]:
        """
        Metric outputs that failed a mandatory gate.
        A failed mandatory gate means the run should not be approved.
        """
        return [
            m for m in self.metric_outputs
            if m.passed is False
        ]

    @property
    def has_critical_failure(self) -> bool:
        """True if any mandatory-gate metric failed."""
        return bool(self.mandatory_gate_failures)

    @property
    def overall_status(self) -> str:
        if self.failed_rows > 0 and self.completed_rows == 0:
            return "failed"
        if self.has_critical_failure:
            return "completed_with_failures"
        return "completed"


class EvaluationOrchestrator:
    """
    Coordinates an evaluation run against a provider and metric registry.

    Usage::

        orchestrator = EvaluationOrchestrator(registry, provider)
        result = await orchestrator.run(spec)
    """

    def __init__(
        self,
        registry: MetricRegistry,
        provider: BaseProviderAdapter,
    ) -> None:
        self.registry = registry
        self.provider = provider

    async def run(self, spec: EvaluationSpec) -> RunResult:
        """Execute the evaluation and return aggregated results."""
        start = time.monotonic()
        errors: list[str] = []
        all_inputs: list[MetricInput] = []
        completed_rows = 0
        failed_rows = 0

        # --- Generate completions for each dataset row ---
        if spec.dataset_rows and self._needs_completions(spec):
            semaphore = asyncio.Semaphore(max(1, spec.max_concurrent))

            async def _process_row(row: dict[str, Any]) -> Optional[MetricInput]:
                nonlocal completed_rows, failed_rows
                async with semaphore:
                    prompt = spec.prompt_template.format(**row)
                    request = CompletionRequest(
                        messages=[{"role": "user", "content": prompt}],
                        model=spec.model_name,
                        max_tokens=512,
                    )
                    try:
                        response = await self.provider.complete(request)
                        completed_rows += 1
                        return MetricInput(
                            predictions=[response.content],
                            references=[str(row.get("reference", row.get("expected", "")))],
                            generated_text=response.content,
                            reference_text=str(row.get("reference", row.get("expected", ""))),
                            source_text=str(row.get("context", row.get("source", ""))),
                            metadata={"row": row, "response": response},
                        )
                    except Exception as exc:
                        failed_rows += 1
                        errors.append(f"Row failed: {exc}")
                        return None

            tasks = [_process_row(row) for row in spec.dataset_rows]
            results = await asyncio.gather(*tasks)
            all_inputs = [r for r in results if r is not None]
        elif spec.dataset_rows:
            # Pre-computed inputs (predictions already in rows)
            for row in spec.dataset_rows:
                all_inputs.append(MetricInput(
                    predictions=[str(row.get("prediction", row.get("output", "")))],
                    references=[str(row.get("reference", row.get("expected", "")))],
                    generated_text=str(row.get("prediction", row.get("output", ""))),
                    reference_text=str(row.get("reference", row.get("expected", ""))),
                    source_text=str(row.get("context", "")),
                    metadata={"row": row},
                ))
            completed_rows = len(all_inputs)

        # --- Aggregate per-metric across all rows ---
        metric_outputs: list[MetricOutput] = []
        for metric_name, threshold in spec.metrics:
            calculator = self.registry.get_or_none(metric_name)
            if calculator is None:
                errors.append(f"Metric {metric_name!r} not found in registry — skipped")
                continue
            outputs = self._compute_metric(calculator, all_inputs, threshold)
            metric_outputs.append(outputs)

        elapsed = time.monotonic() - start
        return RunResult(
            spec=spec,
            metric_outputs=metric_outputs,
            total_rows=len(spec.dataset_rows),
            completed_rows=completed_rows,
            failed_rows=failed_rows,
            elapsed_seconds=elapsed,
            errors=errors,
        )

    def _needs_completions(self, spec: EvaluationSpec) -> bool:
        """True when dataset rows contain 'input' fields (need LLM inference)."""
        if not spec.dataset_rows:
            return False
        first_row = spec.dataset_rows[0]
        return "input" in first_row and "prediction" not in first_row

    def _compute_metric(
        self,
        calculator: MetricCalculator,
        all_inputs: list[MetricInput],
        threshold: Optional[float],
    ) -> MetricOutput:
        """Compute a metric over all inputs, aggregating by mean."""
        if not all_inputs:
            return calculator._make_output(None, threshold, error="No inputs to compute metric over")

        # For aggregate metrics (BLEU, ROUGE, accuracy) pass all predictions/references together
        aggregated = MetricInput(
            predictions=[i.predictions[0] for i in all_inputs if i.predictions],
            references=[i.references[0] for i in all_inputs if i.references],
            generated_text=" ".join(i.generated_text or "" for i in all_inputs[:1]),
            reference_text=" ".join(i.reference_text or "" for i in all_inputs[:1]),
            labels=[i.labels[0] for i in all_inputs if i.labels],
            probabilities=[i.probabilities[0] for i in all_inputs if i.probabilities],
            judge_response=all_inputs[0].judge_response if all_inputs else None,
        )
        return calculator.calculate(aggregated, threshold)
