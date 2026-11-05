"""
Base classes for the metric calculation framework.

MetricCalculator is the ABC every metric implementation inherits from.
MetricRegistry is the central catalogue of available metrics.

Design principles:
- Calculators are stateless; all context is passed to calculate().
- MetricType is recorded on every result for transparency (deterministic vs LLM-judged).
- Mandatory-gate flag marks critical controls whose failure cannot be offset.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional

from aieval360.core.models.evaluation import MetricType


@dataclass
class MetricInput:
    """
    Unified input to any metric calculator.

    Fields are populated only when the metric needs them.
    Calculators should raise ValueError for missing required fields.
    """

    # Core payload (at least one of predictions/references is usually required)
    predictions: list[Any] = field(default_factory=list)
    references: list[Any] = field(default_factory=list)

    # LLM-output–specific
    generated_text: Optional[str] = None
    reference_text: Optional[str] = None
    source_text: Optional[str] = None      # e.g. retrieved context for RAG faithfulness

    # Classification / regression
    labels: list[Any] = field(default_factory=list)       # ground-truth class labels
    probabilities: list[list[float]] = field(default_factory=list)  # per-class probs

    # LLM-as-judge channel
    judge_response: Optional[str] = None   # raw JSON or text from the judge model

    # Metadata passed through (available to calculators but not required)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricOutput:
    """
    Result from a single metric calculator.

    value      – primary numeric score (None if metric could not compute)
    details    – extra breakdown dict (class-level scores, sub-metrics, etc.)
    reasoning  – human-readable justification (especially for LLM-judged)
    error      – set when calculation failed; value will be None
    """

    metric_name: str
    metric_type: MetricType
    value: Optional[float]
    threshold: Optional[float] = None
    details: dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None
    error: Optional[str] = None

    @property
    def passed(self) -> Optional[bool]:
        """True/False if threshold is set; None if no threshold or no value."""
        if self.value is None or self.threshold is None:
            return None
        return self.value >= self.threshold


class MetricCalculator(abc.ABC):
    """
    Abstract base class for every metric implementation.

    Subclasses must implement:
    - metric_name  – unique identifier (e.g. "accuracy", "bleu", "faithfulness")
    - metric_type  – MetricType enum value
    - calculate()  – pure function: MetricInput → MetricOutput
    """

    @property
    @abc.abstractmethod
    def metric_name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def metric_type(self) -> MetricType:
        ...

    @abc.abstractmethod
    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        """
        Compute the metric.

        Must never raise unless inputs are structurally invalid.
        Recoverable errors (e.g. empty lists) should return MetricOutput with error set.
        """
        ...

    def _make_output(
        self,
        value: Optional[float],
        threshold: Optional[float] = None,
        details: Optional[dict] = None,
        reasoning: Optional[str] = None,
        error: Optional[str] = None,
    ) -> MetricOutput:
        """Convenience factory so subclasses don't repeat boilerplate."""
        return MetricOutput(
            metric_name=self.metric_name,
            metric_type=self.metric_type,
            value=value,
            threshold=threshold,
            details=details or {},
            reasoning=reasoning,
            error=error,
        )


class MetricRegistry:
    """
    Central catalogue of registered MetricCalculators.

    Usage::

        registry = MetricRegistry()
        registry.register(AccuracyCalculator())
        calc = registry.get("accuracy")
        result = calc.calculate(inputs, threshold=0.80)
    """

    def __init__(self) -> None:
        self._calculators: dict[str, MetricCalculator] = {}

    def register(self, calculator: MetricCalculator) -> None:
        """Register a calculator. Raises if name is already taken."""
        name = calculator.metric_name
        if name in self._calculators:
            raise ValueError(f"Metric {name!r} is already registered. Use replace=True to override.")
        self._calculators[name] = calculator

    def register_or_replace(self, calculator: MetricCalculator) -> None:
        """Register, silently replacing any existing calculator with the same name."""
        self._calculators[calculator.metric_name] = calculator

    def get(self, name: str) -> MetricCalculator:
        """Return calculator for name, raising KeyError if absent."""
        if name not in self._calculators:
            raise KeyError(f"No metric registered for {name!r}. Available: {self.list_names()}")
        return self._calculators[name]

    def get_or_none(self, name: str) -> Optional[MetricCalculator]:
        return self._calculators.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._calculators.keys())

    def list_by_type(self, metric_type: MetricType) -> list[MetricCalculator]:
        return [c for c in self._calculators.values() if c.metric_type == metric_type]

    def __contains__(self, name: str) -> bool:
        return name in self._calculators

    def __len__(self) -> int:
        return len(self._calculators)
