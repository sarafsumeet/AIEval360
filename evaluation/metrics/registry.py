"""
Default metric registry — pre-populated with all built-in calculators.

Usage::

    from aieval360.evaluation.metrics.registry import default_registry
    calc = default_registry.get("accuracy")
    result = calc.calculate(inputs, threshold=0.80)
"""

from __future__ import annotations

from .base import MetricRegistry
from .llm_metrics import build_llm_registry
from .traditional_ml import build_traditional_ml_registry


def build_default_registry() -> MetricRegistry:
    """Create and return a MetricRegistry pre-populated with all built-in metrics."""
    registry = MetricRegistry()
    all_calculators = {
        **build_traditional_ml_registry(),
        **build_llm_registry(),
    }
    for calc in all_calculators.values():
        registry.register(calc)
    return registry


# Module-level singleton — import and use directly in most cases.
default_registry: MetricRegistry = build_default_registry()
