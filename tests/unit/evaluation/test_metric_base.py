"""
Unit tests for MetricRegistry and MetricOutput base behaviour.
"""

from __future__ import annotations

import pytest

from aieval360.evaluation.metrics.base import MetricCalculator, MetricInput, MetricOutput, MetricRegistry
from aieval360.evaluation.metrics.registry import build_default_registry, default_registry
from aieval360.core.models.evaluation import MetricType


# ---------------------------------------------------------------------------
# Minimal concrete calculator for testing the registry
# ---------------------------------------------------------------------------

class ConstantCalculator(MetricCalculator):
    """Always returns a fixed value — useful for registry tests."""

    def __init__(self, name: str, value: float, metric_type: MetricType = MetricType.DETERMINISTIC):
        self._name = name
        self._value = value
        self._type = metric_type

    @property
    def metric_name(self) -> str:
        return self._name

    @property
    def metric_type(self) -> MetricType:
        return self._type

    def calculate(self, inputs: MetricInput, threshold=None) -> MetricOutput:
        return self._make_output(self._value, threshold)


# ---------------------------------------------------------------------------
# MetricOutput
# ---------------------------------------------------------------------------

class TestMetricOutput:
    def test_passed_true_when_value_meets_threshold(self):
        out = MetricOutput("acc", MetricType.DETERMINISTIC, value=0.90, threshold=0.80)
        assert out.passed is True

    def test_passed_false_when_value_below_threshold(self):
        out = MetricOutput("acc", MetricType.DETERMINISTIC, value=0.70, threshold=0.80)
        assert out.passed is False

    def test_passed_none_when_no_threshold(self):
        out = MetricOutput("acc", MetricType.DETERMINISTIC, value=0.70)
        assert out.passed is None

    def test_passed_none_when_no_value(self):
        out = MetricOutput("acc", MetricType.DETERMINISTIC, value=None, threshold=0.80)
        assert out.passed is None

    def test_error_field(self):
        out = MetricOutput("acc", MetricType.DETERMINISTIC, value=None, error="something broke")
        assert out.error == "something broke"
        assert out.value is None


# ---------------------------------------------------------------------------
# MetricRegistry
# ---------------------------------------------------------------------------

class TestMetricRegistry:
    def test_register_and_get(self):
        registry = MetricRegistry()
        registry.register(ConstantCalculator("score", 0.5))
        calc = registry.get("score")
        assert calc.metric_name == "score"

    def test_get_unknown_raises_key_error(self):
        registry = MetricRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_get_or_none_returns_none_for_unknown(self):
        registry = MetricRegistry()
        assert registry.get_or_none("nonexistent") is None

    def test_duplicate_registration_raises(self):
        registry = MetricRegistry()
        registry.register(ConstantCalculator("x", 1.0))
        with pytest.raises(ValueError):
            registry.register(ConstantCalculator("x", 2.0))

    def test_register_or_replace_overwrites(self):
        registry = MetricRegistry()
        registry.register(ConstantCalculator("x", 1.0))
        registry.register_or_replace(ConstantCalculator("x", 2.0))
        result = registry.get("x").calculate(MetricInput())
        assert result.value == pytest.approx(2.0)

    def test_list_names_sorted(self):
        registry = MetricRegistry()
        registry.register(ConstantCalculator("z_metric", 0.0))
        registry.register(ConstantCalculator("a_metric", 0.0))
        names = registry.list_names()
        assert names == sorted(names)

    def test_contains_operator(self):
        registry = MetricRegistry()
        registry.register(ConstantCalculator("acc", 0.9))
        assert "acc" in registry
        assert "missing" not in registry

    def test_len(self):
        registry = MetricRegistry()
        registry.register(ConstantCalculator("a", 0.0))
        registry.register(ConstantCalculator("b", 0.0))
        assert len(registry) == 2

    def test_list_by_type(self):
        registry = MetricRegistry()
        registry.register(ConstantCalculator("det", 0.0, MetricType.DETERMINISTIC))
        registry.register(ConstantCalculator("llm", 0.0, MetricType.LLM_JUDGED))
        det = registry.list_by_type(MetricType.DETERMINISTIC)
        assert len(det) == 1
        assert det[0].metric_name == "det"


# ---------------------------------------------------------------------------
# Default registry (integration)
# ---------------------------------------------------------------------------

class TestDefaultRegistry:
    def test_default_registry_has_accuracy(self):
        assert "accuracy" in default_registry

    def test_default_registry_has_bleu(self):
        assert "bleu" in default_registry

    def test_default_registry_has_faithfulness(self):
        assert "faithfulness" in default_registry

    def test_default_registry_covers_all_metric_types(self):
        types = {c.metric_type for c in default_registry._calculators.values()}
        # At minimum: DETERMINISTIC, REFERENCE_BASED, HEURISTIC, LLM_JUDGED
        assert MetricType.DETERMINISTIC in types
        assert MetricType.REFERENCE_BASED in types
        assert MetricType.HEURISTIC in types
        assert MetricType.LLM_JUDGED in types

    def test_build_default_registry_creates_fresh_instance(self):
        r1 = build_default_registry()
        r2 = build_default_registry()
        # Should be separate objects
        assert r1 is not r2

    def test_accuracy_runs_end_to_end(self):
        calc = default_registry.get("accuracy")
        result = calc.calculate(MetricInput(predictions=[1, 1], references=[1, 1]))
        assert result.value == pytest.approx(1.0)
