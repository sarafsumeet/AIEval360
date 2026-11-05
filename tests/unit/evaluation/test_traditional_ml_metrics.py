"""
Unit tests for traditional ML metric calculators.

All tests are deterministic and require no network or database access.
"""

from __future__ import annotations

import pytest

from aieval360.evaluation.metrics.base import MetricInput
from aieval360.evaluation.metrics.traditional_ml import (
    AccuracyCalculator,
    F1Calculator,
    MaeCalculator,
    PrecisionCalculator,
    R2Calculator,
    RecallCalculator,
    RmseCalculator,
    build_traditional_ml_registry,
)
from aieval360.core.models.evaluation import MetricType


class TestAccuracy:
    def test_perfect_predictions(self):
        calc = AccuracyCalculator()
        result = calc.calculate(MetricInput(predictions=[1, 0, 1], references=[1, 0, 1]))
        assert result.value == pytest.approx(1.0)

    def test_all_wrong(self):
        calc = AccuracyCalculator()
        result = calc.calculate(MetricInput(predictions=[0, 1, 0], references=[1, 0, 1]))
        assert result.value == pytest.approx(0.0)

    def test_half_correct(self):
        calc = AccuracyCalculator()
        result = calc.calculate(MetricInput(predictions=[1, 1, 0, 0], references=[1, 0, 1, 0]))
        assert result.value == pytest.approx(0.5)

    def test_threshold_pass(self):
        calc = AccuracyCalculator()
        result = calc.calculate(MetricInput(predictions=[1, 1], references=[1, 1]), threshold=0.80)
        assert result.passed is True

    def test_threshold_fail(self):
        calc = AccuracyCalculator()
        result = calc.calculate(MetricInput(predictions=[1, 0], references=[1, 1]), threshold=0.80)
        assert result.passed is False

    def test_empty_inputs_returns_error(self):
        calc = AccuracyCalculator()
        result = calc.calculate(MetricInput(predictions=[], references=[]))
        assert result.value is None
        assert result.error is not None

    def test_mismatched_lengths_returns_error(self):
        calc = AccuracyCalculator()
        result = calc.calculate(MetricInput(predictions=[1, 2], references=[1]))
        assert result.value is None

    def test_metric_type_is_deterministic(self):
        assert AccuracyCalculator().metric_type == MetricType.DETERMINISTIC

    def test_metric_name(self):
        assert AccuracyCalculator().metric_name == "accuracy"

    def test_details_contain_correct_and_total(self):
        calc = AccuracyCalculator()
        result = calc.calculate(MetricInput(predictions=[1, 1, 0], references=[1, 1, 1]))
        assert result.details["correct"] == 2
        assert result.details["total"] == 3


class TestPrecision:
    def test_binary_perfect(self):
        calc = PrecisionCalculator()
        result = calc.calculate(MetricInput(predictions=[1, 0, 1], references=[1, 0, 1]))
        assert result.value == pytest.approx(1.0)

    def test_no_true_positives(self):
        calc = PrecisionCalculator()
        result = calc.calculate(MetricInput(predictions=[0, 0], references=[1, 1]))
        assert result.value == pytest.approx(0.0)

    def test_metric_type_deterministic(self):
        assert PrecisionCalculator().metric_type == MetricType.DETERMINISTIC


class TestRecall:
    def test_binary_perfect(self):
        calc = RecallCalculator()
        result = calc.calculate(MetricInput(predictions=[1, 1, 1], references=[1, 1, 1]))
        assert result.value == pytest.approx(1.0)

    def test_zero_recall(self):
        calc = RecallCalculator()
        result = calc.calculate(MetricInput(predictions=[0, 0], references=[1, 1]))
        assert result.value == pytest.approx(0.0)


class TestF1:
    def test_perfect(self):
        calc = F1Calculator()
        result = calc.calculate(MetricInput(predictions=["A", "B", "A"], references=["A", "B", "A"]))
        assert result.value == pytest.approx(1.0)

    def test_zero(self):
        calc = F1Calculator()
        result = calc.calculate(MetricInput(predictions=["A", "A"], references=["B", "B"]))
        assert result.value == pytest.approx(0.0)

    def test_metric_name(self):
        assert F1Calculator().metric_name == "f1"


class TestRmse:
    def test_perfect(self):
        calc = RmseCalculator()
        result = calc.calculate(MetricInput(predictions=[1.0, 2.0, 3.0], references=[1.0, 2.0, 3.0]))
        assert result.value == pytest.approx(0.0)

    def test_known_rmse(self):
        calc = RmseCalculator()
        # errors: [1, 1, 1] → MSE = 1.0 → RMSE = 1.0
        result = calc.calculate(MetricInput(predictions=[0.0, 1.0, 2.0], references=[1.0, 2.0, 3.0]))
        assert result.value == pytest.approx(1.0)

    def test_lower_is_better_in_details(self):
        calc = RmseCalculator()
        result = calc.calculate(MetricInput(predictions=[1.0], references=[2.0]))
        assert result.details["lower_is_better"] is True

    def test_metric_type_deterministic(self):
        assert RmseCalculator().metric_type == MetricType.DETERMINISTIC


class TestMae:
    def test_perfect(self):
        calc = MaeCalculator()
        result = calc.calculate(MetricInput(predictions=[3.0], references=[3.0]))
        assert result.value == pytest.approx(0.0)

    def test_known_mae(self):
        calc = MaeCalculator()
        result = calc.calculate(MetricInput(predictions=[1.0, 3.0], references=[2.0, 4.0]))
        assert result.value == pytest.approx(1.0)


class TestR2:
    def test_perfect(self):
        calc = R2Calculator()
        result = calc.calculate(MetricInput(predictions=[1.0, 2.0, 3.0], references=[1.0, 2.0, 3.0]))
        assert result.value == pytest.approx(1.0)

    def test_constant_prediction(self):
        """Predicting the mean gives R² = 0."""
        calc = R2Calculator()
        result = calc.calculate(MetricInput(predictions=[2.0, 2.0, 2.0], references=[1.0, 2.0, 3.0]))
        assert result.value == pytest.approx(0.0)

    def test_negative_r2_possible(self):
        """R² can be negative if the model is worse than predicting the mean."""
        calc = R2Calculator()
        result = calc.calculate(MetricInput(predictions=[3.0, 1.0], references=[1.0, 3.0]))
        assert result.value is not None
        assert result.value <= 0.0


class TestTraditionalMLRegistry:
    def test_all_expected_metrics_registered(self):
        registry = build_traditional_ml_registry()
        expected = {"accuracy", "precision", "recall", "f1", "roc_auc", "rmse", "mae", "r2"}
        assert expected == set(registry.keys())

    def test_calculators_are_callable(self):
        registry = build_traditional_ml_registry()
        for name, calc in registry.items():
            assert callable(calc.calculate), f"{name} calculator must be callable"
