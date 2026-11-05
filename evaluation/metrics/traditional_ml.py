"""
Traditional ML metrics: classification and regression.

All calculators are deterministic (no LLM calls).
scikit-learn is used where available; pure-Python fallbacks otherwise.

Metrics implemented:
  Classification: accuracy, precision, recall, f1, roc_auc
  Regression:     rmse, mae, r2
"""

from __future__ import annotations

import math
from typing import Optional

from aieval360.core.models.evaluation import MetricType

from .base import MetricCalculator, MetricInput, MetricOutput


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class AccuracyCalculator(MetricCalculator):
    """Fraction of predictions matching the reference label."""

    @property
    def metric_name(self) -> str:
        return "accuracy"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.DETERMINISTIC

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        preds = inputs.predictions
        refs = inputs.references
        if not preds or not refs:
            return self._make_output(None, threshold, error="predictions and references must be non-empty")
        if len(preds) != len(refs):
            return self._make_output(None, threshold, error="predictions and references must have equal length")
        correct = sum(1 for p, r in zip(preds, refs) if p == r)
        value = correct / len(refs)
        return self._make_output(
            value, threshold,
            details={"correct": correct, "total": len(refs)},
        )


class PrecisionCalculator(MetricCalculator):
    """
    Macro-averaged precision for classification.
    For binary classification, treats the first unique label as positive.
    """

    @property
    def metric_name(self) -> str:
        return "precision"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.DETERMINISTIC

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        preds, refs = inputs.predictions, inputs.references
        if not preds or not refs or len(preds) != len(refs):
            return self._make_output(None, threshold, error="Need equal-length non-empty predictions/references")
        try:
            from sklearn.metrics import precision_score  # type: ignore
            value = float(precision_score(refs, preds, average="macro", zero_division=0))
        except ImportError:
            value = self._macro_precision(preds, refs)
        return self._make_output(value, threshold)

    @staticmethod
    def _macro_precision(preds: list, refs: list) -> float:
        labels = set(refs)
        scores = []
        for lbl in labels:
            tp = sum(1 for p, r in zip(preds, refs) if p == lbl and r == lbl)
            fp = sum(1 for p, r in zip(preds, refs) if p == lbl and r != lbl)
            denom = tp + fp
            scores.append(tp / denom if denom else 0.0)
        return sum(scores) / len(scores) if scores else 0.0


class RecallCalculator(MetricCalculator):
    """Macro-averaged recall for classification."""

    @property
    def metric_name(self) -> str:
        return "recall"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.DETERMINISTIC

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        preds, refs = inputs.predictions, inputs.references
        if not preds or not refs or len(preds) != len(refs):
            return self._make_output(None, threshold, error="Need equal-length non-empty predictions/references")
        try:
            from sklearn.metrics import recall_score  # type: ignore
            value = float(recall_score(refs, preds, average="macro", zero_division=0))
        except ImportError:
            value = self._macro_recall(preds, refs)
        return self._make_output(value, threshold)

    @staticmethod
    def _macro_recall(preds: list, refs: list) -> float:
        labels = set(refs)
        scores = []
        for lbl in labels:
            tp = sum(1 for p, r in zip(preds, refs) if p == lbl and r == lbl)
            fn = sum(1 for p, r in zip(preds, refs) if p != lbl and r == lbl)
            denom = tp + fn
            scores.append(tp / denom if denom else 0.0)
        return sum(scores) / len(scores) if scores else 0.0


class F1Calculator(MetricCalculator):
    """Macro-averaged F1 score."""

    @property
    def metric_name(self) -> str:
        return "f1"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.DETERMINISTIC

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        preds, refs = inputs.predictions, inputs.references
        if not preds or not refs or len(preds) != len(refs):
            return self._make_output(None, threshold, error="Need equal-length non-empty predictions/references")
        try:
            from sklearn.metrics import f1_score  # type: ignore
            value = float(f1_score(refs, preds, average="macro", zero_division=0))
        except ImportError:
            # Derive from precision/recall calculators' pure-Python helpers
            p = PrecisionCalculator._macro_precision(preds, refs)
            r = RecallCalculator._macro_recall(preds, refs)
            value = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return self._make_output(value, threshold)


class RocAucCalculator(MetricCalculator):
    """
    ROC-AUC.  Requires probability scores in inputs.probabilities.
    For binary: probabilities[i] = [p_neg, p_pos]; value = p_pos.
    Multi-class: OvR macro-averaged via sklearn.
    """

    @property
    def metric_name(self) -> str:
        return "roc_auc"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.DETERMINISTIC

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        refs = inputs.references
        probs = inputs.probabilities
        if not refs or not probs:
            return self._make_output(None, threshold, error="references and probabilities required for roc_auc")
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore
            import numpy as np  # type: ignore
            probs_arr = np.array(probs)
            if probs_arr.shape[1] == 2:
                value = float(roc_auc_score(refs, probs_arr[:, 1]))
            else:
                value = float(roc_auc_score(refs, probs_arr, multi_class="ovr", average="macro"))
        except ImportError:
            return self._make_output(None, threshold, error="sklearn required for roc_auc")
        except Exception as exc:
            return self._make_output(None, threshold, error=str(exc))
        return self._make_output(value, threshold)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


class RmseCalculator(MetricCalculator):
    """Root Mean Squared Error. Lower is better; threshold is treated as upper bound."""

    @property
    def metric_name(self) -> str:
        return "rmse"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.DETERMINISTIC

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        preds, refs = inputs.predictions, inputs.references
        if not preds or not refs or len(preds) != len(refs):
            return self._make_output(None, threshold, error="Need equal-length non-empty predictions/references")
        try:
            mse = sum((float(p) - float(r)) ** 2 for p, r in zip(preds, refs)) / len(preds)
            value = math.sqrt(mse)
        except (TypeError, ValueError) as exc:
            return self._make_output(None, threshold, error=str(exc))
        # For RMSE, lower is better: invert the threshold sense in details
        output = self._make_output(value, threshold, details={"lower_is_better": True})
        return output


class MaeCalculator(MetricCalculator):
    """Mean Absolute Error. Lower is better."""

    @property
    def metric_name(self) -> str:
        return "mae"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.DETERMINISTIC

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        preds, refs = inputs.predictions, inputs.references
        if not preds or not refs or len(preds) != len(refs):
            return self._make_output(None, threshold, error="Need equal-length non-empty predictions/references")
        try:
            value = sum(abs(float(p) - float(r)) for p, r in zip(preds, refs)) / len(preds)
        except (TypeError, ValueError) as exc:
            return self._make_output(None, threshold, error=str(exc))
        return self._make_output(value, threshold, details={"lower_is_better": True})


class R2Calculator(MetricCalculator):
    """
    Coefficient of determination (R²).
    Range: (-inf, 1.0]; 1.0 = perfect prediction.
    """

    @property
    def metric_name(self) -> str:
        return "r2"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.DETERMINISTIC

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        preds, refs = inputs.predictions, inputs.references
        if not preds or not refs or len(preds) != len(refs):
            return self._make_output(None, threshold, error="Need equal-length non-empty predictions/references")
        try:
            refs_f = [float(r) for r in refs]
            preds_f = [float(p) for p in preds]
            mean_ref = sum(refs_f) / len(refs_f)
            ss_tot = sum((r - mean_ref) ** 2 for r in refs_f)
            ss_res = sum((r - p) ** 2 for r, p in zip(refs_f, preds_f))
            value = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 0.0
        except (TypeError, ValueError) as exc:
            return self._make_output(None, threshold, error=str(exc))
        return self._make_output(value, threshold)


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def build_traditional_ml_registry() -> dict[str, MetricCalculator]:
    """Return a dict mapping metric name → calculator for all traditional-ML metrics."""
    calculators: list[MetricCalculator] = [
        AccuracyCalculator(),
        PrecisionCalculator(),
        RecallCalculator(),
        F1Calculator(),
        RocAucCalculator(),
        RmseCalculator(),
        MaeCalculator(),
        R2Calculator(),
    ]
    return {c.metric_name: c for c in calculators}
