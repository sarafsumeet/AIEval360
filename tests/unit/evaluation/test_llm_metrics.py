"""
Unit tests for LLM evaluation metrics.

BLEU and ROUGE use pure-Python fallbacks (sacrebleu / rouge-score optional).
FaithfulnessScaffold tests use pre-computed judge responses.
"""

from __future__ import annotations

import json

import pytest

from aieval360.evaluation.metrics.base import MetricInput
from aieval360.evaluation.metrics.llm_metrics import (
    BleuCalculator,
    FaithfulnessScaffoldCalculator,
    LexicalDiversityCalculator,
    NonEmptyResponseCalculator,
    ResponseLengthCalculator,
    RougeCalculator,
    build_llm_registry,
)
from aieval360.core.models.evaluation import MetricType


class TestBleu:
    def test_identical_gives_high_score(self):
        calc = BleuCalculator()
        text = "the quick brown fox jumps over the lazy dog"
        result = calc.calculate(MetricInput(predictions=[text], references=[text]))
        assert result.value is not None
        assert result.value > 0.9

    def test_empty_hypothesis_gives_zero(self):
        calc = BleuCalculator()
        result = calc.calculate(MetricInput(predictions=[""], references=["some reference text here"]))
        assert result.value == pytest.approx(0.0)

    def test_no_overlap_gives_zero(self):
        calc = BleuCalculator()
        result = calc.calculate(MetricInput(predictions=["xyz abc"], references=["qrs def ghi"]))
        assert result.value == pytest.approx(0.0)

    def test_metric_type_reference_based(self):
        assert BleuCalculator().metric_type == MetricType.REFERENCE_BASED

    def test_metric_name(self):
        assert BleuCalculator().metric_name == "bleu"

    def test_missing_inputs_returns_error(self):
        calc = BleuCalculator()
        result = calc.calculate(MetricInput())
        assert result.value is None
        assert result.error is not None

    def test_uses_generated_text_field(self):
        calc = BleuCalculator()
        text = "the quick brown fox"
        result = calc.calculate(MetricInput(generated_text=text, reference_text=text))
        assert result.value is not None
        assert result.value > 0.5

    def test_threshold_applied(self):
        calc = BleuCalculator()
        text = "the quick brown fox jumps over the lazy dog"
        result = calc.calculate(MetricInput(predictions=[text], references=[text]), threshold=0.5)
        assert result.passed is True


class TestRouge:
    def test_identical_gives_one(self):
        calc = RougeCalculator()
        text = "the quick brown fox"
        result = calc.calculate(MetricInput(predictions=[text], references=[text]))
        assert result.value is not None
        assert result.value == pytest.approx(1.0, abs=0.01)

    def test_no_overlap_gives_zero(self):
        calc = RougeCalculator()
        result = calc.calculate(MetricInput(predictions=["xyz abc"], references=["qrs def"]))
        assert result.value == pytest.approx(0.0)

    def test_partial_overlap(self):
        calc = RougeCalculator()
        result = calc.calculate(MetricInput(predictions=["the quick fox"], references=["the lazy fox"]))
        assert result.value is not None
        assert 0.0 < result.value < 1.0

    def test_metric_type_reference_based(self):
        assert RougeCalculator().metric_type == MetricType.REFERENCE_BASED

    def test_metric_name(self):
        assert RougeCalculator().metric_name == "rouge"

    def test_missing_inputs_returns_error(self):
        result = RougeCalculator().calculate(MetricInput())
        assert result.error is not None


class TestResponseLength:
    def test_counts_tokens(self):
        calc = ResponseLengthCalculator()
        result = calc.calculate(MetricInput(generated_text="hello world foo bar"))
        assert result.value == pytest.approx(4.0)

    def test_empty_text_gives_zero(self):
        calc = ResponseLengthCalculator()
        result = calc.calculate(MetricInput(generated_text=""))
        assert result.value == pytest.approx(0.0)

    def test_metric_type_heuristic(self):
        assert ResponseLengthCalculator().metric_type == MetricType.HEURISTIC

    def test_metric_name(self):
        assert ResponseLengthCalculator().metric_name == "response_length"

    def test_details_contains_token_count(self):
        calc = ResponseLengthCalculator()
        result = calc.calculate(MetricInput(generated_text="one two three"))
        assert result.details["token_count"] == 3


class TestNonEmptyResponse:
    def test_non_empty_gives_one(self):
        calc = NonEmptyResponseCalculator()
        result = calc.calculate(MetricInput(generated_text="hello"))
        assert result.value == pytest.approx(1.0)

    def test_empty_gives_zero(self):
        calc = NonEmptyResponseCalculator()
        result = calc.calculate(MetricInput(generated_text=""))
        assert result.value == pytest.approx(0.0)

    def test_whitespace_only_gives_zero(self):
        calc = NonEmptyResponseCalculator()
        result = calc.calculate(MetricInput(generated_text="   \n  "))
        assert result.value == pytest.approx(0.0)

    def test_metric_type_heuristic(self):
        assert NonEmptyResponseCalculator().metric_type == MetricType.HEURISTIC

    def test_as_mandatory_gate_threshold(self):
        calc = NonEmptyResponseCalculator()
        result = calc.calculate(MetricInput(generated_text=""), threshold=1.0)
        assert result.passed is False


class TestLexicalDiversity:
    def test_all_unique_gives_one(self):
        calc = LexicalDiversityCalculator()
        result = calc.calculate(MetricInput(generated_text="alpha beta gamma delta"))
        assert result.value == pytest.approx(1.0)

    def test_all_repeated_gives_low_score(self):
        calc = LexicalDiversityCalculator()
        result = calc.calculate(MetricInput(generated_text="the the the the"))
        assert result.value == pytest.approx(0.25)  # 1/4 unique

    def test_empty_gives_zero(self):
        calc = LexicalDiversityCalculator()
        result = calc.calculate(MetricInput(generated_text=""))
        assert result.value == pytest.approx(0.0)

    def test_metric_type_heuristic(self):
        assert LexicalDiversityCalculator().metric_type == MetricType.HEURISTIC

    def test_range_zero_to_one(self):
        calc = LexicalDiversityCalculator()
        result = calc.calculate(MetricInput(generated_text="the quick brown fox"))
        assert 0.0 <= result.value <= 1.0


class TestFaithfulnessScaffold:
    def _judge_json(self, score: float, reasoning: str = "test") -> str:
        return json.dumps({"score": score, "reasoning": reasoning, "passed": score >= 0.7})

    def test_valid_judge_response_parsed(self):
        calc = FaithfulnessScaffoldCalculator()
        result = calc.calculate(MetricInput(judge_response=self._judge_json(0.85)))
        assert result.value == pytest.approx(0.85)

    def test_reasoning_captured(self):
        calc = FaithfulnessScaffoldCalculator()
        result = calc.calculate(MetricInput(judge_response=self._judge_json(0.9, "Response is faithful")))
        assert "faithful" in result.reasoning

    def test_missing_judge_response_gives_error(self):
        calc = FaithfulnessScaffoldCalculator()
        result = calc.calculate(MetricInput())
        assert result.value is None
        assert result.error is not None

    def test_malformed_json_gives_error(self):
        calc = FaithfulnessScaffoldCalculator()
        result = calc.calculate(MetricInput(judge_response="not json {{"))
        assert result.value is None
        assert result.error is not None

    def test_metric_type_llm_judged(self):
        assert FaithfulnessScaffoldCalculator().metric_type == MetricType.LLM_JUDGED

    def test_metric_name(self):
        assert FaithfulnessScaffoldCalculator().metric_name == "faithfulness"

    def test_threshold_applied(self):
        calc = FaithfulnessScaffoldCalculator()
        result = calc.calculate(MetricInput(judge_response=self._judge_json(0.6)), threshold=0.7)
        assert result.passed is False

    def test_threshold_pass(self):
        calc = FaithfulnessScaffoldCalculator()
        result = calc.calculate(MetricInput(judge_response=self._judge_json(0.9)), threshold=0.7)
        assert result.passed is True


class TestLLMRegistry:
    def test_all_expected_metrics_registered(self):
        registry = build_llm_registry()
        expected = {
            "bleu", "rouge",
            "response_length", "non_empty_response", "lexical_diversity",
            "faithfulness",
        }
        assert expected == set(registry.keys())

    def test_registry_includes_llm_judged(self):
        registry = build_llm_registry()
        llm_judged = [n for n, c in registry.items() if c.metric_type == MetricType.LLM_JUDGED]
        assert "faithfulness" in llm_judged
