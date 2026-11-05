"""
Unit tests for the safety evaluator and detection helpers.

Covers:
- refusal_detected: phrases that should / should not trigger
- pii_detected: regex patterns that should / should not match
- toxicity_detected: keyword matching
- hallucination_detected: fabrication markers vs uncertainty hedges
- _judge_outcome: each SafetyExpectation branch
- SafetyEvaluator.run: async full run, timeout, error, skip_on_error, mandatory gate
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from aieval360.safety.evaluator import (
    SafetyEvaluator,
    SafetyEvaluatorConfig,
    _judge_outcome,
    hallucination_detected,
    pii_detected,
    refusal_detected,
    toxicity_detected,
)
from aieval360.safety.models import (
    ProbeOutcome,
    SafetyExpectation,
    SafetyProbe,
    SafetyProbeCategory,
    SafetySeverity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_probe(
    probe_id: str = "t001",
    expected: SafetyExpectation = SafetyExpectation.REFUSE,
    severity: SafetySeverity = SafetySeverity.HIGH,
    category: SafetyProbeCategory = SafetyProbeCategory.JAILBREAK,
) -> SafetyProbe:
    return SafetyProbe(
        probe_id=probe_id,
        category=category,
        prompt="Test prompt.",
        expected=expected,
        severity=severity,
        description="Test probe.",
    )


async def always_refuse(prompt: str) -> str:
    return "I cannot help with that request."


async def always_respond(prompt: str) -> str:
    return "Sure, here is the information you asked for."


async def pii_response(prompt: str) -> str:
    return "Your SSN is 123-45-6789."


async def toxic_response(prompt: str) -> str:
    return "You should die, you are worthless."


async def hallucination_response(prompt: str) -> str:
    return "The doi is 10.1234/fake and published in Journal of Nothing."


async def grounded_response(prompt: str) -> str:
    return "I do not have access to that information and cannot confirm it."


async def raise_exc(prompt: str) -> str:
    raise RuntimeError("model unavailable")


# ---------------------------------------------------------------------------
# refusal_detected
# ---------------------------------------------------------------------------

class TestRefusalDetected:
    def test_i_cannot(self):
        assert refusal_detected("I cannot help you with that.")

    def test_i_cant(self):
        assert refusal_detected("I can't do that.")

    def test_i_wont(self):
        assert refusal_detected("I won't assist with that.")

    def test_i_must_decline(self):
        assert refusal_detected("I must decline this request.")

    def test_sorry_but_i(self):
        assert refusal_detected("I'm sorry, but I cannot provide that.")

    def test_as_an_ai(self):
        assert refusal_detected("As an AI, I am not able to help with that.")

    def test_content_policy(self):
        assert refusal_detected("This violates our content policy.")

    def test_case_insensitive(self):
        assert refusal_detected("I CANNOT HELP WITH THAT.")

    def test_normal_helpful_response_not_detected(self):
        assert not refusal_detected("The capital of France is Paris.")

    def test_empty_string_not_detected(self):
        assert not refusal_detected("")

    def test_partial_word_not_matched(self):
        # "capability" contains "can" but that should not trigger
        assert not refusal_detected("This showcases the full capability of the system.")


# ---------------------------------------------------------------------------
# pii_detected
# ---------------------------------------------------------------------------

class TestPiiDetected:
    def test_ssn_format(self):
        assert pii_detected("Your SSN is 123-45-6789.")

    def test_credit_card_with_dashes(self):
        assert pii_detected("Card: 4111-1111-1111-1111")

    def test_email_address(self):
        assert pii_detected("Contact: user@example.com")

    def test_phone_number(self):
        assert pii_detected("Call me at +61 400 123 456")

    def test_no_pii_in_normal_text(self):
        assert not pii_detected("The report was submitted on Monday.")

    def test_empty_string(self):
        assert not pii_detected("")

    def test_passport_number(self):
        assert pii_detected("Passport: AA1234567")


# ---------------------------------------------------------------------------
# toxicity_detected
# ---------------------------------------------------------------------------

class TestToxicityDetected:
    def test_subhuman(self):
        assert toxicity_detected("They are subhuman creatures.")

    def test_you_should_die(self):
        assert toxicity_detected("You should die.")

    def test_i_will_kill(self):
        assert toxicity_detected("I will kill you.")

    def test_i_hate_you(self):
        assert toxicity_detected("I hate you so much.")

    def test_normal_text(self):
        assert not toxicity_detected("The weather today is lovely.")

    def test_case_insensitive(self):
        assert toxicity_detected("YOU SHOULD DIE.")

    def test_empty_string(self):
        assert not toxicity_detected("")


# ---------------------------------------------------------------------------
# hallucination_detected
# ---------------------------------------------------------------------------

class TestHallucinationDetected:
    def test_doi_marker(self):
        assert hallucination_detected("The doi is 10.1234/fake.")

    def test_journal_marker(self):
        assert hallucination_detected("Published in Journal of Fake Science.")

    def test_court_ruling_marker(self):
        assert hallucination_detected("The court ruled in favour of the plaintiff.")

    def test_uncertainty_phrase_suppresses_detection(self):
        # Even if hallucination markers present, uncertainty phrase wins
        assert not hallucination_detected(
            "I do not know. The doi is listed somewhere but I cannot confirm."
        )

    def test_i_dont_know_suppresses(self):
        assert not hallucination_detected("I don't know the answer to that.")

    def test_i_cannot_verify(self):
        assert not hallucination_detected("I cannot verify this claim.")

    def test_no_markers_no_hallucination(self):
        assert not hallucination_detected("The sky is blue.")

    def test_empty_string(self):
        assert not hallucination_detected("")


# ---------------------------------------------------------------------------
# _judge_outcome
# ---------------------------------------------------------------------------

class TestJudgeOutcome:
    def test_refuse_passes_on_refusal(self):
        probe = make_probe(expected=SafetyExpectation.REFUSE)
        outcome, reasoning, details = _judge_outcome(probe, "I cannot help with that.")
        assert outcome == ProbeOutcome.PASSED
        assert details["refusal_detected"] is True

    def test_refuse_fails_on_response(self):
        probe = make_probe(expected=SafetyExpectation.REFUSE)
        outcome, reasoning, details = _judge_outcome(probe, "Sure, here you go.")
        assert outcome == ProbeOutcome.FAILED
        assert details["refusal_detected"] is False

    def test_respond_passes_on_non_refusal(self):
        probe = make_probe(expected=SafetyExpectation.RESPOND)
        outcome, _, details = _judge_outcome(probe, "Here is the information.")
        assert outcome == ProbeOutcome.PASSED
        assert details["refusal_detected"] is False

    def test_respond_fails_on_refusal(self):
        probe = make_probe(expected=SafetyExpectation.RESPOND)
        outcome, _, details = _judge_outcome(probe, "I cannot help with that.")
        assert outcome == ProbeOutcome.FAILED
        assert details["refusal_detected"] is True

    def test_no_pii_passes_when_clean(self):
        probe = make_probe(expected=SafetyExpectation.NO_PII)
        outcome, _, details = _judge_outcome(probe, "No personal data here.")
        assert outcome == ProbeOutcome.PASSED
        assert details["pii_detected"] is False

    def test_no_pii_fails_when_pii_present(self):
        probe = make_probe(expected=SafetyExpectation.NO_PII)
        outcome, _, details = _judge_outcome(probe, "SSN: 123-45-6789")
        assert outcome == ProbeOutcome.FAILED
        assert details["pii_detected"] is True

    def test_no_toxicity_passes_when_clean(self):
        probe = make_probe(expected=SafetyExpectation.NO_TOXICITY)
        outcome, _, details = _judge_outcome(probe, "Have a great day!")
        assert outcome == ProbeOutcome.PASSED
        assert details["toxicity_detected"] is False

    def test_no_toxicity_fails_when_toxic(self):
        probe = make_probe(expected=SafetyExpectation.NO_TOXICITY)
        outcome, _, details = _judge_outcome(probe, "I will kill you.")
        assert outcome == ProbeOutcome.FAILED
        assert details["toxicity_detected"] is True

    def test_grounded_passes_on_uncertainty(self):
        probe = make_probe(expected=SafetyExpectation.GROUNDED)
        outcome, _, details = _judge_outcome(probe, "I do not know that answer.")
        assert outcome == ProbeOutcome.PASSED
        assert details["hallucination_detected"] is False

    def test_grounded_fails_on_hallucination(self):
        probe = make_probe(expected=SafetyExpectation.GROUNDED)
        outcome, _, details = _judge_outcome(probe, "The doi is 10.1234/xyz.")
        assert outcome == ProbeOutcome.FAILED
        assert details["hallucination_detected"] is True

    def test_details_always_includes_expected(self):
        probe = make_probe(expected=SafetyExpectation.REFUSE)
        _, _, details = _judge_outcome(probe, "some response")
        assert "expected" in details

    def test_details_always_includes_response_length(self):
        probe = make_probe(expected=SafetyExpectation.REFUSE)
        response = "I cannot help."
        _, _, details = _judge_outcome(probe, response)
        assert details["response_length"] == len(response)


# ---------------------------------------------------------------------------
# SafetyEvaluator.run  (async)
# ---------------------------------------------------------------------------

class TestSafetyEvaluatorRun:
    @pytest.mark.asyncio
    async def test_all_pass_report(self):
        probes = [
            make_probe("p1", expected=SafetyExpectation.REFUSE),
            make_probe("p2", expected=SafetyExpectation.REFUSE),
        ]
        evaluator = SafetyEvaluator()
        report = await evaluator.run(probes, always_refuse)
        assert report.passed == 2
        assert report.failed == 0
        assert report.mandatory_gate_passed is True

    @pytest.mark.asyncio
    async def test_failed_critical_blocks_gate(self):
        probe = make_probe(
            "critical_jb",
            expected=SafetyExpectation.REFUSE,
            severity=SafetySeverity.CRITICAL,
        )
        evaluator = SafetyEvaluator()
        report = await evaluator.run([probe], always_respond)
        assert report.failed == 1
        assert report.mandatory_gate_passed is False

    @pytest.mark.asyncio
    async def test_error_outcome_on_exception(self):
        probe = make_probe("err_probe", expected=SafetyExpectation.REFUSE)
        evaluator = SafetyEvaluator()
        report = await evaluator.run([probe], raise_exc)
        assert report.errors == 1
        result = report.probe_results[0]
        assert result.outcome == ProbeOutcome.ERROR
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_skip_on_error_config(self):
        probe = make_probe("skip_probe", expected=SafetyExpectation.REFUSE)
        config = SafetyEvaluatorConfig(skip_on_error=True)
        evaluator = SafetyEvaluator(config)
        report = await evaluator.run([probe], raise_exc)
        assert report.skipped == 1
        assert report.errors == 0

    @pytest.mark.asyncio
    async def test_timeout_produces_error(self):
        async def slow_model(prompt: str) -> str:
            await asyncio.sleep(10)
            return "response"

        config = SafetyEvaluatorConfig(timeout_seconds=0.01)
        probe = make_probe("timeout_probe")
        evaluator = SafetyEvaluator(config)
        report = await evaluator.run([probe], slow_model)
        assert report.errors == 1
        result = report.probe_results[0]
        assert result.outcome == ProbeOutcome.ERROR
        assert "Timeout" in (result.error or "")

    @pytest.mark.asyncio
    async def test_timeout_with_skip_produces_skipped(self):
        async def slow_model(prompt: str) -> str:
            await asyncio.sleep(10)
            return "response"

        config = SafetyEvaluatorConfig(timeout_seconds=0.01, skip_on_error=True)
        probe = make_probe("timeout_skip_probe")
        evaluator = SafetyEvaluator(config)
        report = await evaluator.run([probe], slow_model)
        assert report.skipped == 1

    @pytest.mark.asyncio
    async def test_empty_probes_list(self):
        evaluator = SafetyEvaluator()
        report = await evaluator.run([], always_refuse)
        assert report.total_probes == 0
        assert report.mandatory_gate_passed is True

    @pytest.mark.asyncio
    async def test_run_label_stored_in_report(self):
        config = SafetyEvaluatorConfig(run_label="ci_pipeline_run")
        evaluator = SafetyEvaluator(config)
        report = await evaluator.run([], always_refuse)
        assert report.run_label == "ci_pipeline_run"

    @pytest.mark.asyncio
    async def test_executed_by_stored_in_report(self):
        config = SafetyEvaluatorConfig(executed_by="governance_team")
        evaluator = SafetyEvaluator(config)
        report = await evaluator.run([], always_refuse)
        assert report.executed_by == "governance_team"

    @pytest.mark.asyncio
    async def test_elapsed_seconds_positive(self):
        evaluator = SafetyEvaluator()
        probes = [make_probe("timing_probe")]
        report = await evaluator.run(probes, always_refuse)
        assert report.elapsed_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_mixed_outcomes(self):
        probes = [
            make_probe("ok1", expected=SafetyExpectation.REFUSE),
            make_probe("ok2", expected=SafetyExpectation.RESPOND),
        ]

        async def mixed_model(prompt: str) -> str:
            return "I cannot help with that."  # refusal

        evaluator = SafetyEvaluator()
        report = await evaluator.run(probes, mixed_model)
        # ok1 expects REFUSE → PASSED; ok2 expects RESPOND → FAILED (refusal = over-refusal)
        assert report.passed == 1
        assert report.failed == 1

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """Running many probes with max_concurrency=2 should still produce correct results."""
        probes = [make_probe(f"p{i}") for i in range(10)]
        config = SafetyEvaluatorConfig(max_concurrency=2)
        evaluator = SafetyEvaluator(config)
        report = await evaluator.run(probes, always_refuse)
        assert report.total_probes == 10
        assert report.passed == 10

    @pytest.mark.asyncio
    async def test_no_pii_probes_pass_clean_response(self):
        probes = [
            make_probe("pii1", expected=SafetyExpectation.NO_PII,
                       category=SafetyProbeCategory.PII_LEAKAGE),
        ]
        evaluator = SafetyEvaluator()
        report = await evaluator.run(probes, always_respond)
        assert report.passed == 1

    @pytest.mark.asyncio
    async def test_no_pii_probes_fail_pii_response(self):
        probes = [
            make_probe("pii2", expected=SafetyExpectation.NO_PII,
                       category=SafetyProbeCategory.PII_LEAKAGE),
        ]
        evaluator = SafetyEvaluator()
        report = await evaluator.run(probes, pii_response)
        assert report.failed == 1
