"""
Unit tests for safety evaluation domain models.

Covers:
- All four enumerations (string values, membership)
- SafetyProbe: construction, validation, defaults
- ProbeResult: properties (passed, failed, is_critical_failure)
- SafetyEvaluationReport: aggregation, properties, filtering methods
"""

from __future__ import annotations

import pytest

from aieval360.safety.models import (
    ProbeOutcome,
    ProbeResult,
    SafetyEvaluationReport,
    SafetyExpectation,
    SafetyProbe,
    SafetyProbeCategory,
    SafetySeverity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_probe(
    probe_id: str = "test_001",
    category: SafetyProbeCategory = SafetyProbeCategory.JAILBREAK,
    expected: SafetyExpectation = SafetyExpectation.REFUSE,
    severity: SafetySeverity = SafetySeverity.HIGH,
) -> SafetyProbe:
    return SafetyProbe(
        probe_id=probe_id,
        category=category,
        prompt="Test adversarial prompt.",
        expected=expected,
        severity=severity,
        description="A test probe.",
    )


def make_result(
    probe: SafetyProbe | None = None,
    outcome: ProbeOutcome = ProbeOutcome.PASSED,
    response: str = "I cannot help with that.",
) -> ProbeResult:
    if probe is None:
        probe = make_probe()
    return ProbeResult(probe=probe, response=response, outcome=outcome)


def make_report(results: list[ProbeResult]) -> SafetyEvaluationReport:
    passed = sum(1 for r in results if r.outcome == ProbeOutcome.PASSED)
    failed = sum(1 for r in results if r.outcome == ProbeOutcome.FAILED)
    errors = sum(1 for r in results if r.outcome == ProbeOutcome.ERROR)
    skipped = sum(1 for r in results if r.outcome == ProbeOutcome.SKIPPED)
    return SafetyEvaluationReport(
        probe_results=results,
        total_probes=len(results),
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestSafetyProbeCategory:
    def test_values_are_strings(self):
        for cat in SafetyProbeCategory:
            assert isinstance(cat.value, str)

    def test_expected_members(self):
        names = {c.name for c in SafetyProbeCategory}
        assert "PROMPT_INJECTION" in names
        assert "JAILBREAK" in names
        assert "PII_LEAKAGE" in names
        assert "TOXICITY" in names
        assert "HALLUCINATION" in names
        assert "BIAS" in names

    def test_string_coercion(self):
        assert SafetyProbeCategory("jailbreak") == SafetyProbeCategory.JAILBREAK


class TestSafetySeverity:
    def test_all_values(self):
        assert SafetySeverity.CRITICAL.value == "critical"
        assert SafetySeverity.HIGH.value == "high"
        assert SafetySeverity.MEDIUM.value == "medium"
        assert SafetySeverity.LOW.value == "low"

    def test_total_count(self):
        assert len(SafetySeverity) == 4


class TestSafetyExpectation:
    def test_all_values(self):
        expected = {"refuse", "respond", "no_pii", "no_toxicity", "grounded"}
        actual = {e.value for e in SafetyExpectation}
        assert expected == actual


class TestProbeOutcome:
    def test_all_values(self):
        expected = {"passed", "failed", "error", "skipped"}
        actual = {o.value for o in ProbeOutcome}
        assert expected == actual


# ---------------------------------------------------------------------------
# SafetyProbe tests
# ---------------------------------------------------------------------------

class TestSafetyProbe:
    def test_basic_construction(self):
        probe = make_probe()
        assert probe.probe_id == "test_001"
        assert probe.category == SafetyProbeCategory.JAILBREAK
        assert probe.expected == SafetyExpectation.REFUSE
        assert probe.severity == SafetySeverity.HIGH

    def test_default_severity_is_high(self):
        probe = SafetyProbe(
            probe_id="p1",
            category=SafetyProbeCategory.TOXICITY,
            prompt="test",
            expected=SafetyExpectation.NO_TOXICITY,
        )
        assert probe.severity == SafetySeverity.HIGH

    def test_default_tags_is_empty_list(self):
        probe = make_probe()
        assert probe.tags == []

    def test_default_metadata_is_empty_dict(self):
        probe = make_probe()
        assert probe.metadata == {}

    def test_tags_are_mutable_per_instance(self):
        p1 = make_probe("a")
        p2 = make_probe("b")
        p1.tags.append("x")
        assert p2.tags == []

    def test_empty_probe_id_raises(self):
        with pytest.raises(ValueError, match="probe_id"):
            SafetyProbe(
                probe_id="",
                category=SafetyProbeCategory.JAILBREAK,
                prompt="test",
                expected=SafetyExpectation.REFUSE,
            )

    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError, match="prompt"):
            SafetyProbe(
                probe_id="p1",
                category=SafetyProbeCategory.JAILBREAK,
                prompt="",
                expected=SafetyExpectation.REFUSE,
            )

    def test_all_categories_valid(self):
        for cat in SafetyProbeCategory:
            probe = SafetyProbe(
                probe_id=f"test_{cat.value}",
                category=cat,
                prompt="x",
                expected=SafetyExpectation.REFUSE,
            )
            assert probe.category == cat


# ---------------------------------------------------------------------------
# ProbeResult tests
# ---------------------------------------------------------------------------

class TestProbeResult:
    def test_passed_property_true(self):
        result = make_result(outcome=ProbeOutcome.PASSED)
        assert result.passed is True
        assert result.failed is False

    def test_failed_property_true(self):
        result = make_result(outcome=ProbeOutcome.FAILED)
        assert result.failed is True
        assert result.passed is False

    def test_error_outcome_not_passed(self):
        result = make_result(outcome=ProbeOutcome.ERROR)
        assert result.passed is False
        assert result.failed is False

    def test_is_critical_failure_requires_failed_and_critical(self):
        critical_probe = make_probe(severity=SafetySeverity.CRITICAL)
        result = make_result(probe=critical_probe, outcome=ProbeOutcome.FAILED)
        assert result.is_critical_failure is True

    def test_is_critical_failure_false_if_high_severity(self):
        high_probe = make_probe(severity=SafetySeverity.HIGH)
        result = make_result(probe=high_probe, outcome=ProbeOutcome.FAILED)
        assert result.is_critical_failure is False

    def test_is_critical_failure_false_if_critical_but_passed(self):
        critical_probe = make_probe(severity=SafetySeverity.CRITICAL)
        result = make_result(probe=critical_probe, outcome=ProbeOutcome.PASSED)
        assert result.is_critical_failure is False

    def test_default_score_is_none(self):
        result = make_result()
        assert result.score is None

    def test_score_can_be_set(self):
        result = make_result()
        result.score = 0.95
        assert result.score == pytest.approx(0.95)

    def test_details_default_empty(self):
        result = make_result()
        assert result.details == {}

    def test_reasoning_default_empty_string(self):
        result = make_result()
        assert result.reasoning == ""


# ---------------------------------------------------------------------------
# SafetyEvaluationReport tests
# ---------------------------------------------------------------------------

class TestSafetyEvaluationReport:
    def test_empty_report(self):
        report = SafetyEvaluationReport()
        assert report.total_probes == 0
        assert report.mandatory_gate_passed is True  # no failures → gate passes
        assert report.pass_rate is None
        assert report.overall_status == "passed"

    def test_all_passed(self):
        results = [make_result(outcome=ProbeOutcome.PASSED) for _ in range(5)]
        report = make_report(results)
        assert report.mandatory_gate_passed is True
        assert report.pass_rate == pytest.approx(1.0)
        assert report.overall_status == "passed"

    def test_critical_failure_gates_mandatory(self):
        critical_probe = make_probe(severity=SafetySeverity.CRITICAL)
        result = make_result(probe=critical_probe, outcome=ProbeOutcome.FAILED)
        report = make_report([result])
        assert report.mandatory_gate_passed is False
        assert report.overall_status == "critical_failure"

    def test_non_critical_failure_does_not_block_gate(self):
        high_probe = make_probe(severity=SafetySeverity.HIGH)
        result = make_result(probe=high_probe, outcome=ProbeOutcome.FAILED)
        report = make_report([result])
        assert report.mandatory_gate_passed is True
        assert report.overall_status == "failed"

    def test_overall_status_completed_with_errors(self):
        result = make_result(outcome=ProbeOutcome.ERROR)
        report = make_report([result])
        # No failed, no critical — only an error
        assert report.overall_status == "completed_with_errors"

    def test_pass_rate_excludes_skipped(self):
        results = [
            make_result(outcome=ProbeOutcome.PASSED),
            make_result(outcome=ProbeOutcome.PASSED),
            make_result(outcome=ProbeOutcome.SKIPPED),
        ]
        report = make_report(results)
        # Skipped excluded: 2 passed / 2 attempted = 1.0
        assert report.pass_rate == pytest.approx(1.0)

    def test_pass_rate_all_skipped_returns_none(self):
        results = [make_result(outcome=ProbeOutcome.SKIPPED) for _ in range(3)]
        report = make_report(results)
        assert report.pass_rate is None

    def test_critical_failures_list(self):
        critical_probe = make_probe(severity=SafetySeverity.CRITICAL)
        r1 = make_result(probe=critical_probe, outcome=ProbeOutcome.FAILED)
        r2 = make_result(outcome=ProbeOutcome.PASSED)
        report = make_report([r1, r2])
        assert len(report.critical_failures) == 1

    def test_results_by_category(self):
        pii_probe = make_probe(category=SafetyProbeCategory.PII_LEAKAGE)
        jb_probe = make_probe(probe_id="jb_001", category=SafetyProbeCategory.JAILBREAK)
        r1 = make_result(probe=pii_probe)
        r2 = make_result(probe=jb_probe)
        report = make_report([r1, r2])
        pii = report.results_by_category(SafetyProbeCategory.PII_LEAKAGE)
        assert len(pii) == 1
        assert pii[0].probe.category == SafetyProbeCategory.PII_LEAKAGE

    def test_results_by_severity(self):
        critical_probe = make_probe(severity=SafetySeverity.CRITICAL)
        low_probe = make_probe(probe_id="low_001", severity=SafetySeverity.LOW)
        report = make_report([make_result(probe=critical_probe), make_result(probe=low_probe)])
        criticals = report.results_by_severity(SafetySeverity.CRITICAL)
        assert len(criticals) == 1

    def test_multiple_critical_failures(self):
        probes = [make_probe(probe_id=f"c{i}", severity=SafetySeverity.CRITICAL) for i in range(3)]
        results = [make_result(probe=p, outcome=ProbeOutcome.FAILED) for p in probes]
        report = make_report(results)
        assert report.mandatory_gate_passed is False
        assert len(report.critical_failures) == 3
