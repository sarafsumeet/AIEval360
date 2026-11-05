"""
Safety evaluation domain models.

Defines the core data structures for safety probing:
- SafetyProbeCategory  — taxonomy of probe types
- SafetyProbe          — a single test case with expected safe behaviour
- ProbeResult          — outcome of running one probe against an AI response
- SafetyEvaluationReport — aggregated report across all probes in a run

Design decisions
----------------
- Probes are data, not code: the evaluator decides how to judge each probe.
- Expected behaviour is expressed as a SafetyExpectation enum so comparisons
  are unambiguous in test assertions and audit logs.
- All results carry a `severity` field matching the probe's declared severity,
  so governance reports can gate on critical failures independently.
- The report's `mandatory_gate_passed` property is the single boolean that
  CI/CD pipelines and governance committees should read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SafetyProbeCategory(str, Enum):
    """Taxonomy of safety probe types."""
    PROMPT_INJECTION   = "prompt_injection"    # attempt to override system instructions
    JAILBREAK          = "jailbreak"           # attempt to bypass safety guardrails
    PII_LEAKAGE        = "pii_leakage"         # extraction of personal/sensitive data
    TOXICITY           = "toxicity"            # harmful, abusive or offensive content
    HALLUCINATION      = "hallucination"       # fabricated facts presented as true
    BIAS               = "bias"                # discriminatory or unfair output


class SafetySeverity(str, Enum):
    """Risk severity of a probe failure."""
    CRITICAL = "critical"   # mandatory gate — blocks deployment
    HIGH     = "high"       # must be investigated before release
    MEDIUM   = "medium"     # tracked and triaged
    LOW      = "low"        # informational / best-effort


class SafetyExpectation(str, Enum):
    """
    What safe behaviour looks like for a probe.

    REFUSE      — model should decline to answer
    RESPOND     — model should answer (not over-refuse)
    NO_PII      — response must not contain PII markers
    NO_TOXICITY — response must not contain toxic content
    GROUNDED    — response must be grounded (no hallucination markers)
    """
    REFUSE      = "refuse"
    RESPOND     = "respond"
    NO_PII      = "no_pii"
    NO_TOXICITY = "no_toxicity"
    GROUNDED    = "grounded"


class ProbeOutcome(str, Enum):
    """Outcome of a single probe run."""
    PASSED  = "passed"   # model behaved as expected
    FAILED  = "failed"   # model violated safety expectation
    ERROR   = "error"    # evaluation could not be completed
    SKIPPED = "skipped"  # probe not applicable in this context


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class SafetyProbe:
    """
    A single adversarial test case.

    probe_id        Unique identifier (e.g. "pi_001").
    category        SafetyProbeCategory enum.
    prompt          The adversarial prompt to send to the model.
    expected        SafetyExpectation describing safe behaviour.
    severity        How critical failure is.
    description     Human-readable explanation of what this tests.
    tags            Free-form labels for filtering (e.g. ["finance", "gdpr"]).
    metadata        Any extra context (sector, regulation, source, etc.).
    """

    probe_id:    str
    category:    SafetyProbeCategory
    prompt:      str
    expected:    SafetyExpectation
    severity:    SafetySeverity = SafetySeverity.HIGH
    description: str = ""
    tags:        list[str] = field(default_factory=list)
    metadata:    dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("probe_id must not be empty")
        if not self.prompt:
            raise ValueError("prompt must not be empty")


@dataclass
class ProbeResult:
    """
    Outcome of running one SafetyProbe against an AI response.

    probe        The probe that was run.
    response     The raw AI response (empty string if none).
    outcome      ProbeOutcome enum.
    score        Numeric safety score 0.0–1.0 (1.0 = fully safe).
    details      Breakdown dict (matched_patterns, refusal_detected, etc.).
    reasoning    Human-readable explanation of the outcome.
    error        Set if evaluation raised an exception.
    """

    probe:     SafetyProbe
    response:  str
    outcome:   ProbeOutcome
    score:     Optional[float] = None
    details:   dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    error:     Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.outcome == ProbeOutcome.PASSED

    @property
    def failed(self) -> bool:
        return self.outcome == ProbeOutcome.FAILED

    @property
    def is_critical_failure(self) -> bool:
        return self.failed and self.probe.severity == SafetySeverity.CRITICAL


@dataclass
class SafetyEvaluationReport:
    """
    Aggregated safety evaluation report for one run.

    probe_results       All ProbeResult objects from this run.
    total_probes        Total number of probes attempted.
    passed              Count of PASSED probes.
    failed              Count of FAILED probes.
    errors              Count of ERROR probes.
    skipped             Count of SKIPPED probes.
    critical_failures   List of ProbeResult objects with is_critical_failure=True.
    elapsed_seconds     Wall-clock time for the full run.
    run_label           Human-readable label for this run.
    executed_by         Identity of person / system that triggered the run.
    metadata            Any extra run-level context.
    """

    probe_results:     list[ProbeResult] = field(default_factory=list)
    total_probes:      int = 0
    passed:            int = 0
    failed:            int = 0
    errors:            int = 0
    skipped:           int = 0
    elapsed_seconds:   float = 0.0
    run_label:         str = "safety_evaluation"
    executed_by:       str = "system"
    metadata:          dict[str, Any] = field(default_factory=dict)

    @property
    def critical_failures(self) -> list[ProbeResult]:
        return [r for r in self.probe_results if r.is_critical_failure]

    @property
    def mandatory_gate_passed(self) -> bool:
        """True only when zero critical failures were recorded."""
        return len(self.critical_failures) == 0

    @property
    def pass_rate(self) -> Optional[float]:
        """Fraction of probes that passed (excluding skipped)."""
        attempted = self.total_probes - self.skipped
        if attempted == 0:
            return None
        return self.passed / attempted

    @property
    def overall_status(self) -> str:
        if not self.mandatory_gate_passed:
            return "critical_failure"
        if self.failed > 0:
            return "failed"
        if self.errors > 0:
            return "completed_with_errors"
        return "passed"

    def results_by_category(self, category: SafetyProbeCategory) -> list[ProbeResult]:
        return [r for r in self.probe_results if r.probe.category == category]

    def results_by_severity(self, severity: SafetySeverity) -> list[ProbeResult]:
        return [r for r in self.probe_results if r.probe.severity == severity]
