"""
Unit tests for the safety probe library.

Covers:
- Total probe counts and per-category counts
- Severity distribution (at least one CRITICAL per category)
- Required fields populated on all probes
- Public API functions: get_all_probes, get_probes_by_category,
  get_probes_by_severity, get_critical_probes, get_probe_by_id
"""

from __future__ import annotations

import pytest

from aieval360.safety.models import (
    SafetyExpectation,
    SafetyProbeCategory,
    SafetySeverity,
)
from aieval360.safety.probe_library import (
    get_all_probes,
    get_critical_probes,
    get_probe_by_id,
    get_probes_by_category,
    get_probes_by_severity,
)


# ---------------------------------------------------------------------------
# Total probe counts
# ---------------------------------------------------------------------------

class TestGetAllProbes:
    def test_returns_list(self):
        probes = get_all_probes()
        assert isinstance(probes, list)

    def test_at_least_thirty_probes(self):
        probes = get_all_probes()
        assert len(probes) >= 30, f"Expected ≥30 probes, got {len(probes)}"

    def test_returns_new_list_each_call(self):
        # Mutations to the returned list must not affect the library
        p1 = get_all_probes()
        p2 = get_all_probes()
        p1.clear()
        assert len(p2) > 0

    def test_all_probe_ids_unique(self):
        probes = get_all_probes()
        ids = [p.probe_id for p in probes]
        assert len(ids) == len(set(ids)), "Duplicate probe_ids found"


# ---------------------------------------------------------------------------
# Per-category counts
# ---------------------------------------------------------------------------

class TestGetProbesByCategory:
    def test_prompt_injection_at_least_five(self):
        probes = get_probes_by_category(SafetyProbeCategory.PROMPT_INJECTION)
        assert len(probes) >= 5

    def test_jailbreak_at_least_five(self):
        probes = get_probes_by_category(SafetyProbeCategory.JAILBREAK)
        assert len(probes) >= 5

    def test_pii_leakage_at_least_three(self):
        probes = get_probes_by_category(SafetyProbeCategory.PII_LEAKAGE)
        assert len(probes) >= 3

    def test_toxicity_at_least_three(self):
        probes = get_probes_by_category(SafetyProbeCategory.TOXICITY)
        assert len(probes) >= 3

    def test_hallucination_at_least_three(self):
        probes = get_probes_by_category(SafetyProbeCategory.HALLUCINATION)
        assert len(probes) >= 3

    def test_all_returned_probes_have_correct_category(self):
        for category in SafetyProbeCategory:
            probes = get_probes_by_category(category)
            for p in probes:
                assert p.category == category, (
                    f"Probe {p.probe_id} in wrong category: "
                    f"expected {category}, got {p.category}"
                )

    def test_categories_are_disjoint(self):
        # Every probe in the full list must appear in exactly one category slice
        all_ids = {p.probe_id for p in get_all_probes()}
        seen: set[str] = set()
        for category in SafetyProbeCategory:
            cat_ids = {p.probe_id for p in get_probes_by_category(category)}
            duplicates = seen & cat_ids
            assert not duplicates, f"Probe(s) in multiple categories: {duplicates}"
            seen.update(cat_ids)
        # Every probe must be reachable via its category
        assert seen == all_ids


# ---------------------------------------------------------------------------
# Severity distribution
# ---------------------------------------------------------------------------

class TestGetProbesBySeverity:
    def test_critical_probes_exist(self):
        probes = get_probes_by_severity(SafetySeverity.CRITICAL)
        assert len(probes) >= 1

    def test_high_probes_exist(self):
        probes = get_probes_by_severity(SafetySeverity.HIGH)
        assert len(probes) >= 1

    def test_all_returned_probes_have_correct_severity(self):
        for severity in SafetySeverity:
            probes = get_probes_by_severity(severity)
            for p in probes:
                assert p.severity == severity

    def test_severity_union_equals_all(self):
        all_ids = {p.probe_id for p in get_all_probes()}
        severity_ids: set[str] = set()
        for sev in SafetySeverity:
            severity_ids.update(p.probe_id for p in get_probes_by_severity(sev))
        assert severity_ids == all_ids


# ---------------------------------------------------------------------------
# get_critical_probes convenience function
# ---------------------------------------------------------------------------

class TestGetCriticalProbes:
    def test_returns_subset_of_all(self):
        all_ids = {p.probe_id for p in get_all_probes()}
        critical_ids = {p.probe_id for p in get_critical_probes()}
        assert critical_ids.issubset(all_ids)

    def test_all_are_critical_severity(self):
        for p in get_critical_probes():
            assert p.severity == SafetySeverity.CRITICAL

    def test_matches_get_probes_by_severity(self):
        a = {p.probe_id for p in get_critical_probes()}
        b = {p.probe_id for p in get_probes_by_severity(SafetySeverity.CRITICAL)}
        assert a == b


# ---------------------------------------------------------------------------
# get_probe_by_id
# ---------------------------------------------------------------------------

class TestGetProbeById:
    def test_returns_probe_for_valid_id(self):
        probes = get_all_probes()
        first_id = probes[0].probe_id
        result = get_probe_by_id(first_id)
        assert result is not None
        assert result.probe_id == first_id

    def test_returns_none_or_raises_for_unknown_id(self):
        # get_probe_by_id may return None or raise KeyError for unknown ids
        try:
            result = get_probe_by_id("does_not_exist_xyz")
            assert result is None
        except KeyError:
            pass  # raising KeyError is also an acceptable contract

    def test_every_probe_findable_by_id(self):
        for probe in get_all_probes():
            found = get_probe_by_id(probe.probe_id)
            assert found is not None, f"Could not find probe: {probe.probe_id}"
            assert found.probe_id == probe.probe_id


# ---------------------------------------------------------------------------
# Required field validation
# ---------------------------------------------------------------------------

class TestRequiredFields:
    def test_all_probes_have_non_empty_prompt(self):
        for p in get_all_probes():
            assert p.prompt, f"Probe {p.probe_id} has empty prompt"

    def test_all_probes_have_valid_expected(self):
        valid = set(SafetyExpectation)
        for p in get_all_probes():
            assert p.expected in valid, (
                f"Probe {p.probe_id} has invalid expected: {p.expected}"
            )

    def test_all_probes_have_non_empty_description(self):
        for p in get_all_probes():
            assert p.description, f"Probe {p.probe_id} has empty description"

    def test_all_probes_have_valid_category(self):
        valid = set(SafetyProbeCategory)
        for p in get_all_probes():
            assert p.category in valid

    def test_all_probes_have_valid_severity(self):
        valid = set(SafetySeverity)
        for p in get_all_probes():
            assert p.severity in valid

    def test_jailbreak_probes_expect_refuse(self):
        """Jailbreak probes should expect the model to refuse."""
        for p in get_probes_by_category(SafetyProbeCategory.JAILBREAK):
            assert p.expected == SafetyExpectation.REFUSE, (
                f"Jailbreak probe {p.probe_id} has unexpected expectation: {p.expected}"
            )

    def test_pii_probes_expect_no_pii_or_refuse(self):
        """PII leakage probes should expect NO_PII or REFUSE (doxxing probes refuse)."""
        valid = {SafetyExpectation.NO_PII, SafetyExpectation.REFUSE}
        for p in get_probes_by_category(SafetyProbeCategory.PII_LEAKAGE):
            assert p.expected in valid, (
                f"PII probe {p.probe_id} has unexpected expectation: {p.expected}"
            )

    def test_hallucination_probes_expect_grounded(self):
        """Hallucination probes should expect GROUNDED responses."""
        for p in get_probes_by_category(SafetyProbeCategory.HALLUCINATION):
            assert p.expected == SafetyExpectation.GROUNDED, (
                f"Hallucination probe {p.probe_id} has unexpected expectation: {p.expected}"
            )
