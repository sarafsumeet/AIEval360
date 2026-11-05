"""
Tests for evidence fixture files — ensure golden outputs are valid JSON
and contain required fields for contract testing.
"""

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    path = FIXTURES_DIR / name
    assert path.exists(), f"Fixture not found: {path}"
    return json.loads(path.read_text())


class TestGoldenBankingRun:
    def test_fixture_loads(self):
        data = load_fixture("golden_banking_run.json")
        assert isinstance(data, dict)

    def test_required_fields_present(self):
        data = load_fixture("golden_banking_run.json")
        for field in ["run_id", "sector", "records_evaluated", "run_status", "aggregate_metrics"]:
            assert field in data, f"Missing field: {field}"

    def test_sector_is_banking(self):
        data = load_fixture("golden_banking_run.json")
        assert data["sector"] == "banking"

    def test_records_evaluated(self):
        data = load_fixture("golden_banking_run.json")
        assert data["records_evaluated"] == 15

    def test_intentional_errors_flagged(self):
        data = load_fixture("golden_banking_run.json")
        errors = data.get("intentional_errors", [])
        assert "BNK_002" in errors
        assert "BNK_004" in errors
        assert "BNK_007" in errors

    def test_accuracy_below_banking_threshold(self):
        """Banking accuracy threshold is 0.92; fixture accuracy is 0.80 — should FAIL."""
        data = load_fixture("golden_banking_run.json")
        acc = data["aggregate_metrics"]["accuracy"]
        assert acc["value"] < acc["threshold"]
        assert acc["status"] == "FAIL"

    def test_governance_flag_raised(self):
        data = load_fixture("golden_banking_run.json")
        assert "BELOW_SECTOR_THRESHOLD" in data.get("governance_flags", [])


class TestGoldenSafetyReport:
    def test_fixture_loads(self):
        data = load_fixture("golden_safety_report.json")
        assert isinstance(data, dict)

    def test_mandatory_gate_passed(self):
        """Banking sector: no CRITICAL failures → gate must PASS."""
        data = load_fixture("golden_safety_report.json")
        assert data["mandatory_gate_passed"] is True

    def test_no_critical_failures(self):
        data = load_fixture("golden_safety_report.json")
        assert data["summary"]["critical_failures"] == 0

    def test_all_categories_present(self):
        data = load_fixture("golden_safety_report.json")
        expected = {"PROMPT_INJECTION", "JAILBREAK", "PII_LEAKAGE", "TOXICITY", "HALLUCINATION"}
        actual = set(data["category_summary"].keys())
        assert expected == actual

    def test_probe_counts_sum(self):
        data = load_fixture("golden_safety_report.json")
        summary = data["summary"]
        total = summary["passed"] + summary["failed"] + summary["errored"]
        assert total == data["total_probes"]
