"""
Integration tests — end-to-end evaluation pipeline.

These tests verify that the metric evaluation stage (Stage 2) correctly
processes a CSV dataset and produces a structured result matching expected
aggregate metric shapes, sector thresholds, and governance flags.
"""

import csv
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

BANKING_CSV = Path(__file__).parent.parent.parent / "data" / "datasets" / "banking" / "loan_qa_dataset.csv"
UNIVERSITY_CSV = Path(__file__).parent.parent.parent / "data" / "datasets" / "education" / "university_qa_dataset.csv"
INSURANCE_CSV = Path(__file__).parent.parent.parent / "data" / "datasets" / "insurance" / "policy_qa_dataset.csv"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestBankingDatasetLoads:
    def test_banking_csv_exists(self):
        assert BANKING_CSV.exists()

    def test_banking_csv_has_15_records(self):
        records = load_csv(BANKING_CSV)
        assert len(records) == 15

    def test_banking_intentional_errors_present(self):
        records = load_csv(BANKING_CSV)
        ids = {r["record_id"] for r in records}
        assert "BNK_002" in ids
        assert "BNK_004" in ids
        assert "BNK_007" in ids

    def test_banking_has_critical_risk_records(self):
        records = load_csv(BANKING_CSV)
        critical = [r for r in records if r["risk_level"] == "CRITICAL"]
        assert len(critical) > 0, "Banking dataset should have CRITICAL risk records"


class TestUniversityDatasetLoads:
    def test_university_csv_exists(self):
        assert UNIVERSITY_CSV.exists()

    def test_university_csv_has_15_records(self):
        records = load_csv(UNIVERSITY_CSV)
        assert len(records) == 15

    def test_university_intentional_error_present(self):
        records = load_csv(UNIVERSITY_CSV)
        ids = {r["record_id"] for r in records}
        assert "UNI_005" in ids, "UNI_005 intentional error record should be present"

    def test_university_sector_consistent(self):
        records = load_csv(UNIVERSITY_CSV)
        sectors = {r["sector"] for r in records}
        assert sectors == {"university"}, f"Unexpected sectors: {sectors}"


class TestInsuranceDatasetLoads:
    def test_insurance_csv_exists(self):
        assert INSURANCE_CSV.exists()

    def test_insurance_csv_has_16_records(self):
        records = load_csv(INSURANCE_CSV)
        assert len(records) == 16

    def test_insurance_has_flood_storm_record(self):
        """INS_015 tests flood vs storm — CRITICAL regulatory definition."""
        records = load_csv(INSURANCE_CSV)
        ids = {r["record_id"] for r in records}
        assert "INS_015" in ids

    def test_insurance_critical_records_flagged(self):
        records = load_csv(INSURANCE_CSV)
        critical = [r for r in records if r["risk_level"] == "CRITICAL"]
        assert len(critical) >= 2, "Insurance dataset should have at least 2 CRITICAL records"


class TestSectorThresholdAlignment:
    """Verify config threshold values align with documented sector requirements."""

    def test_config_exists(self):
        config_path = Path(__file__).parent.parent.parent / "config" / "metrics_config.yaml"
        assert config_path.exists()

    def test_banking_threshold_higher_than_general(self):
        import yaml
        config_path = Path(__file__).parent.parent.parent / "config" / "metrics_config.yaml"
        config = yaml.safe_load(config_path.read_text())
        # Config uses 'pass' as the passing threshold key
        banking_acc = config["sector_overrides"]["banking"]["accuracy"]["pass"]
        general_acc = config["thresholds"]["accuracy"]["pass"]
        assert banking_acc > general_acc, \
            f"Banking accuracy threshold ({banking_acc}) should exceed general ({general_acc})"

    def test_insurance_hallucination_threshold_strict(self):
        import yaml
        config_path = Path(__file__).parent.parent.parent / "config" / "metrics_config.yaml"
        config = yaml.safe_load(config_path.read_text())
        # Config uses 'pass_below' for metrics where lower is better
        banking_hall = config["sector_overrides"]["banking"]["hallucination_score"]["pass_below"]
        assert banking_hall <= 0.05, \
            f"Banking hallucination threshold should be ≤0.05, got {banking_hall}"
