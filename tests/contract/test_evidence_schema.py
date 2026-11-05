"""
Contract tests — validate that evidence JSON files in evidence/evaluation_runs/
conform to data/schemas/evidence_run_schema.json.
"""

import json
import re
from pathlib import Path

import pytest

EVIDENCE_DIR = Path(__file__).parent.parent.parent / "evidence" / "evaluation_runs"
SCHEMA_PATH = Path(__file__).parent.parent.parent / "data" / "schemas" / "evidence_run_schema.json"

REQUIRED_FIELDS = {
    "run_id", "run_timestamp", "dataset", "dataset_checksum",
    "sector", "records_evaluated", "run_status", "aggregate_metrics",
}
VALID_STATUSES = {"COMPLETED", "PARTIAL", "FAILED", "RUNNING"}
CHECKSUM_PATTERN = re.compile(r"^sha256:[a-f0-9]{64}$")


def all_run_files() -> list[tuple[str, Path]]:
    return [(p.stem, p) for p in EVIDENCE_DIR.glob("*.json")]


class TestEvidenceSchemaFile:
    def test_schema_exists(self):
        assert SCHEMA_PATH.exists()

    def test_schema_defines_checksum_pattern(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        checksum_prop = schema["properties"]["dataset_checksum"]
        assert "pattern" in checksum_prop
        assert "sha256" in checksum_prop["pattern"]


@pytest.mark.parametrize("name,path", all_run_files())
class TestEvidenceRunFiles:
    def test_file_is_valid_json(self, name, path):
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_required_fields_present(self, name, path):
        data = json.loads(path.read_text())
        for field in REQUIRED_FIELDS:
            assert field in data, f"{name}: Missing required field '{field}'"

    def test_checksum_format(self, name, path):
        data = json.loads(path.read_text())
        checksum = data.get("dataset_checksum", "")
        assert CHECKSUM_PATTERN.match(checksum), \
            f"{name}: dataset_checksum '{checksum}' does not match sha256:<64hex>"

    def test_run_status_is_valid(self, name, path):
        data = json.loads(path.read_text())
        assert data["run_status"] in VALID_STATUSES

    def test_records_evaluated_positive(self, name, path):
        data = json.loads(path.read_text())
        assert data["records_evaluated"] >= 1

    def test_aggregate_metrics_present(self, name, path):
        data = json.loads(path.read_text())
        agg = data.get("aggregate_metrics", {})
        assert isinstance(agg, dict)
        assert len(agg) > 0, f"{name}: aggregate_metrics is empty"
