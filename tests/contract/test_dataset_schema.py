"""
Contract tests — validate that all synthetic CSV datasets conform to
the JSON Schema defined in data/schemas/dataset_schema.json.

These tests run on every CI push to catch schema drift early.
"""

import csv
import json
from pathlib import Path

import pytest

SCHEMA_PATH = Path(__file__).parent.parent.parent / "data" / "schemas" / "dataset_schema.json"
DATASETS_DIR = Path(__file__).parent.parent.parent / "data" / "datasets"

REQUIRED_COLUMNS = {
    "record_id", "question", "context",
    "ground_truth_answer", "model_answer",
    "sector", "use_case", "risk_level",
}
VALID_RISK_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
VALID_SECTORS = {"banking", "university", "insurance", "healthcare", "manufacturing", "government"}


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def all_datasets() -> list[tuple[str, Path]]:
    datasets = []
    for csv_path in DATASETS_DIR.rglob("*.csv"):
        datasets.append((csv_path.stem, csv_path))
    return datasets


class TestSchemaFileExists:
    def test_dataset_schema_exists(self):
        assert SCHEMA_PATH.exists(), f"Schema not found: {SCHEMA_PATH}"

    def test_schema_is_valid_json(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        assert "$schema" in schema
        assert "required" in schema

    def test_schema_requires_all_core_fields(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        for field in REQUIRED_COLUMNS:
            assert field in schema["required"], f"Schema missing required field: {field}"


@pytest.mark.parametrize("name,path", all_datasets())
class TestDatasetConformsToSchema:
    def test_csv_loads(self, name, path):
        records = load_csv(path)
        assert len(records) > 0, f"{name}: CSV is empty"

    def test_required_columns_present(self, name, path):
        records = load_csv(path)
        columns = set(records[0].keys())
        missing = REQUIRED_COLUMNS - columns
        assert not missing, f"{name}: Missing columns: {missing}"

    def test_no_empty_required_fields(self, name, path):
        records = load_csv(path)
        for rec in records:
            for field in ["question", "ground_truth_answer", "model_answer"]:
                assert rec.get(field, "").strip(), \
                    f"{name} record {rec.get('record_id')}: Empty field '{field}'"

    def test_valid_risk_levels(self, name, path):
        records = load_csv(path)
        for rec in records:
            level = rec.get("risk_level", "")
            assert level in VALID_RISK_LEVELS, \
                f"{name} record {rec.get('record_id')}: Invalid risk_level '{level}'"

    def test_valid_sectors(self, name, path):
        records = load_csv(path)
        for rec in records:
            sector = rec.get("sector", "")
            assert sector in VALID_SECTORS, \
                f"{name} record {rec.get('record_id')}: Invalid sector '{sector}'"

    def test_no_duplicate_record_ids(self, name, path):
        records = load_csv(path)
        ids = [r["record_id"] for r in records]
        assert len(ids) == len(set(ids)), f"{name}: Duplicate record_ids found"

    def test_record_id_format(self, name, path):
        import re
        records = load_csv(path)
        pattern = re.compile(r"^[A-Z]{2,6}_\d{3}$")
        for rec in records:
            rid = rec.get("record_id", "")
            assert pattern.match(rid), \
                f"{name}: record_id '{rid}' does not match pattern ^[A-Z]{{2,6}}_\\d{{3}}$"
