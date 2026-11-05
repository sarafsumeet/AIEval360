#!/usr/bin/env python3
"""
AIEval360 — Dataset Validator

Validates a CSV dataset against the AIEval360 schema requirements before
running evaluation. Checks required columns, data types, and risk level values.

Usage:
    python scripts/validate_dataset.py --dataset data/datasets/banking/loan_qa_dataset.csv
    python scripts/validate_dataset.py --dataset data/datasets/education/university_qa_dataset.csv --sector university
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

REQUIRED_COLUMNS = {
    "record_id",
    "question",
    "context",
    "ground_truth_answer",
    "model_answer",
    "sector",
    "use_case",
    "risk_level",
}

VALID_RISK_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

VALID_SECTORS = {"banking", "university", "insurance", "healthcare", "manufacturing"}


def validate(dataset_path: Path, expected_sector: str | None) -> list[str]:
    errors: list[str] = []

    if not dataset_path.exists():
        return [f"File not found: {dataset_path}"]

    with dataset_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        actual_columns = set(reader.fieldnames or [])

        missing = REQUIRED_COLUMNS - actual_columns
        if missing:
            errors.append(f"Missing required columns: {sorted(missing)}")

        extra = actual_columns - REQUIRED_COLUMNS
        if extra:
            # Extra columns are allowed; warn only
            print(f"  INFO: Extra columns (OK): {sorted(extra)}")

        rows = list(reader)

    if not rows:
        errors.append("Dataset is empty — no records found.")
        return errors

    record_ids: list[str] = []
    for i, row in enumerate(rows, start=2):  # row 1 = header
        rid = row.get("record_id", "").strip()
        if not rid:
            errors.append(f"Row {i}: record_id is empty.")
        else:
            record_ids.append(rid)

        if not row.get("question", "").strip():
            errors.append(f"Row {i} ({rid}): question is empty.")
        if not row.get("ground_truth_answer", "").strip():
            errors.append(f"Row {i} ({rid}): ground_truth_answer is empty.")
        if not row.get("model_answer", "").strip():
            errors.append(f"Row {i} ({rid}): model_answer is empty.")

        risk = row.get("risk_level", "").strip()
        if risk not in VALID_RISK_LEVELS:
            errors.append(f"Row {i} ({rid}): invalid risk_level '{risk}'. Must be one of {VALID_RISK_LEVELS}.")

        sector = row.get("sector", "").strip()
        if sector not in VALID_SECTORS:
            errors.append(f"Row {i} ({rid}): unknown sector '{sector}'.")
        if expected_sector and sector != expected_sector:
            errors.append(f"Row {i} ({rid}): sector '{sector}' does not match expected '{expected_sector}'.")

    # Check for duplicate record IDs
    seen: set[str] = set()
    for rid in record_ids:
        if rid in seen:
            errors.append(f"Duplicate record_id: '{rid}'.")
        seen.add(rid)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="AIEval360 dataset validator")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    parser.add_argument("--sector", default=None, help="Expected sector value (optional)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    print(f"\nValidating: {dataset_path}")

    # Compute checksum
    if dataset_path.exists():
        checksum = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        print(f"  SHA-256: {checksum[:16]}...{checksum[-8:]}")

    errors = validate(dataset_path, args.sector)

    if errors:
        print(f"\n❌ Validation FAILED — {len(errors)} error(s):")
        for err in errors:
            print(f"  • {err}")
        return 1

    print(f"\n✅ Validation PASSED — {dataset_path.name} is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
