"""
End-to-end pipeline tests.

These tests exercise the full flow: dataset validation → metric evaluation
→ safety probe → governance record. They use mock providers so no API
keys are required. They are slower than unit tests and are tagged e2e.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_script(script: str, args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script)] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )


@pytest.mark.e2e
class TestDatasetValidationScript:
    def test_validate_banking_dataset(self):
        result = run_script("validate_dataset.py", [
            "--dataset", "data/datasets/banking/loan_qa_dataset.csv",
            "--sector", "banking",
        ])
        assert result.returncode == 0, f"Validation failed:\n{result.stdout}\n{result.stderr}"

    def test_validate_university_dataset(self):
        result = run_script("validate_dataset.py", [
            "--dataset", "data/datasets/education/university_qa_dataset.csv",
            "--sector", "university",
        ])
        assert result.returncode == 0, f"Validation failed:\n{result.stdout}\n{result.stderr}"

    def test_validate_insurance_dataset(self):
        result = run_script("validate_dataset.py", [
            "--dataset", "data/datasets/insurance/policy_qa_dataset.csv",
            "--sector", "insurance",
        ])
        assert result.returncode == 0, f"Validation failed:\n{result.stdout}\n{result.stderr}"


@pytest.mark.e2e
class TestSafetyProbeScript:
    def test_banking_safety_gate_passes(self, tmp_path):
        """Banking sector should pass the mandatory gate (no CRITICAL failures in mock)."""
        result = run_script("run_safety_probes.py", [
            "--sector", "banking",
            "--provider", "mock",
            "--output", str(tmp_path),
        ])
        # Banking mock: gate PASSES
        assert result.returncode == 0, \
            f"Banking gate unexpectedly FAILED:\n{result.stdout}\n{result.stderr}"

    def test_safety_report_written(self, tmp_path):
        run_script("run_safety_probes.py", [
            "--sector", "banking",
            "--provider", "mock",
            "--output", str(tmp_path),
        ])
        reports = list(tmp_path.glob("*.json"))
        assert len(reports) >= 1, "Safety probe script should write a JSON report"

    def test_safety_report_valid_json(self, tmp_path):
        run_script("run_safety_probes.py", [
            "--sector", "banking",
            "--provider", "mock",
            "--output", str(tmp_path),
        ])
        for report_path in tmp_path.glob("*.json"):
            data = json.loads(report_path.read_text())
            assert "mandatory_gate_passed" in data
            assert "total_probes" in data


@pytest.mark.e2e
class TestReportGenerationScript:
    def test_report_generates_from_evidence(self, tmp_path):
        result = run_script("generate_report.py", [
            "--run-dir", "evidence/evaluation_runs",
            "--safety-dir", "evidence/safety_reports",
            "--output", str(tmp_path / "test_report.md"),
            "--title", "E2E Test Report",
        ])
        assert result.returncode == 0, f"Report generation failed:\n{result.stderr}"
        report = tmp_path / "test_report.md"
        assert report.exists()
        content = report.read_text()
        assert "# E2E Test Report" in content
        assert "Metric Evaluation Results" in content
        assert "Safety Evaluation Results" in content
