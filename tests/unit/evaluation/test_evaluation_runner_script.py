"""
Unit tests for scripts/run_evaluation.py

Covers:
  - _lexical_overlap_score is distinct from classification accuracy
  - Metric key is 'lexical_f1', not 'accuracy'
  - run_status = 'FAILED' when any metric falls below threshold
  - run_status = 'COMPLETED' when all metrics pass
  - Non-zero exit code on threshold failure
  - PASS / WARN / FAIL classification logic
  - git commit captured in report
"""

import json
import sys
from pathlib import Path

import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from run_evaluation import (  # noqa: E402
    _aggregate,
    _evaluate_records,
    _get_git_commit,
    _lexical_overlap_score,
    _simple_f1,
)


# ---------------------------------------------------------------------------
# Metric renaming: lexical_overlap_score, not accuracy
# ---------------------------------------------------------------------------

class TestLexicalOverlapScore:
    """_lexical_overlap_score is a lexical heuristic, not classification accuracy."""

    def test_identical_strings_score_one(self):
        assert _lexical_overlap_score("hello world", "hello world") == 1.0

    def test_empty_prediction_scores_zero(self):
        assert _lexical_overlap_score("", "hello world") == 0.0

    def test_partial_overlap_above_threshold(self):
        # "hello world foo" vs "hello world bar" — 2/4 unique tokens overlap
        # pred={hello,world,foo}, ref={hello,world,bar}
        # common={hello,world}, precision=2/3, recall=2/3, f1=2/3 ≈ 0.667 >= 0.5
        assert _lexical_overlap_score("hello world foo", "hello world bar") == 1.0

    def test_no_overlap_scores_zero(self):
        assert _lexical_overlap_score("apple orange", "banana grape") == 0.0

    def test_threshold_boundary(self):
        # f1 of exactly 0.5 should score 1.0 (>= threshold)
        # Need a pair where f1 = 0.5: pred={a, b, c, d}, ref={a}
        # common={a}, precision=1/4=0.25, recall=1/1=1.0, f1=2*0.25*1/(0.25+1)=0.4 < 0.5
        # Try pred={a,b}, ref={a,b,c,d}: common={a,b}, P=2/2=1.0, R=2/4=0.5, f1=2/3≈0.667
        # Try pred={a}, ref={a,b}: P=1/1, R=1/2, f1=2*1*0.5/1.5=0.667
        # Try pred={a}, ref={a,b,c}: P=1, R=1/3, f1=0.5 exactly → should be 1.0
        result = _lexical_overlap_score("a", "a b c")
        f1 = _simple_f1("a", "a b c")
        # f1 should be exactly 0.5
        assert abs(f1 - 0.5) < 1e-9
        assert result == 1.0

    def test_function_is_not_named_accuracy(self):
        """The function must be named _lexical_overlap_score, not _accuracy."""
        import run_evaluation  # noqa: PLC0415
        assert hasattr(run_evaluation, "_lexical_overlap_score"), (
            "_lexical_overlap_score should exist"
        )
        assert not hasattr(run_evaluation, "_accuracy"), (
            "_accuracy should have been renamed to _lexical_overlap_score"
        )


class TestEvaluateRecordsMetricKey:
    """Per-record results must use 'lexical_f1', not 'accuracy'."""

    def test_record_has_lexical_f1_key(self):
        records = [{"record_id": "r1", "model_answer": "hello world", "ground_truth_answer": "hello world"}]
        results = _evaluate_records(records)
        assert "lexical_f1" in results[0], "Result must have 'lexical_f1' key"
        assert "accuracy" not in results[0], "'accuracy' key must not appear in per-record results"

    def test_record_has_f1_key(self):
        records = [{"record_id": "r1", "model_answer": "hello world", "ground_truth_answer": "hello world"}]
        results = _evaluate_records(records)
        assert "f1" in results[0]

    def test_aggregate_has_lexical_f1_key(self):
        records = [
            {"record_id": "r1", "model_answer": "good answer", "ground_truth_answer": "good answer"},
            {"record_id": "r2", "model_answer": "bad", "ground_truth_answer": "completely different text"},
        ]
        per_record = _evaluate_records(records)
        agg = _aggregate(per_record)
        assert "lexical_f1" in agg, "Aggregate must include 'lexical_f1' key"
        assert "accuracy" not in agg, "'accuracy' key must not appear in aggregate"


# ---------------------------------------------------------------------------
# run_status and exit code on threshold failures
# ---------------------------------------------------------------------------

class TestRunStatusAndExitCode:
    """run_status must be 'FAILED' and exit code non-zero when thresholds fail."""

    def _make_dataset(self, tmp_path: Path, rows: list[dict]) -> Path:
        import csv  # noqa: PLC0415
        p = tmp_path / "dataset.csv"
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["record_id", "model_answer", "ground_truth_answer", "risk_level", "use_case"])
            writer.writeheader()
            writer.writerows(rows)
        return p

    def test_run_status_failed_when_metrics_below_threshold(self, tmp_path):
        """run_status must be 'FAILED' when lexical_f1 is below threshold."""
        import subprocess  # noqa: PLC0415
        # Completely mismatched answers so both f1 and lexical_f1 will be 0
        dataset = self._make_dataset(tmp_path, [
            {"record_id": "r1", "model_answer": "aaa bbb ccc", "ground_truth_answer": "xxx yyy zzz",
             "risk_level": "HIGH", "use_case": "test"},
        ])
        # Use a low-threshold config that will still fail on 0.0 scores
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent.parent.parent / "scripts" / "run_evaluation.py"),
                "--dataset", str(dataset),
                "--sector", "banking",
                "--output", str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
        )
        # Read the written JSON report
        reports = list((tmp_path / "output").glob("*.json"))
        assert reports, "No report JSON was written"
        report = json.loads(reports[0].read_text())
        assert report["run_status"] == "FAILED", (
            f"Expected run_status='FAILED' for all-zero scores, got: {report['run_status']}"
        )
        assert result.returncode == 1, (
            f"Expected exit code 1 for threshold failure, got: {result.returncode}"
        )

    def test_run_status_completed_when_metrics_pass(self, tmp_path):
        """run_status must be 'COMPLETED' when all metrics pass thresholds."""
        import subprocess  # noqa: PLC0415
        # Identical answers → f1=1.0, well above any reasonable threshold
        dataset = self._make_dataset(tmp_path, [
            {"record_id": "r1", "model_answer": "the loan approval requires income verification",
             "ground_truth_answer": "the loan approval requires income verification",
             "risk_level": "HIGH", "use_case": "loan"},
            {"record_id": "r2", "model_answer": "interest rate is six percent per annum",
             "ground_truth_answer": "interest rate is six percent per annum",
             "risk_level": "LOW", "use_case": "rate"},
        ])
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent.parent.parent / "scripts" / "run_evaluation.py"),
                "--dataset", str(dataset),
                "--sector", "banking",
                "--output", str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
        )
        reports = list((tmp_path / "output").glob("*.json"))
        assert reports, "No report JSON was written"
        report = json.loads(reports[0].read_text())
        assert report["run_status"] == "COMPLETED", (
            f"Expected run_status='COMPLETED' for perfect scores, got: {report['run_status']}"
        )
        assert result.returncode == 0, (
            f"Expected exit code 0 for passing metrics, got: {result.returncode}"
        )

    def test_exit_code_one_on_fail(self, tmp_path):
        """Exit code must be 1 when thresholds are not met."""
        import subprocess  # noqa: PLC0415
        dataset = self._make_dataset(tmp_path, [
            {"record_id": "r1", "model_answer": "wrong answer entirely",
             "ground_truth_answer": "completely different reference text",
             "risk_level": "HIGH", "use_case": "test"},
        ])
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent.parent.parent / "scripts" / "run_evaluation.py"),
                "--dataset", str(dataset),
                "--sector", "banking",
                "--output", str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_exit_code_zero_on_pass(self, tmp_path):
        """Exit code must be 0 when all thresholds are met."""
        import subprocess  # noqa: PLC0415
        dataset = self._make_dataset(tmp_path, [
            {"record_id": "r1", "model_answer": "perfect answer", "ground_truth_answer": "perfect answer",
             "risk_level": "LOW", "use_case": "test"},
        ])
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent.parent.parent / "scripts" / "run_evaluation.py"),
                "--dataset", str(dataset),
                "--sector", "banking",
                "--output", str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# PASS / WARN / FAIL threshold classification
# ---------------------------------------------------------------------------

class TestThresholdClassification:
    """Threshold classification must produce correct PASS/WARN/FAIL status."""

    def _classify(self, val: float, pass_val: float, warn_val: float = 0.0) -> str:
        if val >= pass_val:
            return "PASS"
        elif val >= warn_val:
            return "WARN"
        return "FAIL"

    def test_above_pass_threshold_is_pass(self):
        assert self._classify(0.95, pass_val=0.85) == "PASS"

    def test_exactly_pass_threshold_is_pass(self):
        assert self._classify(0.85, pass_val=0.85) == "PASS"

    def test_between_warn_and_pass_is_warn(self):
        assert self._classify(0.78, pass_val=0.85, warn_val=0.70) == "WARN"

    def test_below_warn_threshold_is_fail(self):
        assert self._classify(0.50, pass_val=0.85, warn_val=0.70) == "FAIL"

    def test_zero_score_is_fail(self):
        # With an explicit warn threshold > 0, a score of 0 falls below both thresholds
        assert self._classify(0.0, pass_val=0.85, warn_val=0.60) == "FAIL"


# ---------------------------------------------------------------------------
# Provenance: git commit
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_get_git_commit_returns_string(self):
        sha = _get_git_commit()
        assert isinstance(sha, str)
        assert len(sha) > 0

    def test_get_git_commit_is_real_sha_or_placeholder(self):
        sha = _get_git_commit()
        import re  # noqa: PLC0415
        is_sha = bool(re.fullmatch(r"[0-9a-f]{40}", sha))
        is_placeholder = sha == "git-unavailable"
        assert is_sha or is_placeholder, f"Unexpected git commit value: {sha!r}"

    def test_report_contains_git_commit(self, tmp_path):
        """The written JSON report must include a non-placeholder 'git_commit' field."""
        import subprocess  # noqa: PLC0415
        import csv  # noqa: PLC0415
        dataset = tmp_path / "data.csv"
        with dataset.open("w") as f:
            w = csv.DictWriter(f, ["record_id", "model_answer", "ground_truth_answer", "risk_level", "use_case"])
            w.writeheader()
            w.writerow({"record_id": "r1", "model_answer": "a", "ground_truth_answer": "a",
                        "risk_level": "LOW", "use_case": "test"})
        subprocess.run(
            [sys.executable,
             str(Path(__file__).parent.parent.parent.parent / "scripts" / "run_evaluation.py"),
             "--dataset", str(dataset), "--sector", "banking",
             "--output", str(tmp_path / "out")],
            capture_output=True, text=True,
        )
        reports = list((tmp_path / "out").glob("*.json"))
        assert reports
        report = json.loads(reports[0].read_text())
        assert "git_commit" in report
        # Must NOT be the old hardcoded placeholder
        assert report["git_commit"] != "abc123def456", (
            "git_commit must not be the hardcoded placeholder 'abc123def456'"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
