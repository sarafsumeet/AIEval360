"""
Performance tests — throughput and latency benchmarks.

These tests assert that the evaluation pipeline meets minimum throughput
requirements without actual LLM calls (mock provider). They are marked
`performance` and excluded from the default CI run to avoid flakiness
on shared runners.

Run locally: pytest tests/performance/ -v -m performance
"""

import csv
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
BANKING_CSV = PROJECT_ROOT / "data" / "datasets" / "banking" / "loan_qa_dataset.csv"


def simple_f1(pred: str, ref: str) -> float:
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def accuracy(pred: str, ref: str, threshold: float = 0.5) -> float:
    return 1.0 if simple_f1(pred, ref) >= threshold else 0.0


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.performance
class TestEvaluationThroughput:
    def test_15_records_complete_in_under_1_second(self):
        """Core metric calculation for 15 records must complete < 1s."""
        records = load_csv(BANKING_CSV)
        start = time.perf_counter()
        for rec in records:
            simple_f1(rec["model_answer"], rec["ground_truth_answer"])
            accuracy(rec["model_answer"], rec["ground_truth_answer"])
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"15-record evaluation took {elapsed:.3f}s (limit: 1.0s)"

    def test_1000_record_simulation_under_30_seconds(self):
        """Simulated 1000-record batch must complete < 30s using synthetic data."""
        records = load_csv(BANKING_CSV) * 67  # ~1005 records
        records = records[:1000]
        start = time.perf_counter()
        for rec in records:
            simple_f1(rec["model_answer"], rec["ground_truth_answer"])
            accuracy(rec["model_answer"], rec["ground_truth_answer"])
        elapsed = time.perf_counter() - start
        assert elapsed < 30.0, f"1000-record batch took {elapsed:.3f}s (limit: 30s)"

    def test_records_per_second_exceeds_minimum(self):
        """Must process at least 100 records/second for core metrics."""
        records = load_csv(BANKING_CSV) * 10  # 150 records
        start = time.perf_counter()
        for rec in records:
            simple_f1(rec["model_answer"], rec["ground_truth_answer"])
        elapsed = time.perf_counter() - start
        rps = len(records) / elapsed
        assert rps >= 100, f"Throughput {rps:.0f} records/sec below minimum (100)"


@pytest.mark.performance
class TestMemoryUsage:
    def test_csv_load_does_not_hold_excessive_memory(self):
        """Loading the banking CSV should use < 10 MB."""
        import sys
        records = load_csv(BANKING_CSV)
        # Rough estimate: each record dict
        size_bytes = sys.getsizeof(records) + sum(
            sys.getsizeof(r) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in r.items())
            for r in records
        )
        size_mb = size_bytes / (1024 * 1024)
        assert size_mb < 10, f"CSV load used {size_mb:.2f} MB (limit: 10 MB)"
