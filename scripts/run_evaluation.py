#!/usr/bin/env python3
"""
AIEval360 — Run Metric Evaluation Pipeline

Loads a CSV dataset and runs Stage 2 metric evaluation, writing JSON results
to the evidence/evaluation_runs/ directory.

Usage:
    python scripts/run_evaluation.py \
        --dataset data/datasets/banking/loan_qa_dataset.csv \
        --sector banking \
        --output evidence/evaluation_runs/

Environment Variables:
    DATABASE_URL   — async SQLAlchemy connection string (optional; skip DB if unset)
    LOG_LEVEL      — DEBUG, INFO, WARNING (default: INFO)

Metric naming:
    lexical_f1     — token-overlap F1 (Jaccard-based set similarity); a measure of
                     lexical overlap between prediction and reference, NOT semantic
                     correctness or classification accuracy. Use alongside a semantic
                     metric (e.g. BERTScore) for a fuller picture.
    f1             — same token-overlap F1, also reported for backward compatibility.
"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _sha256_dataset(csv_path: Path) -> str:
    """Compute a stable SHA-256 checksum for the dataset file."""
    h = hashlib.sha256()
    h.update(csv_path.read_bytes())
    return f"sha256:{h.hexdigest()}"


def _load_csv(csv_path: Path) -> list[dict]:
    """Load CSV into a list of dicts without depending on pandas at import time."""
    import csv  # noqa: PLC0415

    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _simple_f1(pred: str, ref: str) -> float:
    """Token-overlap F1 (set-based Jaccard precision/recall).

    Computes precision and recall over the *set* of unique tokens, then
    takes the harmonic mean.  Token frequency is intentionally ignored to
    keep the metric dependency-free and deterministic.

    Note: this is a *lexical overlap* heuristic.  It does not measure
    semantic correctness.  A score of 1.0 means every unique token in the
    reference appears in the prediction, and vice versa.
    """
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lexical_overlap_score(pred: str, ref: str, threshold: float = 0.5) -> float:
    """Binary lexical-overlap score: 1.0 if token-overlap F1 >= threshold, else 0.0.

    Previously labelled '_accuracy()' — renamed to reflect that this is a
    lexical heuristic, not classification accuracy or semantic correctness.
    The threshold default (0.5) is a configurable soft boundary, not a
    precision/recall operating point.
    """
    return 1.0 if _simple_f1(pred, ref) >= threshold else 0.0


def _evaluate_records(records: list[dict]) -> list[dict]:
    results = []
    for rec in records:
        pred = rec.get("model_answer", "")
        ref = rec.get("ground_truth_answer", "")
        f1 = _simple_f1(pred, ref)
        lexical_f1_score = _lexical_overlap_score(pred, ref)
        results.append(
            {
                "record_id": rec.get("record_id"),
                # lexical_f1: binary score indicating whether token-overlap F1 >= threshold
                "lexical_f1": round(lexical_f1_score, 4),
                # f1: continuous token-overlap F1 for per-record comparison
                "f1": round(f1, 4),
                "risk_level": rec.get("risk_level"),
                "use_case": rec.get("use_case"),
                "status": "PASS" if f1 >= 0.5 else "FAIL",
            }
        )
    return results


def _aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    avg_lexical_f1 = sum(r["lexical_f1"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n
    return {
        # lexical_f1: mean of per-record binary lexical overlap scores
        "lexical_f1": round(avg_lexical_f1, 4),
        # f1: mean continuous token-overlap F1
        "f1": round(avg_f1, 4),
        "total": n,
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "failed": sum(1 for r in results if r["status"] == "FAIL"),
    }


def _load_thresholds(sector: str, config_path: Path) -> dict:
    """Load sector-aware thresholds from metrics_config.yaml if available."""
    try:
        import yaml  # type: ignore  # noqa: PLC0415

        raw = yaml.safe_load(config_path.read_text())
        base = raw.get("thresholds", {})
        overrides = raw.get("sector_overrides", {}).get(sector, {})
        merged = {**base}
        for metric, vals in overrides.items():
            merged[metric] = vals
        return merged
    except Exception:
        # Return sensible defaults if yaml or file unavailable
        return {"lexical_f1": {"pass": 0.85}, "f1": {"pass": 0.80}}


def _get_git_commit() -> str:
    """Return the current HEAD git commit SHA, or a descriptive placeholder if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = result.stdout.strip()
        if result.returncode == 0 and sha:
            return sha
        return "git-unavailable"
    except Exception:
        return "git-unavailable"


async def main() -> int:
    parser = argparse.ArgumentParser(description="AIEval360 metric evaluation runner")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    parser.add_argument("--sector", required=True, help="Sector (banking/university/insurance)")
    parser.add_argument("--output", default="evidence/evaluation_runs", help="Output directory")
    parser.add_argument("--config", default="config/metrics_config.yaml", help="Metrics config path")
    parser.add_argument("--provider", default="mock", help="Provider identifier (for provenance)")
    parser.add_argument("--model", default="mock", help="Model identifier (for provenance)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    config_path = Path(args.config)

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"eval-run-{args.sector}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    checksum = _sha256_dataset(dataset_path)
    records = _load_csv(dataset_path)

    print(f"[{run_id}] Loaded {len(records)} records from {dataset_path}")
    print(f"[{run_id}] Dataset checksum: {checksum}")

    per_record = _evaluate_records(records)
    agg = _aggregate(per_record)
    thresholds = _load_thresholds(args.sector, config_path)

    # Classify aggregate metrics against thresholds
    agg_classified = {}
    any_fail = False
    for metric, val in [("lexical_f1", agg.get("lexical_f1", 0)), ("f1", agg.get("f1", 0))]:
        t = thresholds.get(metric, thresholds.get("f1", {}))  # fall back to f1 key if renamed
        pass_val = t.get("pass", 0.0)
        warn_val = t.get("warn", 0.0)
        if val >= pass_val:
            status = "PASS"
        elif val >= warn_val:
            status = "WARN"
        else:
            status = "FAIL"
            any_fail = True
        agg_classified[metric] = {
            "value": val,
            "threshold_pass": pass_val,
            "status": status,
        }

    # run_status reflects whether any mandatory threshold was missed
    run_status = "FAILED" if any_fail else "COMPLETED"

    report = {
        "run_id": run_id,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "dataset_checksum": checksum,
        "sector": args.sector,
        "provider": args.provider,
        "model": args.model,
        "git_commit": _get_git_commit(),
        "evaluator_version": "1.0.0",  # from pyproject.toml
        "records_evaluated": len(records),
        "run_status": run_status,
        "aggregate_metrics": agg_classified,
        "per_record_results": per_record,
    }

    out_file = output_dir / f"{run_id}.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(f"[{run_id}] Results written to: {out_file}")

    # Print summary
    for metric, vals in agg_classified.items():
        icon = "✅" if vals["status"] == "PASS" else ("⚠️" if vals["status"] == "WARN" else "❌")
        print(f"  {icon} {metric}: {vals['value']:.3f} (threshold: {vals['threshold_pass']:.2f}) [{vals['status']}]")

    # Exit code 0 = all thresholds met | Exit code 1 = one or more thresholds failed
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
