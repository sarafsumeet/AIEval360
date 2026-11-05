# Screenshots

This directory holds CLI output screenshots demonstrating runtime execution of the AIEval360 platform.

## Planned captures

| File | Description |
|---|---|
| `validate_dataset_banking.png` | `python scripts/validate_dataset.py --dataset data/datasets/banking/loan_qa_dataset.csv --sector banking` |
| `run_safety_probes_banking.png` | `python scripts/run_safety_probes.py --sector banking --provider mock --output evidence/safety_reports/` |
| `generate_report.png` | `python scripts/generate_report.py --run-dir evidence/evaluation_runs --safety-dir evidence/safety_reports --output reports/` |

> These screenshots will be added after the initial repository publish.
> Live evidence artefacts (JSON outputs from actual script runs) are available in `evidence/safety_reports/`.
