# AIEval360 — Scripts

Utility scripts for running evaluations, safety probes, dataset validation, and report generation.

All scripts are runnable from the project root with:

```bash
python scripts/<script_name>.py [options]
```

---

## Scripts

### `run_evaluation.py`

Runs Stage 2 metric evaluation on a CSV dataset.

```bash
python scripts/run_evaluation.py \
    --dataset data/datasets/banking/loan_qa_dataset.csv \
    --sector banking \
    --output evidence/evaluation_runs/
```

**Options:**
- `--dataset` — path to CSV file (required)
- `--sector` — banking / university / insurance (required)
- `--output` — output directory for JSON evidence (default: `evidence/evaluation_runs/`)
- `--config` — metrics config YAML (default: `config/metrics_config.yaml`)

**Output:** JSON evidence file in `evidence/evaluation_runs/`

---

### `run_safety_probes.py`

Runs Stage 4 safety probe evaluation against a model.

```bash
# Against mock provider (no API key needed)
python scripts/run_safety_probes.py --sector banking --provider mock

# Against OpenAI (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python scripts/run_safety_probes.py --sector insurance --provider openai
```

**Options:**
- `--sector` — banking / university / insurance (required)
- `--provider` — mock / openai / ollama (default: mock)
- `--output` — output directory (default: `evidence/safety_reports/`)

**Returns:** Exit code 0 if mandatory gate PASSES, 1 if gate FAILS (suitable for CI/CD).

---

### `validate_dataset.py`

Validates a CSV dataset against AIEval360 schema requirements before evaluation.

```bash
python scripts/validate_dataset.py \
    --dataset data/datasets/banking/loan_qa_dataset.csv \
    --sector banking
```

**Checks:**
- Required columns present
- No empty `question`, `ground_truth_answer`, `model_answer`
- `risk_level` is one of: CRITICAL, HIGH, MEDIUM, LOW
- `sector` matches expected value
- No duplicate `record_id` values

---

### `generate_report.py`

Generates a consolidated Markdown report from evidence JSON files.

```bash
python scripts/generate_report.py \
    --run-dir evidence/evaluation_runs \
    --safety-dir evidence/safety_reports \
    --output reports/generated_report.md \
    --title "November 2026 Evaluation Report"
```

---

## CI/CD Usage

The safety probe runner exits with code 1 on mandatory gate failure, making it suitable for CI gates:

```yaml
# .github/workflows/safety-gate.yml example
- name: Run safety probes
  run: python scripts/run_safety_probes.py --sector ${{ matrix.sector }} --provider mock
```

See `docs/testing-guide.md` for a full CI/CD example.

---

## Notes

- Scripts use stdlib where possible to minimise dependencies
- Full dependency set available via `pip install -e ".[dev]"` (see `pyproject.toml`)
- Mock provider returns deterministic responses — suitable for CI without API keys
- All scripts write JSON evidence to `evidence/` for audit trail purposes
