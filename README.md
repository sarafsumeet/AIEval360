# AIEval360 — Enterprise AI Evaluation & Assurance Platform

[![CI](https://github.com/sarafsumeet/AIEval360/actions/workflows/ci.yml/badge.svg)](https://github.com/sarafsumeet/AIEval360/actions/workflows/ci.yml)
[![Safety Gate](https://github.com/sarafsumeet/AIEval360/actions/workflows/safety-gate.yml/badge.svg)](https://github.com/sarafsumeet/AIEval360/actions/workflows/safety-gate.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

AIEval360 is a **governance-by-design** AI evaluation and assurance platform for regulated industries. It provides a structured, auditable pipeline for assessing AI model quality, safety, and regulatory alignment — with sector-specific thresholds for banking, insurance, and education.

> **Disclaimer:** All datasets and evidence artefacts in this repository are **synthetic data generated for evaluation and demonstration purposes only**. No real customer, patient, or organisational data is used. The platform does not claim production deployment, commercial adoption, or University of Sydney involvement.

---

## Key Capabilities

| Capability | Description |
|-----------|-------------|
| **Stage 1 — Foundation** | Provider adapter pattern, base metric contracts, dataset loader |
| **Stage 2 — Metric Evaluation** | Deterministic (accuracy, F1, ROUGE-L) and heuristic metrics with sector-aware thresholds |
| **Stage 3 — RAG Evaluation** | Context precision, retrieval hit rate, hallucination detection for Retrieval-Augmented Generation |
| **Stage 4 — Safety Probes** | Prompt injection, jailbreak, PII leakage, toxicity, hallucination — mandatory CI/CD gate |
| **Agent Evaluation** | Goal completion, step efficiency, loop detection for autonomous AI agents |
| **Governance Records** | 4-layer evidence chain: dataset checksum → run provenance → result traceability → governance record |

---

## Architecture

```
aieval360/
├── aieval360/                  # Core Python package
│   ├── core/                   # Config, base models, persistence
│   ├── evaluation/             # Metrics, RAG pipeline, runners
│   │   ├── metrics/            # MetricCalculator ABC + registry
│   │   ├── rag/                # RAGEvaluationPipeline
│   │   └── datasets/           # Dataset loader
│   ├── agents/                 # Agent evaluator (goal completion, loop detection)
│   ├── safety/                 # SafetyProbeLibrary, pattern-based detectors
│   └── providers/              # BaseProviderAdapter → mock / openai / ollama
├── tests/                      # 632 tests across 7 suites
│   ├── unit/                   # Unit tests per module
│   ├── evidence/               # Test fixtures (golden outputs)
│   ├── integration/            # Cross-stage integration tests
│   ├── e2e/                    # End-to-end pipeline tests
│   ├── contract/               # JSON Schema contract tests
│   └── security/               # Adversarial probe tests
├── config/                     # YAML configuration
│   ├── settings.yaml           # Platform settings, retention (2555 days / 7 years)
│   ├── metrics_config.yaml     # Sector thresholds (banking ≥0.92, insurance ≥0.90)
│   └── probe_config.yaml       # Safety probe configuration
├── data/                       # Synthetic datasets and reference materials
│   ├── datasets/               # banking / education / insurance CSVs
│   ├── schemas/                # JSON Schema (draft 2020-12) for datasets and evidence
│   └── governance_frameworks/  # Australian regulatory alignment reference
├── evidence/                   # Evaluation artefacts (audit trail)
│   ├── evaluation_runs/        # Metric evaluation JSON records
│   ├── safety_reports/         # Safety probe JSON reports
│   └── governance_records/     # Governance decisions and mandatory actions
├── reports/                    # Human-readable evaluation reports
└── scripts/                    # CLI utilities
    ├── run_evaluation.py       # Metric evaluation runner
    ├── run_safety_probes.py    # Safety gate runner (exit 1 on CRITICAL failure)
    ├── validate_dataset.py     # Pre-evaluation dataset validation
    └── generate_report.py      # Markdown report generator
```

---

## Sector-Specific Thresholds

| Metric | Banking | Insurance | Education | General |
|--------|---------|-----------|-----------|---------|
| Accuracy | ≥ 0.92 | ≥ 0.90 | ≥ 0.85 | ≥ 0.85 |
| F1 Score | ≥ 0.88 | ≥ 0.85 | ≥ 0.80 | ≥ 0.80 |
| Hallucination | ≤ 0.05 | ≤ 0.08 | ≤ 0.10 | ≤ 0.10 |
| Answer Relevance | ≥ 0.85 | ≥ 0.82 | ≥ 0.78 | ≥ 0.75 |

---

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### Install

```bash
git clone https://github.com/sarafsumeet/AIEval360.git
cd AIEval360
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v
# Expected: 632+ tests passing
```

### Run Metric Evaluation (Mock)

```bash
python scripts/run_evaluation.py \
    --dataset data/datasets/banking/loan_qa_dataset.csv \
    --sector banking \
    --output evidence/evaluation_runs/
```

### Run Safety Probes (CI/CD Gate)

```bash
python scripts/run_safety_probes.py --sector banking --provider mock
# Exit code 0 = gate PASSED | Exit code 1 = CRITICAL failure, block deployment
```

### Validate a Dataset

```bash
python scripts/validate_dataset.py \
    --dataset data/datasets/insurance/policy_qa_dataset.csv \
    --sector insurance
```

### Generate Report

```bash
python scripts/generate_report.py \
    --run-dir evidence/evaluation_runs \
    --safety-dir evidence/safety_reports \
    --output reports/generated_report.md
```

---

## Synthetic Datasets

Three sector datasets are included for evaluation demonstration:

| Dataset | Records | Sector | Intentional Errors | Purpose |
|---------|---------|--------|--------------------|---------|
| `banking/loan_qa_dataset.csv` | 15 | Banking | BNK_002, BNK_004, BNK_007 | Tests hallucination and accuracy detection |
| `education/university_qa_dataset.csv` | 15 | Education | UNI_005 | Tests scholarship hallucination (CRITICAL) |
| `insurance/policy_qa_dataset.csv` | 16 | Insurance | INS_015 (flood vs storm) | Tests regulatory definition accuracy |

Intentional errors are documented and used to verify that the evaluation pipeline **correctly flags bad model outputs**.

---

## Australian Regulatory Alignment

AIEval360's governance model is designed with Australian regulated-industry obligations in mind:

| Regulation | Relevance |
|-----------|-----------|
| APRA CPS 234 | Information security for APRA-regulated entities. The platform provides a configurable evidence-retention period (`retention_days` in `config/settings.yaml`, default 2555 days / ~7 years). Whether this configuration satisfies an entity's actual CPS 234 obligations depends on the entity's specific circumstances and requires legal review. |
| Privacy Act 1988 (APPs) | PII leakage probes test APP 11 compliance |
| AML/CTF Act 2006 | Banking dataset includes AML identification scenarios |
| Insurance Contracts Act 1984 | Insurance dataset tests statutory definitions (flood/storm, duty of disclosure) |
| ASIC RG 271 (IDR) | Insurance dataset tests dispute resolution accuracy |
| NCCP Act 2009 | Banking dataset tests responsible lending narratives |

> **Disclaimer:** Regulatory alignment is for **evaluation design purposes only** and does not constitute legal advice. The platform's configurable retention settings are a governance tool, not a determination that any specific retention period satisfies a legal requirement. Entities operating under Australian financial-services regulation should obtain independent legal advice on their applicable obligations.

---

## Evidence Chain

Every evaluation run produces a 4-layer audit trail:

```
Layer 1: Dataset Checksum    sha256:<hex digest of CSV at evaluation time>
    ↓
Layer 2: Run Provenance      run_id, UTC timestamp, git HEAD SHA (or "git-unavailable"),
                             provider, model, evaluator version
    ↓
Layer 3: Result Traceability per-record scores, flags, governance status
    ↓
Layer 4: Governance Record   mandatory actions, regulatory alignment, retention schedule
```

Evidence JSON files are stored in `evidence/` and conform to `data/schemas/evidence_run_schema.json`.

**Actual integrity guarantees:**
- Dataset checksums (SHA-256) detect accidental modification of input data between runs.
- Git provenance records the commit SHA of the codebase that produced the results, enabling code-level auditability when the repository is accessible.
- Each evidence record is a standalone JSON file; schema conformance is validated by the contract test suite.

**Limitations:**
- Evidence files are stored on the local filesystem. They are **not** stored in an append-only or immutable log, cryptographically signed, or written to a tamper-evident store. Files can be modified or deleted by anyone with filesystem access.
- If `git` is unavailable at runtime, the commit SHA is recorded as `"git-unavailable"` rather than failing the run.
- Retention enforcement (archival, deletion after `retention_days`) is not automatically applied by the platform; it requires an external process or operational procedure.

---

## CI/CD Safety Gate

The safety probe runner is designed for CI/CD integration:

```yaml
# .github/workflows/safety-gate.yml
- name: Run safety probes
  run: python scripts/run_safety_probes.py --sector ${{ matrix.sector }} --provider mock
# Returns exit code 1 if any CRITICAL probe fails → blocks deployment
```

---

## Test Coverage

| Suite | Tests (collected) | Scope |
|-------|-------------------|-------|
| `tests/unit/` | 535 | Unit tests across all stages and scripts |
| `tests/contract/` | 38 | Schema and evidence provenance contracts |
| `tests/security/` | 21 | Adversarial probe execution |
| `tests/evidence/` | 12 | Evidence fixture validation |
| `tests/integration/` | 15 | Pipeline integration tests |
| `tests/e2e/` | 7 | End-to-end subprocess invocation |
| `tests/performance/` | 4 | Throughput benchmarks |
| **Total** | **632** | **Core evaluation engine** |

> **Note on test counts:** The total of 632 is the number of tests *collected and run* by pytest, which includes parametrised test variants. The underlying test functions number approximately 490; the remaining tests are generated at runtime through `@pytest.mark.parametrize`. Both figures are correct — 632 is what `pytest -q` reports and what CI runs.
>
> Tests verify that evaluation **logic runs correctly**. They do not validate the AI model being evaluated.

---

## Author

**Sumeet Saraf**  
Responsible AI & Data Governance Leader  
[LinkedIn](https://www.linkedin.com/in/sumeet-saraf/) | [Email](mailto:saraf.sumeet@gmail.com)

> *"I design the Data Supply Chain to ensure AI models receive high quality, 'governed by design' data."*

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
