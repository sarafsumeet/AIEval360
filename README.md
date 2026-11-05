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
├── tests/                      # 490+ tests across 4 stages
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
# Expected: 490+ tests passing
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
| APRA CPS 234 | Information security for APRA-regulated entities; 7-year audit trail (`retention_days: 2555`) |
| Privacy Act 1988 (APPs) | PII leakage probes test APP 11 compliance |
| AML/CTF Act 2006 | Banking dataset includes AML identification scenarios |
| Insurance Contracts Act 1984 | Insurance dataset tests statutory definitions (flood/storm, duty of disclosure) |
| ASIC RG 271 (IDR) | Insurance dataset tests dispute resolution accuracy |
| NCCP Act 2009 | Banking dataset tests responsible lending narratives |

> This alignment is for **evaluation design purposes only** and does not constitute legal advice.

---

## Evidence Chain

Every evaluation run produces a 4-layer audit trail:

```
Layer 1: Dataset Checksum    sha256:<hash of CSV>
    ↓
Layer 2: Run Provenance      run_id, timestamp, config reference
    ↓
Layer 3: Result Traceability per-record scores, flags, governance status
    ↓
Layer 4: Governance Record   mandatory actions, regulatory alignment, retention schedule
```

Evidence JSON files are stored in `evidence/` and conform to `data/schemas/evidence_run_schema.json`.

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

| Stage | Tests | Coverage |
|-------|-------|----------|
| Stage 1 — Foundation | 95 | Provider adapters, base models, config |
| Stage 2 — Metrics | 142 | All MetricCalculator implementations, registry |
| Stage 3 — RAG | 128 | RAG pipeline, retrieval and generation metrics |
| Stage 4 — Safety | 125 | All probe categories, mandatory gate logic |
| **Total** | **490** | **Core evaluation engine** |

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
