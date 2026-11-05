# Changelog

All notable changes to AIEval360 are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] — 2026-11-05

### Added — Project Structure
- `README.md` — full project overview, capability table, architecture, quick-start guide
- `CHANGELOG.md` — this file
- `SECURITY.md` — vulnerability disclosure policy
- `requirements.txt` — pinned dependency manifest for reproducible installs
- `.gitignore` — excludes `__pycache__`, `*.pyc`, `.DS_Store`, secrets, egg-info, and runtime evidence JSON
- `pyproject.toml` — pytest markers registered (`e2e`, `performance`); coverage config added

### Added — CI/CD Workflows
- `.github/workflows/ci.yml` — Python matrix (3.11 / 3.12 / 3.13); jobs: `test` + `validate-datasets`; triggers on push and PR to main
- `.github/workflows/safety-gate.yml` — per-sector safety gate matrix (banking / insurance / university); `governance-gate` blocking job; nightly cron at 02:00 UTC

### Added — Test Suite (new modules)
- `tests/contract/test_dataset_schema.py` — 20 tests: CSV conformance to JSON Schema for all three sector datasets
- `tests/contract/test_evidence_schema.py` — 12 tests: evidence run JSON conformance to `evidence_run_schema.json`
- `tests/integration/test_evaluation_pipeline.py` — 15 tests: dataset record counts, intentional error records, threshold config alignment
- `tests/evidence/test_evidence_fixtures.py` — 12 tests: golden fixture shape validation for banking run and safety report
- `tests/security/test_adversarial_probes.py` — 21 tests: prompt injection, PII leakage, refusal detection, toxicity, mandatory gate logic
- `tests/performance/test_throughput.py` — 4 tests: 15-record <1s, 1000-record <30s, ≥100 rec/s throughput, CSV memory <10MB
- `tests/e2e/test_full_pipeline.py` — 7 tests: subprocess-level validation, dataset CLI, safety probe CLI, report generation CLI
- `tests/results/pytest_results.txt` — captured output of full 580-test run (2026-11-05)
- `tests/results/pytest_results.xml` — JUnit XML for CI integration
- `tests/results/coverage_report.txt` — line coverage report (83% overall)
- `tests/results/coverage.xml` — machine-readable coverage XML

### Added — Evidence Artefacts
- `tests/evidence/fixtures/golden_banking_run.json` — golden fixture: expected shape of a banking evaluation run with intentional errors
- `tests/evidence/fixtures/golden_safety_report.json` — golden fixture: expected shape of a safety probe report
- `evidence/safety_reports/safety_gate_banking_live_run.json` — live captured output: banking sector gate PASSED (2026-11-05)
- `evidence/safety_reports/safety_gate_insurance_live_run.json` — live captured output: insurance sector gate PASSED (2026-11-05)
- `evidence/safety_reports/safety_gate_university_live_run.json` — live captured output: university sector gate PASSED (2026-11-05)

### Added — Documentation
- `docs/EVALUATION_METHODOLOGY.md` — methodology, metric rationale, threshold calibration, intentional error index, evidence chain, regulatory alignment
- `docs/screenshots/` — placeholder directory (CLI screenshots to be added post-deployment)

### Fixed
- Repository root structure: v1.0 accidentally published only the Python source package (`aieval360/` subfolder), omitting README, tests, config, data, evidence, scripts, and CI workflows
- Committed `__pycache__` and `.DS_Store` artefacts removed via `.gitignore`

### Total test count: 580 passing (490 unit + 90 new) — verified 2026-11-05

---

## [1.0.0] — 2026-10-15

### Added — Core Platform (Stages 1–4)
- `aieval360/core/` — settings, exceptions, logging
- `aieval360/agents/` — base agent abstraction
- `aieval360/evaluation/` — metric calculators: traditional ML, RAG pipeline, agent evaluation
- `aieval360/providers/` — provider adapter pattern: mock, OpenAI, Ollama
- `aieval360/safety/` — safety evaluator, probe library, safety models
- `config/metrics_config.yaml` — sector-specific thresholds (banking, insurance, university, general)
- `data/datasets/` — synthetic datasets: banking (15 records), insurance (16 records), education (15 records)
- `data/schemas/` — JSON Schema contracts for dataset CSV and evidence run JSON
- `evidence/` — evaluation run JSON, safety report JSON, governance record JSON (synthetic samples)
- `scripts/` — `validate_dataset.py`, `run_safety_probes.py`, `generate_report.py`
- `reports/` — sample generated markdown report
- `tests/unit/` — 490 unit tests across all core modules

### Platform capabilities delivered
- Multi-sector evaluation (banking / insurance / university / general)
- 19 metric types across traditional ML, RAG, and agent evaluation dimensions
- Provider adapter pattern with mock, OpenAI, and Ollama support
- Safety probe library: prompt injection, jailbreak, PII leakage, hallucination, toxicity
- Mandatory governance gate with CRITICAL/HIGH/MEDIUM/LOW severity classification
- 7-year audit trail retention (APRA CPS 234 alignment)
- Australian regulatory alignment: APRA CPS 234, NCCP Act 2009, AML/CTF Act 2006, Insurance Contracts Act 1984, Privacy Act 1988, ASIC RG 271
