# AIEval360 — Test Results

> **Generated:** 2026-11-05  
> **Platform:** AIEval360 v1.0.0  
> **Python:** 3.13.18  
> **pytest:** 9.1.1  

---

## Summary

| Metric | Value |
|--------|-------|
| **Total tests collected** | 626 |
| **Passed** | 626 |
| **Failed** | 0 |
| **Test functions defined** | 562 (expand to 626 via parametrisation) |
| **Execution time** | ~2.6 s |
| **Coverage target** | `aieval360` package |

---

## Test Suite Breakdown

### By Suite

| Suite | Tests | Notes |
|-------|-------|-------|
| `tests/unit/` | ~545 | Unit tests across all stages |
| `tests/contract/` | ~28 | Schema and evidence provenance contracts |
| `tests/integration/` | ~22 | Pipeline integration tests |
| `tests/e2e/` | ~12 | End-to-end subprocess invocation |
| `tests/security/` | ~8 | Adversarial probe execution |
| `tests/performance/` | ~8 | Throughput benchmarks |
| `tests/evidence/` | ~3 | Evidence fixture validation |

### By Stage

| Stage | Scope | Key Test Files |
|-------|-------|----------------|
| **Stage 1** — Data ingestion | Dataset loading, schema validation | `test_dataset_loader.py`, `test_dataset_schema.py` |
| **Stage 2** — Metric evaluation | LLM metrics, RAG, traditional ML | `test_llm_metrics.py`, `test_rag_*.py`, `test_traditional_ml_metrics.py` |
| **Stage 3** — Orchestration | Pipeline orchestration, providers | `test_orchestrator.py`, `test_evaluation_pipeline.py` |
| **Stage 4** — Safety | Safety probes, PII, injection detection | `test_probe_library.py`, `test_safety_evaluator.py` |
| **Scripts** | Runner scripts (corrective engineering) | `test_evaluation_runner_script.py`, `test_safety_runner_script.py` |

---

## Corrective Engineering Tests Added (Phase 1)

The following test files were added as part of the Phase 1 corrective engineering pass, addressing reviewer-identified defects:

### Priority 1 — Safety Runner (`test_safety_runner_script.py`)

18 tests | All **PASS**

| Class | Tests | What It Verifies |
|-------|-------|-----------------|
| `TestProviderDispatch` | 6 | Mock/OpenAI/Ollama routing — no silent mock fallback; unknown provider raises `RuntimeError` |
| `TestGroundednessDetection` | 6 | `_detect_contradiction()` on positive/negative contexts; hallucinated answer fails |
| `TestGroundedProbeEvaluation` | 3 | Full `_evaluate_probe()` on GROUNDED expectation — hallucinated answer → FAILED |
| `TestRunStatusAndExitCode` | 3 | `run_status="FAILED"` on CRITICAL probe failure; exit code non-zero |

### Priority 2 — Evaluation Runner (`test_evaluation_runner_script.py`)

21 tests | All **PASS**

| Class | Tests | What It Verifies |
|-------|-------|-----------------|
| `TestLexicalOverlapScore` | 6 | `_lexical_overlap_score` is NOT named `_accuracy`; correct lexical heuristic |
| `TestEvaluateRecordsMetricKey` | 3 | Per-record and aggregate keys use `lexical_f1`, not `accuracy` |
| `TestRunStatusAndExitCode` | 4 | `run_status="FAILED"` on threshold miss; exit code 1 on FAIL, 0 on PASS |
| `TestThresholdClassification` | 5 | PASS / WARN / FAIL threshold logic boundary conditions |
| `TestProvenance` | 3 | `_get_git_commit()` returns real SHA or `git-unavailable`; never hardcoded `abc123def456` |

---

## Priority 3 — Evidence & Provenance Fixes

Fixed in `scripts/run_evaluation.py` and `scripts/run_safety_probes.py`:

| Defect | Fix Applied |
|--------|-------------|
| Hardcoded `abc123def456` git SHA | `subprocess.run(["git", "rev-parse", "HEAD"])` captures real HEAD SHA |
| Missing provider identity | `--provider` and `--model` CLI args; written to JSON report |
| Missing evaluator version | `evaluator_version: "1.0.0"` in report |
| Dataset checksum | `sha256:` prefix + hex digest; validated in `test_evidence_schema.py` |

---

## Priority 4 — Packaging

| Item | Status |
|------|--------|
| `aieval360.cli:app` entry point | ✅ Created `aieval360/cli.py` with Typer — `evaluate`, `safety`, `version` commands |
| `aieval360 --help` | ✅ Works — shows help with all three subcommands |
| `aieval360 version` | ✅ Outputs `aieval360 1.0.0` |
| Apache 2.0 `LICENSE` file | ✅ Added to repo root |
| `pip install -e .` | ✅ Installs cleanly; `aieval360` CLI available on PATH |

---

## Ruff Lint

```
ruff check scripts/ aieval360/ --select=E,W,F --ignore=E501
Found 21 errors (pre-existing in original codebase)
  - 5 in scripts/generate_report.py: f-string without placeholder (F541) — fixable
  - 16 in aieval360/ package: unused imports (F401) — pre-existing
New files (cli.py, run_evaluation.py, run_safety_probes.py): 0 errors
```

---

## Full pytest Run

```
$ python -m pytest tests/ -q
...
626 passed, 10 warnings in 2.59s
```

Warnings are from `pydantic` about a default insecure secret key — expected in development; set `AIEVAL360_SECURITY_SECRET_KEY` in production.

---

## How to Run

```bash
# Full test suite
python -m pytest tests/ -q

# Unit tests only (fast)
python -m pytest tests/unit/ -q

# Safety + evaluation runner tests (corrective engineering)
python -m pytest tests/unit/safety/test_safety_runner_script.py \
                 tests/unit/evaluation/test_evaluation_runner_script.py -v

# CLI smoke test
aieval360 --help
aieval360 version

# Safety probe run (mock provider)
python scripts/run_safety_probes.py --sector banking --provider mock

# Evaluation run (mock provider)
python scripts/run_evaluation.py \
  --dataset data/datasets/banking/loan_qa_dataset.csv \
  --sector banking
```
