# Testing Guide

**Purpose:** How to run the AIEval360 test suite, understand its structure, interpret results, and integrate it into CI/CD pipelines.

---

## Quick Start

```bash
# Install in editable mode (from repo root)
pip install -e ".[dev]"

# Run all 490 tests
pytest tests/unit/ -v

# Expected result
# ========================= 490 passed in X.XXs =========================
```

---

## Test Structure

```
tests/
└── unit/
    ├── test_models.py                   # Stage 1 — domain models
    ├── test_provider_interface.py       # Stage 1 — provider adapter ABC
    ├── test_settings.py                 # Stage 1 — settings / env vars
    ├── evaluation/
    │   ├── test_metric_base.py          # Stage 2 — MetricType, MetricInput/Output, ABC
    │   ├── test_traditional_ml_metrics.py  # Stage 2 — 8 DETERMINISTIC calculators
    │   ├── test_llm_metrics.py          # Stage 2 — 6 HEURISTIC/LLM_JUDGED calculators
    │   ├── test_dataset_loader.py       # Stage 2 — CSV/JSON load, checksums
    │   ├── test_orchestrator.py         # Stage 2 — full orchestration run
    │   ├── test_rag_retrieval_metrics.py  # Stage 3 — 7 retrieval functions
    │   ├── test_rag_generation_metrics.py # Stage 3 — 4 generation functions
    │   └── test_rag_pipeline.py          # Stage 3 — async pipeline + aggregation
    ├── safety/
    │   ├── test_safety_models.py        # Stage 4 — enums, models, mandatory_gate_passed
    │   ├── test_probe_library.py        # Stage 4 — 40 probes, filters, lookup
    │   └── test_safety_evaluator.py     # Stage 4 — detection functions + evaluator
    └── agents/
        └── test_agent_evaluator.py      # Stage 4 — 5 metrics, loop detection, scoring
```

---

## Running Tests by Stage

```bash
# Stage 1 — Foundation
pytest tests/unit/test_models.py tests/unit/test_provider_interface.py tests/unit/test_settings.py -v

# Stage 2 — Evaluation Engine
pytest tests/unit/evaluation/ -v --ignore=tests/unit/evaluation/test_rag_*.py

# Stage 3 — RAG Pipeline
pytest tests/unit/evaluation/test_rag_*.py -v

# Stage 4 — Safety
pytest tests/unit/safety/ -v

# Stage 4 — Agent Evaluation
pytest tests/unit/agents/ -v
```

---

## Running Tests by Type

```bash
# All async tests (uses asyncio event loop — pytest-asyncio required)
pytest tests/unit/ -v -k "async"

# Only deterministic metric tests (fastest, no I/O)
pytest tests/unit/evaluation/test_traditional_ml_metrics.py -v

# Safety gate tests only
pytest tests/unit/safety/test_safety_models.py -v -k "gate"

# Loop detection tests only
pytest tests/unit/agents/test_agent_evaluator.py -v -k "loop"
```

---

## Test Categories and What They Verify

### Stage 1 Tests

| File | Key assertions |
|------|---------------|
| `test_models.py` | AuditedModel auto-generates UUID and timestamps; EvaluationRun status transitions; GovernanceRecord links run correctly |
| `test_provider_interface.py` | MockAdapter returns configured responses in order; cycles on exhaustion; `health_check()` returns True |
| `test_settings.py` | Default DATABASE_URL is SQLite; settings override via env var; API keys are not required at import |

### Stage 2 Tests

| File | Key assertions |
|------|---------------|
| `test_metric_base.py` | MetricType has exactly 5 values; MetricCalculator is abstract; MetricInput/Output are dataclasses |
| `test_traditional_ml_metrics.py` | F1(exact match) = 1.0; F1(no overlap) = 0.0; BLEU returns float in [0,1]; CER(empty ref) handles gracefully |
| `test_llm_metrics.py` | LLM_JUDGED returns 0.5 when judge_fn=None; judge_fn result flows through; HEURISTIC metrics return float in [0,1] |
| `test_dataset_loader.py` | Checksum is deterministic; different column order = same checksum; different data = different checksum |
| `test_orchestrator.py` | Run result contains all requested metrics; dataset_checksum is set; async concurrency is bounded |

### Stage 3 Tests

| File | Key assertions |
|------|---------------|
| `test_rag_retrieval_metrics.py` | context_precision([same context]) = 1.0; MRR first position = 1.0; NDCG handles empty list gracefully |
| `test_rag_generation_metrics.py` | answer_completeness with judge_fn=None returns 0.5; hallucination_score is in [0,1]; faithfulness scores high for grounded answer |
| `test_rag_pipeline.py` | Pipeline runs all examples; mean_scores are averaged correctly; run_label propagates to result |

### Stage 4 Tests

| File | Key assertions |
|------|---------------|
| `test_safety_models.py` | mandatory_gate_passed=False when any CRITICAL probe fails; True when only HIGH/MEDIUM fail; ProbeOutcome enum has 4 values |
| `test_probe_library.py` | Exactly 40 probes; each category has exactly 8; probe IDs are unique; get_probe_by_id("nonexistent") returns None |
| `test_safety_evaluator.py` | refusal_detected("I cannot help") = True; pii_detected("john@ex.com") = True; toxicity_detected raises correctly; evaluator uses provider adapter |
| `test_agent_evaluator.py` | Loop detected when same (tool, input) pair appears twice; overall_score = 0.5*gc + 0.3*eff + 0.2*(1-err); goal_status=ACHIEVED when score >= 0.7 |

---

## Test Configuration

### `pyproject.toml` test settings

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

`asyncio_mode = "auto"` means all `async def test_*` functions are automatically run with an event loop — no `@pytest.mark.asyncio` decorator needed on individual tests.

### Dependencies

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.0",
]
```

---

## Test Output Interpretation

### Passing Run

```
tests/unit/evaluation/test_traditional_ml_metrics.py::TestF1Calculator::test_exact_match PASSED
tests/unit/evaluation/test_traditional_ml_metrics.py::TestF1Calculator::test_no_overlap PASSED
...
========================= 490 passed in 1.47s =========================
```

### Coverage Report

```bash
# Generate coverage report
pytest tests/unit/ --cov=aieval360 --cov-report=term-missing

# Generate HTML report (opens in browser)
pytest tests/unit/ --cov=aieval360 --cov-report=html
open htmlcov/index.html
```

### Failing Test

```
FAILED tests/unit/safety/test_safety_models.py::TestSafetyEvaluationReport::test_gate_critical_failure
AssertionError: mandatory_gate_passed should be False when CRITICAL probe fails
```

This indicates either:
1. `is_critical_failure` logic changed on `ProbeResult`
2. Severity threshold was modified
3. Test expectation is wrong

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: AIEval360 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
      
      - name: Run tests
        run: pytest tests/unit/ -v --tb=short
      
      - name: Generate coverage
        run: pytest tests/unit/ --cov=aieval360 --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

### Mandatory Safety Gate in CI

```python
# Example CI script: fail the pipeline if safety gate fails
import asyncio
from aieval360.safety.evaluator import SafetyEvaluator
from aieval360.providers.mock.adapter import MockProviderAdapter

async def run_safety_gate():
    # In production: replace MockAdapter with real provider
    provider = MockProviderAdapter(responses=["I cannot help with that."])
    evaluator = SafetyEvaluator(provider=provider, executed_by="ci_pipeline")
    report = await evaluator.run_all_probes()
    
    if not report.mandatory_gate_passed:
        print("SAFETY GATE FAILED — deployment blocked")
        exit(1)
    else:
        print("Safety gate passed")

asyncio.run(run_safety_gate())
```

---

## Test Governance Notes

### What the Tests Demonstrate

| Claim | Evidence |
|-------|---------|
| All 14 metric calculators compute correctly | 490 passing tests including edge cases |
| SHA-256 checksums are column-order-independent | `test_dataset_loader.py` — same checksum for shuffled columns |
| Mandatory gate blocks CRITICAL failures | `test_safety_models.py` — `mandatory_gate_passed=False` verified |
| Loop detection catches repeated (tool, input) pairs | `test_agent_evaluator.py` — `loop_detected=True` verified |
| LLM_JUDGED metrics gracefully fall back | `test_llm_metrics.py` — 0.5 returned when `judge_fn=None` |

### What the Tests Do Not Demonstrate

- **Production performance at scale** — Tests use in-memory mocks and small datasets
- **Real LLM provider accuracy** — LLM_JUDGED metrics use mock `judge_fn` returning fixed values
- **Adversarial robustness in production** — 40 probes represent a sample, not exhaustive coverage
- **Network resilience** — Provider adapters are mocked; real API failures are not tested

These limitations are by design. The test suite demonstrates functional correctness of the platform's logic, not deployment-environment validation.
