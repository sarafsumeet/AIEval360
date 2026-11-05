# AIEval360 — Enterprise AI Evaluation & Assurance Platform

**Version:** 0.4.0 (Stages 1–4)  
**Python:** 3.13  
**Author:** Sumeet Saraf — Responsible AI & Data Governance Leader  

---

## Overview

AIEval360 is a governance-first AI evaluation platform designed for enterprise use. It provides structured, auditable, and reproducible tooling to assess AI system quality, safety, and reliability across the full model lifecycle.

The platform is built around four core principles:

| Principle | Implementation |
|-----------|---------------|
| **Auditability** | Every evaluation result carries provenance metadata, run labels, and executor identity |
| **Reproducibility** | SHA-256 dataset checksums; deterministic metric calculations; pure-function evaluation patterns |
| **Governance-by-design** | Mandatory safety gates; critical-failure blocking; structured evidence artefacts |
| **Separation of concerns** | Evaluation ≠ Governance — they are distinct capabilities connected through assurance |

---

## Project Structure

```
aieval360/
├── aieval360/                  # Source package
│   ├── core/                   # Stage 1 — Foundation
│   │   ├── config/             #   Application settings
│   │   ├── models/             #   Domain models (evaluation, governance, organisation)
│   │   └── persistence/        #   Database layer (async SQLAlchemy 2.0)
│   ├── providers/              # Stage 1 — AI Provider Adapters
│   │   ├── base/               #   BaseProviderAdapter ABC
│   │   ├── mock/               #   Mock adapter for testing
│   │   ├── openai/             #   OpenAI adapter
│   │   └── ollama/             #   Ollama adapter
│   ├── evaluation/             # Stage 2 & 3 — Evaluation Engine
│   │   ├── metrics/            #   Stage 2: metric calculators & registry
│   │   ├── datasets/           #   Stage 2: dataset loading with lineage
│   │   ├── runners/            #   Stage 2: evaluation orchestrator
│   │   └── rag/                #   Stage 3: RAG-specific evaluation pipeline
│   ├── safety/                 # Stage 4 — Safety Evaluation
│   │   ├── models.py           #   Enums, probe, result, report models
│   │   ├── probe_library.py    #   40 adversarial probes across 5 categories
│   │   └── evaluator.py        #   Pattern-based detection + async evaluator
│   └── agents/                 # Stage 4 — Agent Trace Evaluation
│       ├── models.py           #   AgentStep, AgentTrace, AgentEvaluationResult
│       └── evaluator.py        #   5-metric evaluator with loop detection
├── tests/
│   ├── unit/                   # 490 unit tests (all passing)
│   │   ├── test_models.py      #   Stage 1
│   │   ├── test_provider_interface.py
│   │   ├── test_settings.py
│   │   ├── evaluation/         #   Stages 2 & 3 (8 test files)
│   │   ├── safety/             #   Stage 4 safety (3 test files)
│   │   └── agents/             #   Stage 4 agents (1 test file)
│   └── evidence/               #   Governance evidence artefacts (Stage 6)
└── docs/                       # This folder
    ├── README.md               #   This file
    ├── architecture.md         #   System architecture & design decisions
    ├── stage1-foundation.md    #   Stage 1 reference
    ├── stage2-evaluation.md    #   Stage 2 reference
    ├── stage3-rag.md           #   Stage 3 reference
    ├── stage4-safety-agents.md #   Stage 4 reference
    ├── governance-model.md     #   Governance framework & audit trail design
    └── testing-guide.md        #   Running tests & interpreting results
```

---

## Quick Start

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run all tests
pytest tests/unit/ -v

# Run a specific stage's tests
pytest tests/unit/safety/ -v
pytest tests/unit/agents/ -v
pytest tests/unit/evaluation/ -v
```

Expected output: **490 passed** in < 2 seconds.

---

## Stages at a Glance

| Stage | Focus | Source Files | Tests |
|-------|-------|-------------|-------|
| 1 | Foundation — models, providers, persistence | 11 | 3 files |
| 2 | Evaluation engine — metrics, registry, orchestrator | 6 | 5 files, 120 tests |
| 3 | RAG evaluation pipeline | 5 | 3 files, 120 tests |
| 4 | Safety probing + agent trace evaluation | 6 | 4 files, ~130 tests |
| **Total** | | **28** | **490 tests** |

---

## Governance Note

AIEval360 is a **demonstration and assurance platform** built for local evaluation, portfolio evidence, and governance design purposes. Test results demonstrate the platform's functional correctness under controlled conditions; production scalability must be validated separately in a target deployment environment.
