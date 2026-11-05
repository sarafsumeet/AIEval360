# Stage 3 — RAG Evaluation Pipeline

**Files:** 5 source · 3 test files · 120 tests  
**Purpose:** Retrieval-Augmented Generation (RAG) evaluation — retrieval quality, generation quality, async pipeline

---

## What Was Built

Stage 3 extends the evaluation engine with RAG-specific tooling. RAG systems have two distinct failure modes — poor retrieval (wrong documents retrieved) and poor generation (wrong answer despite good documents) — requiring separate metric families. The pipeline composes them into a unified, async evaluation run.

---

## Module Reference

### `aieval360/evaluation/rag/models.py`

RAG-specific input/output models.

#### `RAGInput`

```python
@dataclass
class RAGInput:
    query: str                          # The user's question
    retrieved_contexts: list[str]       # Documents returned by the retriever
    generated_answer: str               # The LLM's answer given those contexts
    ground_truth_answer: str | None = None  # Optional; needed by reference-based metrics
    ground_truth_contexts: list[str] = field(default_factory=list)  # Optional
    metadata: dict = field(default_factory=dict)
```

#### `RAGMetricOutput`

```python
@dataclass
class RAGMetricOutput:
    metric_name: str
    value: float                        # Score in [0, 1]
    details: dict = field(default_factory=dict)
    reasoning: str = ""
```

---

### `aieval360/evaluation/rag/retrieval_metrics.py`

**7 pure-function retrieval metrics.** Each is a standalone function (not a class) that takes a `RAGInput` and returns a `RAGMetricOutput`.

| Function | What it measures |
|----------|-----------------|
| `context_precision(input)` | Fraction of retrieved contexts that are relevant to the query |
| `context_recall(input)` | Fraction of ground-truth contexts that were retrieved |
| `context_relevance(input)` | Token overlap between query terms and retrieved contexts |
| `context_coverage(input)` | Breadth: how many distinct query concepts are covered |
| `retrieval_hit_rate(input)` | Binary: did retrieval return at least one relevant context? |
| `mean_reciprocal_rank(input)` | MRR: position-weighted relevance of first hit |
| `ndcg_at_k(input, k=5)` | NDCG@k: graded relevance of top-k results |

**Pure function pattern — why:**
- Functions are simpler to test than classes for single-output computations
- No shared state means no test ordering issues
- Each function can be imported and used independently without registry setup

---

### `aieval360/evaluation/rag/generation_metrics.py`

**4 generation metrics.** Mix of pure functions and judge-dependent functions.

| Function | Type | What it measures |
|----------|------|-----------------|
| `answer_faithfulness(input)` | Heuristic | Does the answer only use information from the retrieved contexts? |
| `answer_relevance(input)` | Heuristic | Does the answer address the query? |
| `answer_completeness(input, judge_fn)` | LLM_JUDGED | Does the answer fully resolve the query? |
| `hallucination_score(input)` | Heuristic | Is the answer grounded in the contexts, or does it introduce outside claims? |

**`judge_fn` signature:**
```python
async def judge_fn(query: str, context: str, answer: str) -> float:
    # Returns score in [0, 1]
    ...
```

When `judge_fn=None`, `answer_completeness` returns 0.5 (neutral) — not 0.0 and not 1.0. This prevents silent failure while making the dependency explicit.

---

### `aieval360/evaluation/rag/pipeline.py`

`RAGEvaluationPipeline` — the main entry point for running RAG evaluations.

```python
from aieval360.evaluation.rag.pipeline import RAGEvaluationPipeline
from aieval360.evaluation.rag.retrieval_metrics import (
    context_precision, context_recall, context_relevance
)
from aieval360.evaluation.rag.generation_metrics import (
    answer_faithfulness, answer_relevance
)

pipeline = RAGEvaluationPipeline(
    retrieval_metrics=[context_precision, context_recall, context_relevance],
    generation_metrics=[answer_faithfulness, answer_relevance],
    judge_fn=None,       # Optional LLM judge
    run_label="rag_eval_v1",
    executed_by="data_team",
)

examples = [
    RAGInput(
        query="What is the capital of France?",
        retrieved_contexts=["France is a country in Europe. Paris is its capital."],
        generated_answer="The capital of France is Paris.",
        ground_truth_answer="Paris",
    )
]

result = await pipeline.run(examples)
print(result.mean_scores)        # Average per metric across all examples
print(result.per_example_scores) # Per-example breakdown
```

**Pipeline internals:**
1. Runs all examples concurrently via `asyncio.gather`
2. Applies each retrieval metric and each generation metric to every example
3. Aggregates: mean score per metric across all examples
4. Returns `RAGPipelineResult` with per-example and aggregated scores

---

### `aieval360/evaluation/rag/registry.py`

Pre-built registry of all 11 RAG metrics (7 retrieval + 4 generation) for convenience:

```python
from aieval360.evaluation.rag.registry import (
    RAG_RETRIEVAL_METRICS,
    RAG_GENERATION_METRICS,
    build_default_rag_pipeline,
)

# Build a pipeline with all metrics registered
pipeline = build_default_rag_pipeline(judge_fn=my_llm_judge)
```

---

## Test Coverage

| File | Tests | Covers |
|------|-------|--------|
| `test_rag_retrieval_metrics.py` | ~40 | All 7 retrieval functions; edge cases (empty contexts, no ground truth) |
| `test_rag_generation_metrics.py` | ~40 | All 4 generation functions; judge_fn injection; neutral fallback |
| `test_rag_pipeline.py` | ~40 | Full pipeline run; concurrency; aggregation; run metadata |
| **Total** | **~120** | |

---

## RAG Evaluation Framework

```
Query ──────────────────────────────────────────────────┐
                                                         │
Retriever → Retrieved Contexts ──────────────────────────┼──→ RETRIEVAL METRICS
                                                         │       context_precision
                     LLM Generator ──→ Generated Answer ─┼──→   context_recall
                                                         │       context_relevance
                                                         │       retrieval_hit_rate
                                                         │       MRR, NDCG@k
                                                         │
                                                         ├──→ GENERATION METRICS
                                                         │       answer_faithfulness
                                                         │       answer_relevance
                                                         │       answer_completeness (judge)
                                                         │       hallucination_score
                                                         │
                                              Ground Truth
                                              (optional)
```

---

## Governance Notes

- Retrieval and generation failures are reported separately — this enables targeted remediation (fix the retriever vs. fix the prompt)
- `judge_fn=None` mode allows organisations to run full RAG evaluation without LLM API access
- `run_label` and `executed_by` are propagated to `RAGPipelineResult` for audit trail continuity
- `hallucination_score` uses context-grounding heuristics consistent with the safety evaluator's `hallucination_detected` function (Stage 4)
