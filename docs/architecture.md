# AIEval360 — System Architecture

---

## Design Philosophy

AIEval360 follows **governance-by-design**: every architectural decision is made with audit trails, reproducibility, and regulatory alignment in mind — not retrofitted as an afterthought.

Three principles govern the architecture:

1. **Separation of concerns** — Evaluation (measuring quality) and Governance (managing risk) are distinct layers. They connect through the assurance layer (Stage 5), not by conflating them.
2. **Pure functions where possible** — Metric calculators are stateless; dataset loading is deterministic; pattern detection is regex/keyword-based (not stochastic LLM calls) so results are reproducible.
3. **Pluggable providers** — The evaluation engine never depends on a specific AI vendor. Provider adapters are swapped without touching evaluation logic.

---

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GOVERNANCE LAYER                        │
│  Audit trails · Mandatory gates · Evidence artefacts       │
│  (Stage 5 — Assurance)                                      │
├────────────────────────┬────────────────────────────────────┤
│   SAFETY EVALUATION    │      AGENT EVALUATION              │
│   Stage 4              │      Stage 4                       │
│   40 adversarial probes│   Trace analysis · Loop detection  │
│   Pattern detection    │   5-metric composite score         │
├────────────────────────┴────────────────────────────────────┤
│                   RAG EVALUATION PIPELINE                   │
│   Stage 3 — Retrieval + Generation metrics                  │
│   Async pipeline · Optional LLM judge                       │
├─────────────────────────────────────────────────────────────┤
│              EVALUATION ENGINE (Metrics + Datasets)         │
│   Stage 2 — 14 calculators · Registry · Orchestrator        │
│   DETERMINISTIC · REFERENCE_BASED · HEURISTIC              │
│   LLM_JUDGED · HUMAN_ASSESSED (taxonomy)                   │
├─────────────────────────────────────────────────────────────┤
│                     FOUNDATION                              │
│   Stage 1 — Core models · Provider adapters · Persistence  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Provider Adapter Pattern (Stage 1)

`BaseProviderAdapter` is an abstract base class (ABC). All vendor integrations — OpenAI, Ollama, Mock — implement the same interface:

```python
class BaseProviderAdapter(ABC):
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str: ...
    @abstractmethod
    async def health_check(self) -> bool: ...
```

**Why:** Decouples the evaluation engine from vendor lock-in. Tests always use the `MockAdapter`. Production swaps it transparently.

---

### Metric Calculator ABC (Stage 2)

Every metric is a stateless class implementing `MetricCalculator`:

```python
class MetricCalculator(ABC):
    metric_name: str         # unique key
    metric_type: MetricType  # taxonomy
    def calculate(self, input: MetricInput) -> MetricOutput: ...
```

**Why:** Stateless calculators are testable in isolation, composable in pipelines, and registerable at runtime.

**MetricType taxonomy:**

| Type | Description | Example |
|------|-------------|---------|
| `DETERMINISTIC` | Exact computation, no variance | F1, Accuracy, BLEU |
| `REFERENCE_BASED` | Requires ground truth | Precision, Recall, ROUGE |
| `HEURISTIC` | Rule-based approximation | Readability, Length |
| `LLM_JUDGED` | Calls an LLM to score | Coherence, Relevance |
| `HUMAN_ASSESSED` | Requires human review | Fairness, Ethics |

---

### Dataset Lineage (Stage 2)

`DatasetLoader.checksum_dataframe()` computes a SHA-256 hash of dataset contents:

```python
# Column-order-independent checksum
json_str = df[sorted(df.columns)].to_json(orient="records")
return hashlib.sha256(json_str.encode()).hexdigest()
```

**Why:** A dataset checksum is the first link in an AI governance audit trail. If the checksum changes between evaluation runs, the results are not comparable. Governance committees require this traceability.

---

### RAG Pipeline Design (Stage 3)

`RAGEvaluationPipeline` runs all examples concurrently with `asyncio.gather`. An optional `judge_fn` parameter lets callers inject an LLM judge for generation metrics:

```python
pipeline = RAGEvaluationPipeline(
    retrieval_metrics=[...],
    generation_metrics=[...],
    judge_fn=my_llm_judge,   # Optional; None = heuristic fallback
)
results = await pipeline.run(examples)
```

**Why:** Most organisations cannot afford LLM calls for every evaluation. The heuristic fallback makes the platform usable at zero LLM cost. The optional judge enables higher fidelity when budget allows.

---

### Safety Gate Design (Stage 4)

`SafetyEvaluationReport.mandatory_gate_passed` is the single boolean used in CI/CD pipelines:

```python
@property
def mandatory_gate_passed(self) -> bool:
    return not any(r.is_critical_failure for r in self.probe_results)
```

A `CRITICAL` probe failure blocks the gate. `HIGH`, `MEDIUM`, `LOW` failures are recorded but do not block deployment.

**Why:** Governance committees need a clear, unambiguous signal. A "nuanced" gate that sometimes passes on critical failures is not a gate — it is a suggestion.

---

### Pattern-Based Safety Detection (Stage 4)

All detection functions (`refusal_detected`, `pii_detected`, `toxicity_detected`, `hallucination_detected`) use hardcoded regex patterns and keyword lists — not LLM calls.

**Why:**
- **Reproducible:** Same response always produces same detection result
- **Auditable:** Any governance auditor can review a list of 20–30 patterns
- **Cost-free:** No LLM API calls required for safety evaluation
- **Fast:** Sub-millisecond detection; supports high-volume probe runs

---

### Loop Detection (Stage 4)

Agent loop detection hashes `(tool_name, normalised_input)` pairs using SHA-256:

```python
key = f"{step.tool_name}::{_normalise_input(step.input)}"
key_hash = hashlib.sha256(key.encode()).hexdigest()[:8]
```

`_normalise_input` uses `json.dumps(value, sort_keys=True)` so dict key ordering does not produce false negatives.

**Why:** Structural comparison (not string comparison) means `{"a":1,"b":2}` and `{"b":2,"a":1}` are correctly identified as the same call.

---

### Goal Completion Heuristic (Stage 4)

```python
def _goal_completion_heuristic(goal: str, final_answer: str) -> float:
    goal_tokens = set(goal.lower().split())
    answer_tokens = set(final_answer.lower().split())
    overlap = goal_tokens & answer_tokens
    return len(overlap) / len(goal_tokens)
```

This is a Jaccard-style overlap score. It is intentionally simple and auditable.

**Why:** Organisations with LLM budgets can override via `AgentEvaluatorConfig.goal_completion_fn`. The heuristic is a sensible, zero-cost default that any evaluator can explain to a board.

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.13 |
| Data validation | Pydantic | v2 |
| Web framework | FastAPI | 0.115+ |
| ORM | SQLAlchemy (async) | 2.0 |
| Data manipulation | pandas + numpy | latest |
| Testing | pytest + pytest-asyncio | latest |
| Package management | pyproject.toml | PEP 517 |
