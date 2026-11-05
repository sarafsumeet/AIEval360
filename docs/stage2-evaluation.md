# Stage 2 — Evaluation Engine

**Files:** 6 source · 5 test files · 120 tests  
**Purpose:** Metric calculators, metric registry, dataset loading with lineage, evaluation orchestrator

---

## What Was Built

Stage 2 is the core evaluation engine. It introduces 14 metric calculators across 5 metric types, a runtime registry for metric discovery, a dataset loader with SHA-256 lineage tracking, and an async orchestrator that runs evaluations end-to-end.

---

## Module Reference

### `aieval360/evaluation/metrics/base.py`

Defines the foundational abstractions for all metrics.

#### `MetricType` (Enum)

```python
class MetricType(str, Enum):
    DETERMINISTIC   = "deterministic"    # Exact computation
    REFERENCE_BASED = "reference_based"  # Requires ground truth
    HEURISTIC       = "heuristic"        # Rule-based approximation
    LLM_JUDGED      = "llm_judged"       # Calls an LLM scorer
    HUMAN_ASSESSED  = "human_assessed"   # Requires human review
```

#### `MetricInput`

```python
@dataclass
class MetricInput:
    prediction: str
    reference: str | None = None      # Ground truth (required by REFERENCE_BASED)
    context: str | None = None        # Source context (used by some metrics)
    metadata: dict = field(default_factory=dict)
```

#### `MetricOutput`

```python
@dataclass
class MetricOutput:
    metric_name: str
    metric_type: MetricType
    value: float                        # Primary score, always in [0, 1]
    details: dict = field(default_factory=dict)
    reasoning: str = ""
    computed_at: datetime = field(default_factory=datetime.utcnow)
```

#### `MetricCalculator` (ABC)

```python
class MetricCalculator(ABC):
    metric_name: str         # Must be unique across the registry
    metric_type: MetricType

    @abstractmethod
    def calculate(self, input: MetricInput) -> MetricOutput: ...
```

---

### `aieval360/evaluation/metrics/traditional_ml.py`

**8 DETERMINISTIC calculators:**

| Class | Metric | Input required |
|-------|--------|---------------|
| `AccuracyCalculator` | Exact match accuracy | prediction, reference |
| `PrecisionCalculator` | Token-level precision | prediction, reference |
| `RecallCalculator` | Token-level recall | prediction, reference |
| `F1Calculator` | Harmonic mean of P & R | prediction, reference |
| `BLEUCalculator` | BLEU-1 n-gram overlap | prediction, reference |
| `ROUGELCalculator` | ROUGE-L longest common subsequence | prediction, reference |
| `ExactMatchCalculator` | Binary exact match (0 or 1) | prediction, reference |
| `CharacterErrorRateCalculator` | CER — edit distance / reference length | prediction, reference |

All calculators handle edge cases: empty strings return 0.0 rather than raising.

---

### `aieval360/evaluation/metrics/llm_metrics.py`

**6 calculators across HEURISTIC and LLM_JUDGED types:**

| Class | Type | Description |
|-------|------|-------------|
| `CoherenceCalculator` | HEURISTIC | Sentence-structure heuristic (avg sentence length normalised) |
| `FluencyCalculator` | HEURISTIC | Vocabulary diversity (unique tokens / total tokens) |
| `ReadabilityCalculator` | HEURISTIC | Flesch-style syllable count approximation |
| `RelevanceCalculator` | HEURISTIC | Token overlap between prediction and context |
| `AnswerCompletenessCalculator` | LLM_JUDGED | Requires judge_fn; falls back to 0.5 if no judge provided |
| `FactualConsistencyCalculator` | LLM_JUDGED | Requires judge_fn; falls back to 0.5 if no judge provided |

**Note on LLM_JUDGED calculators:** When no `judge_fn` is provided, they return 0.5 (neutral score) with `reasoning="no_judge_provided"`. This is intentional — it prevents silent failures while making the dependency explicit.

---

### `aieval360/evaluation/metrics/registry.py`

Runtime registry for metric discovery and instantiation.

```python
from aieval360.evaluation.metrics.registry import MetricRegistry, default_registry

# Use the pre-built default registry (all 14 metrics registered)
calc = default_registry.get("f1")
result = calc.calculate(MetricInput(prediction="hello", reference="hello world"))

# List by type
deterministic = default_registry.list_by_type(MetricType.DETERMINISTIC)

# Build a fresh registry
registry = MetricRegistry()
registry.register(F1Calculator())
```

**Key methods:**

| Method | Description |
|--------|-------------|
| `register(calculator)` | Add a calculator; raises if name conflicts |
| `get(metric_name)` | Return calculator or raise `KeyError` |
| `list_by_type(metric_type)` | Return all calculators of that type |
| `build_default_registry()` | Factory: returns registry with all 14 built-in metrics |

`default_registry` is a module-level singleton — the pre-built registry for immediate use.

---

### `aieval360/evaluation/datasets/loader.py`

Dataset loading with SHA-256 lineage tracking.

```python
from aieval360.evaluation.datasets.loader import DatasetLoader

loader = DatasetLoader()

# Load from CSV
df, checksum = loader.load_csv("path/to/dataset.csv")

# Load from JSON
df, checksum = loader.load_json("path/to/dataset.json")

# Compute checksum for an existing DataFrame
checksum = loader.checksum_dataframe(df)
```

**Checksum implementation:**

```python
# Column-order-independent: sorts columns before serialising
json_str = df[sorted(df.columns)].to_json(orient="records")
return hashlib.sha256(json_str.encode()).hexdigest()
```

The checksum is stored alongside evaluation results in `EvaluationRun.metadata["dataset_checksum"]`. If the dataset changes between runs, the checksum changes — making result comparisons invalid.

---

### `aieval360/evaluation/runners/orchestrator.py`

`EvaluationOrchestrator` runs a full evaluation job: loads a dataset, applies a list of metrics to each row, and aggregates results.

```python
from aieval360.evaluation.runners.orchestrator import (
    EvaluationOrchestrator,
    OrchestratorConfig,
)

config = OrchestratorConfig(
    run_label="ci_pipeline_v2",
    executed_by="governance_team",
    max_concurrency=5,
)

orchestrator = EvaluationOrchestrator(config)
run_result = await orchestrator.run(
    dataset=df,
    metrics=["f1", "bleu", "rouge_l"],
    prediction_col="model_output",
    reference_col="ground_truth",
)

print(run_result.summary)        # Aggregated scores per metric
print(run_result.dataset_checksum)  # SHA-256 lineage
```

---

## Test Coverage

| File | Tests | Covers |
|------|-------|--------|
| `test_metric_base.py` | ~15 | MetricType enum, MetricInput/Output, ABC contract |
| `test_traditional_ml_metrics.py` | ~30 | All 8 DETERMINISTIC calculators, edge cases |
| `test_llm_metrics.py` | ~25 | All 6 LLM/heuristic calculators, judge_fn injection |
| `test_dataset_loader.py` | ~20 | CSV/JSON load, checksum determinism, column-order independence |
| `test_orchestrator.py` | ~30 | Full run, metric selection, concurrency, run metadata |
| **Total** | **~120** | |

---

## Governance Notes

- Dataset checksums create an unbroken lineage from raw data → evaluation result
- `run_label` and `executed_by` on every orchestrator run satisfy audit trail requirements
- All 8 DETERMINISTIC metrics are fully reproducible: given the same input, they will always produce the same output
- LLM_JUDGED metrics are explicitly marked as requiring a judge — no silent degradation to heuristics
