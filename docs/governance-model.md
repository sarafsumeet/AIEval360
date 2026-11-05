# Governance Model

**Purpose:** AIEval360's governance design — audit trail structure, mandatory gates, evidence artefacts, and how the platform connects evaluation to governance decision-making.

---

## Principle: Evaluation ≠ Governance

A common failure in AI governance programs is conflating *evaluation* (measuring model quality) with *governance* (managing model risk and decisions). AIEval360 keeps them separate and connects them through *assurance*:

```
EVALUATION          ASSURANCE           GOVERNANCE
─────────────       ─────────────       ──────────────────────
Measure quality  →  Interpret results  →  Make risk decisions
Run metrics         Assess thresholds    Approve / reject / monitor
Collect evidence    Gate on critical     Record decisions
                    failures             Manage accountability
```

Evaluation without governance produces numbers with no organisational weight.  
Governance without evaluation produces decisions with no empirical basis.

---

## Audit Trail Design

Every evaluation result in AIEval360 carries provenance from input to decision.

### Layer 1 — Dataset Lineage

```python
# Every EvaluationRun stores the input dataset's SHA-256 checksum
run.metadata["dataset_checksum"] = loader.checksum_dataframe(df)
```

The checksum is column-order-independent. If the dataset changes between runs — even one row, one column rename — the checksum changes. This makes two results with different checksums **incomparable**, which the governance layer enforces.

### Layer 2 — Run Provenance

Every `EvaluationRun` stores:

| Field | Purpose |
|-------|---------|
| `run_id` | Unique identifier (UUID) for this evaluation run |
| `run_label` | Human-readable name ("ci_pipeline_v2", "pre-release-safety") |
| `executed_by` | Identity of the team or person who initiated the run |
| `model_id` | The model being evaluated |
| `dataset_id` | The dataset version used |
| `started_at` / `completed_at` | Timestamps |
| `status` | PENDING / RUNNING / COMPLETED / FAILED |

### Layer 3 — Result Traceability

Every `EvaluationResult` stores:

| Field | Purpose |
|-------|---------|
| `result_id` | Unique identifier (UUID) |
| `run_id` | Links back to the parent EvaluationRun |
| `metric_name` | The specific metric computed |
| `metric_value` | Score in [0, 1] |
| `metric_type` | DETERMINISTIC / HEURISTIC / LLM_JUDGED / etc. |
| `input_hash` | SHA-256 of the specific input row |
| `details` | Metric-specific breakdown (e.g., BLEU n-gram scores) |

### Layer 4 — Governance Record

A `GovernanceRecord` links an `EvaluationRun` to a governance decision:

```python
GovernanceRecord(
    evaluation_run_id = "run-uuid",
    risk_level        = "HIGH",
    decision          = "CONDITIONAL_APPROVAL",
    approved_by       = "AI Governance Committee",
    approval_date     = date.today(),
    notes             = "Approved with mandatory monitoring for 90 days",
    mandatory_controls = [
        "Monthly re-evaluation with same probe set",
        "Human review of all CRITICAL safety probe responses",
        "Bias audit before next model version",
    ]
)
```

---

## Mandatory Gate

The mandatory gate is the **binary deployment decision point** in the safety evaluation pipeline.

```python
@property
def mandatory_gate_passed(self) -> bool:
    return not any(r.is_critical_failure for r in self.probe_results)
```

**What counts as a critical failure:**
- A probe with `severity = CRITICAL` where `outcome = FAILED`

**Gate behaviour:**

| Probe Severity | Failure Effect |
|----------------|----------------|
| CRITICAL | Gate fails — deployment blocked |
| HIGH | Recorded in report — gate passes |
| MEDIUM | Recorded in report — gate passes |
| LOW | Recorded in report — gate passes |

**Why binary:** Governance committees need a clear signal. A gate that sometimes passes on CRITICAL failures with caveats is not a gate — it is a suggestion. The mandatory gate is a hard stop.

---

## Evidence Artefact Model

AIEval360 is designed to generate governance evidence that satisfies AI governance committee and regulatory requirements.

### Recommended Evidence Package

For each model evaluation cycle, the following artefacts should be collected:

```
evidence/
├── evaluation/
│   ├── run_metadata.json           # EvaluationRun record
│   ├── metric_results.json         # All EvaluationResult records
│   ├── dataset_checksum.txt        # SHA-256 of evaluation dataset
│   └── summary_report.md          # Human-readable summary
├── safety/
│   ├── safety_evaluation_report.json  # SafetyEvaluationReport record
│   ├── probe_results_detail.json      # Per-probe outcomes
│   ├── mandatory_gate_status.txt      # PASSED / FAILED
│   └── critical_failures.json         # If gate failed, details
├── rag/ (if applicable)
│   ├── rag_pipeline_result.json
│   └── mean_scores.json
├── agents/ (if applicable)
│   ├── agent_evaluation_results.json
│   └── loop_detection_summary.json
└── governance/
    ├── risk_assessment.json         # RiskAssessment record
    └── governance_record.json       # GovernanceRecord with decision
```

### Evidence Traceability Chain

```
Raw Dataset
     │ SHA-256 checksum
     ▼
EvaluationRun (run_id, model_id, executed_by)
     │ run_id FK
     ▼
EvaluationResults (per metric, per input row)
     │
     ▼
SafetyEvaluationReport (probe_id, outcome, severity)
     │ mandatory_gate_passed
     ▼
RiskAssessment (overall_risk_level, mandatory_controls)
     │
     ▼
GovernanceRecord (decision, approved_by, approval_date)
```

Every link in this chain is a foreign-key relationship in the data model. The chain is **unbroken from dataset to governance decision**.

---

## ModelCard and Organisational Context

### ModelCard

`ModelCard` captures the provenance and intended use of the model being evaluated:

```python
ModelCard(
    model_id              = "gpt-4o-2024-11",
    model_name            = "GPT-4o",
    model_version         = "2024-11",
    provider              = "OpenAI",
    intended_use          = "Customer service question answering",
    out_of_scope_uses     = ["Medical advice", "Legal advice", "Financial decisions"],
    training_data_description = "...",
    known_limitations     = ["May hallucinate product details", "Limited to English"],
    fairness_considerations   = "Evaluated for demographic parity across age/gender groups",
    governance_contact    = "ai-governance@organisation.com",
)
```

This card is a mandatory input for the governance record — a decision made about an unidentified model is not an auditable decision.

### Organisation

`Organisation` supports multi-tenancy for enterprise deployments:

```python
Organisation(
    org_id   = "uuid",
    org_name = "ACME Financial Services",
    metadata = {"regulatory_regime": "APRA", "industry": "banking"},
)
```

---

## Risk Assessment

`RiskAssessment` captures the structured risk profile of a model at evaluation time:

```python
RiskAssessment(
    model_id          = "gpt-4o-2024-11",
    overall_risk_level = "HIGH",
    risk_dimensions = {
        "safety":         0.85,  # Safety probe pass rate
        "accuracy":       0.72,  # F1 against ground truth
        "hallucination":  0.61,  # Inverse of hallucination rate
        "bias":           0.78,  # Fairness score
    },
    mandatory_controls = [
        "Human review of all HIGH-severity safety failures",
        "Re-evaluation after each model update",
    ]
)
```

---

## Regulatory Alignment

AIEval360's governance model is designed to be compatible with common AI governance frameworks:

| Framework | Alignment |
|-----------|-----------|
| **APRA CPS 234** | Audit trail satisfies information asset governance requirements; mandatory gate provides operational control |
| **ISO/IEC 42001** | ModelCard, GovernanceRecord, and RiskAssessment map to AI system documentation requirements |
| **DAMA DMBOK** | Dataset lineage (SHA-256 checksums) addresses data quality and lineage requirements |
| **NIST AI RMF** | Mandatory gate, risk assessment, and governance record implement GOVERN, MAP, MEASURE, MANAGE functions |
| **EU AI Act** | Evidence artefact model supports technical documentation requirements for high-risk AI systems |

**Important:** This mapping is indicative. Formal regulatory compliance requires legal review and assessment against the specific obligations applicable to each organisation and jurisdiction.

---

## Governance vs. Evaluation — Separation Illustrated

```
Team              Responsibility                   Platform Component
────────────────  ───────────────────────────      ──────────────────────
Data / ML Team    Run evaluations                  EvaluationOrchestrator
                  Collect metrics                  MetricRegistry
                  Document dataset lineage         DatasetLoader (SHA-256)

Safety Team       Run safety probes                SafetyEvaluator
                  Identify critical failures       SafetyEvaluationReport
                  Confirm mandatory gate           mandatory_gate_passed

AI Governance     Interpret results                GovernanceRecord
Committee         Assess risk                      RiskAssessment
                  Make deployment decisions        (Human decision)
                  Record decisions with rationale  GovernanceRecord.notes

Audit / Compliance  Review evidence chain          Evidence artefacts
                    Verify traceability            run_id, checksums, FKs
```

The platform provides the **data and structure** for governance decisions. It does not make governance decisions itself.
