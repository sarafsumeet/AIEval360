"""
Evaluation run and evidence models.
Every evaluation run produces evidence that is immutable and traceable.
"""

from __future__ import annotations

import enum
import json
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, Enum, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, UUIDPrimaryKey


class EvaluationStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class EvaluationCategory(str, enum.Enum):
    ML_CLASSIFICATION = "ml_classification"
    ML_REGRESSION = "ml_regression"
    LLM_BENCHMARK = "llm_benchmark"
    RAG_PIPELINE = "rag_pipeline"
    AI_AGENT = "ai_agent"
    SAFETY_ADVERSARIAL = "safety_adversarial"
    MULTIMODAL = "multimodal"
    GOVERNANCE = "governance"


class MetricType(str, enum.Enum):
    """Measurement type per evaluation methodology."""
    DETERMINISTIC = "deterministic"           # Computed from known ground truth
    REFERENCE_BASED = "reference_based"       # Computed against reference labels
    HEURISTIC = "heuristic"                   # Rule-based estimate
    LLM_JUDGED = "llm_judged"                 # Scored by a judge LLM
    HUMAN_ASSESSED = "human_assessed"         # Reviewed by a human


class DecisionOutcome(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL_PASS = "conditional_pass"
    NOT_EVALUATED = "not_evaluated"
    BLOCKED = "blocked"  # Failed mandatory gate


class Dataset(Base, UUIDPrimaryKey, TimestampMixin):
    """
    Versioned dataset used for evaluation.
    Datasets are immutable after registration — versions are created for changes.
    """

    __tablename__ = "datasets"

    organisation_id: Mapped[str] = mapped_column(String(36), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(100))  # banking/healthcare/etc.
    evaluation_type: Mapped[Optional[str]] = mapped_column(String(100))
    record_count: Mapped[Optional[int]] = mapped_column()
    file_path: Mapped[Optional[str]] = mapped_column(String(1000))
    schema_version: Mapped[Optional[str]] = mapped_column(String(50))
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    checksum: Mapped[Optional[str]] = mapped_column(String(255))  # SHA256
    metadata_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON blob

    evaluation_runs: Mapped[list["EvaluationRun"]] = relationship(
        "EvaluationRun", back_populates="dataset"
    )

    def get_metadata(self) -> dict:
        if self.metadata_json:
            return json.loads(self.metadata_json)
        return {}

    def __repr__(self) -> str:
        return f"<Dataset id={self.id} name={self.name!r} version={self.version!r}>"


class EvaluationRun(Base, UUIDPrimaryKey, TimestampMixin):
    """
    A single evaluation run — the atomic unit of assurance evidence.
    Every field here is evidence; this record is append-only after completion.
    """

    __tablename__ = "evaluation_runs"

    # Identification
    organisation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    ai_system_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("ai_systems.id", ondelete="SET NULL"), nullable=True
    )
    dataset_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )

    # What was evaluated
    evaluation_category: Mapped[str] = mapped_column(
        Enum(EvaluationCategory), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Provider / model evidence (immutable after completion)
    provider_name: Mapped[Optional[str]] = mapped_column(String(100))
    model_name: Mapped[Optional[str]] = mapped_column(String(255))
    model_version: Mapped[Optional[str]] = mapped_column(String(100))
    model_parameters_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    prompt_template_version: Mapped[Optional[str]] = mapped_column(String(50))

    # Judge model (if LLM-as-judge was used)
    judge_model: Mapped[Optional[str]] = mapped_column(String(255))
    judge_model_version: Mapped[Optional[str]] = mapped_column(String(100))

    # Evaluation configuration evidence
    evaluation_config_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON
    metric_definitions_version: Mapped[Optional[str]] = mapped_column(String(50))
    thresholds_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON

    # Execution
    status: Mapped[str] = mapped_column(
        Enum(EvaluationStatus), default=EvaluationStatus.PENDING, nullable=False
    )
    started_at: Mapped[Optional[str]] = mapped_column(String(50))
    completed_at: Mapped[Optional[str]] = mapped_column(String(50))
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    executed_by: Mapped[Optional[str]] = mapped_column(String(255))

    # Results
    overall_score: Mapped[Optional[float]] = mapped_column(Float)
    overall_outcome: Mapped[Optional[str]] = mapped_column(Enum(DecisionOutcome))
    scores_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON dict of scores
    metrics_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON dict of detailed metrics
    errors_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON list of errors/warnings

    # Reproducibility
    is_reproducible: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    reproducibility_notes: Mapped[Optional[str]] = mapped_column(Text)

    # Human review
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(255))
    review_notes: Mapped[Optional[str]] = mapped_column(Text)
    review_outcome: Mapped[Optional[str]] = mapped_column(Enum(DecisionOutcome))

    # Relationships
    organisation: Mapped["Organisation"] = relationship("Organisation")
    project: Mapped[Optional["Project"]] = relationship("Project")
    ai_system: Mapped[Optional["AISystem"]] = relationship("AISystem")
    dataset: Mapped[Optional["Dataset"]] = relationship(
        "Dataset", back_populates="evaluation_runs"
    )
    test_cases: Mapped[list["EvaluationTestCase"]] = relationship(
        "EvaluationTestCase", back_populates="evaluation_run", cascade="all, delete-orphan"
    )
    metric_results: Mapped[list["MetricResult"]] = relationship(
        "MetricResult", back_populates="evaluation_run", cascade="all, delete-orphan"
    )

    def get_scores(self) -> dict[str, float]:
        if self.scores_json:
            return json.loads(self.scores_json)
        return {}

    def get_metrics(self) -> dict[str, Any]:
        if self.metrics_json:
            return json.loads(self.metrics_json)
        return {}

    def get_errors(self) -> list[str]:
        if self.errors_json:
            return json.loads(self.errors_json)
        return []

    def get_model_parameters(self) -> dict:
        if self.model_parameters_json:
            return json.loads(self.model_parameters_json)
        return {}

    def __repr__(self) -> str:
        return (
            f"<EvaluationRun id={self.id} category={self.evaluation_category} "
            f"status={self.status} score={self.overall_score}>"
        )


class EvaluationTestCase(Base, UUIDPrimaryKey, TimestampMixin):
    """
    Individual test case result within an evaluation run.
    Provides the atomic evidence unit for failed case review and audit.
    """

    __tablename__ = "evaluation_test_cases"

    evaluation_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False
    )
    test_case_id: Mapped[Optional[str]] = mapped_column(String(255))  # from dataset
    test_case_name: Mapped[Optional[str]] = mapped_column(String(255))
    category: Mapped[Optional[str]] = mapped_column(String(100))

    # Input
    input_text: Mapped[Optional[str]] = mapped_column(Text)
    input_metadata_json: Mapped[Optional[str]] = mapped_column(Text)
    expected_output: Mapped[Optional[str]] = mapped_column(Text)
    reference_context: Mapped[Optional[str]] = mapped_column(Text)

    # Output
    actual_output: Mapped[Optional[str]] = mapped_column(Text)
    retrieved_context: Mapped[Optional[str]] = mapped_column(Text)  # RAG
    tool_calls_json: Mapped[Optional[str]] = mapped_column(Text)  # Agent

    # Scores
    scores_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON dict
    outcome: Mapped[Optional[str]] = mapped_column(Enum(DecisionOutcome))
    failure_reason: Mapped[Optional[str]] = mapped_column(Text)

    # Latency / cost
    latency_ms: Mapped[Optional[float]] = mapped_column(Float)
    tokens_input: Mapped[Optional[int]] = mapped_column()
    tokens_output: Mapped[Optional[int]] = mapped_column()
    estimated_cost_usd: Mapped[Optional[float]] = mapped_column(Float)

    evaluation_run: Mapped["EvaluationRun"] = relationship(
        "EvaluationRun", back_populates="test_cases"
    )

    def get_scores(self) -> dict[str, float]:
        if self.scores_json:
            return json.loads(self.scores_json)
        return {}

    def __repr__(self) -> str:
        return f"<TestCase id={self.id} outcome={self.outcome}>"


class MetricResult(Base, UUIDPrimaryKey, TimestampMixin):
    """
    A single metric result within an evaluation run.
    Stores the metric definition alongside the result for reproducibility.
    """

    __tablename__ = "metric_results"

    evaluation_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False
    )
    metric_name: Mapped[str] = mapped_column(String(255), nullable=False)
    metric_category: Mapped[Optional[str]] = mapped_column(String(100))
    metric_type: Mapped[str] = mapped_column(
        Enum(MetricType), default=MetricType.DETERMINISTIC, nullable=False
    )

    # Definition (preserved for reproducibility and audit)
    metric_definition: Mapped[Optional[str]] = mapped_column(Text)
    calculation_method: Mapped[Optional[str]] = mapped_column(Text)
    known_limitations: Mapped[Optional[str]] = mapped_column(Text)

    # Result
    value: Mapped[Optional[float]] = mapped_column(Float)
    value_json: Mapped[Optional[str]] = mapped_column(Text)  # for complex values
    threshold: Mapped[Optional[float]] = mapped_column(Float)
    outcome: Mapped[Optional[str]] = mapped_column(Enum(DecisionOutcome))
    is_mandatory_gate: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    evidence_source: Mapped[Optional[str]] = mapped_column(Text)

    evaluation_run: Mapped["EvaluationRun"] = relationship(
        "EvaluationRun", back_populates="metric_results"
    )

    def __repr__(self) -> str:
        return (
            f"<MetricResult metric={self.metric_name!r} "
            f"value={self.value} outcome={self.outcome}>"
        )
