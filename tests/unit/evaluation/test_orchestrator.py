"""
Unit tests for the EvaluationOrchestrator.

Uses MockProviderAdapter (no real API calls) and the default metric registry.
All tests are async (pytest-asyncio).
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from aieval360.core.models.base import new_uuid
from aieval360.evaluation.metrics.base import MetricInput
from aieval360.evaluation.metrics.registry import build_default_registry
from aieval360.evaluation.runners.orchestrator import EvaluationOrchestrator, EvaluationSpec
from aieval360.providers.mock.adapter import MockProviderAdapter


@pytest.fixture()
def provider():
    return MockProviderAdapter()


@pytest.fixture()
def registry():
    return build_default_registry()


@pytest.fixture()
def orchestrator(registry, provider):
    return EvaluationOrchestrator(registry=registry, provider=provider)


def _spec(**overrides) -> EvaluationSpec:
    defaults = dict(
        name="Test Run",
        evaluation_category="llm_benchmark",
        organisation_id=new_uuid(),
        project_id=new_uuid(),
        ai_system_id=new_uuid(),
        dataset_id=new_uuid(),
        provider_name="mock",
        model_name="mock-gpt-v1",
        model_version="1.0.0",
        executed_by="test_user",
    )
    defaults.update(overrides)
    return EvaluationSpec(**defaults)


class TestOrchestratorBasicRun:
    @pytest.mark.asyncio
    async def test_empty_dataset_runs_without_error(self, orchestrator):
        spec = _spec(dataset_rows=[], metrics=[("accuracy", 0.8)])
        result = await orchestrator.run(spec)
        assert result is not None
        assert result.total_rows == 0

    @pytest.mark.asyncio
    async def test_precomputed_predictions_no_provider_call(self, orchestrator, provider):
        rows = [
            {"prediction": "cat", "reference": "cat"},
            {"prediction": "dog", "reference": "dog"},
        ]
        spec = _spec(dataset_rows=rows, metrics=[("accuracy", 0.8)])
        result = await orchestrator.run(spec)
        # Provider should not have been called — predictions already in rows
        assert provider.call_count == 0
        assert result.completed_rows == 2

    @pytest.mark.asyncio
    async def test_input_rows_trigger_provider_call(self, orchestrator, provider):
        rows = [{"input": "What is 2+2?", "reference": "4"}]
        spec = _spec(dataset_rows=rows, metrics=[("non_empty_response", 1.0)])
        result = await orchestrator.run(spec)
        assert provider.call_count == 1
        assert result.completed_rows == 1

    @pytest.mark.asyncio
    async def test_elapsed_time_recorded(self, orchestrator):
        spec = _spec(dataset_rows=[], metrics=[])
        result = await orchestrator.run(spec)
        assert result.elapsed_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_unknown_metric_adds_error_does_not_crash(self, orchestrator):
        spec = _spec(dataset_rows=[], metrics=[("nonexistent_metric", 0.5)])
        result = await orchestrator.run(spec)
        assert any("nonexistent_metric" in e for e in result.errors)


class TestOrchestratorMetricResults:
    @pytest.mark.asyncio
    async def test_accuracy_computed_from_precomputed_rows(self, orchestrator):
        rows = [
            {"prediction": "A", "reference": "A"},
            {"prediction": "B", "reference": "A"},
        ]
        spec = _spec(dataset_rows=rows, metrics=[("accuracy", 0.8)])
        result = await orchestrator.run(spec)
        accuracy_outputs = [m for m in result.metric_outputs if m.metric_name == "accuracy"]
        assert len(accuracy_outputs) == 1
        assert accuracy_outputs[0].value == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_multiple_metrics_returned(self, orchestrator):
        rows = [{"prediction": "hello world", "reference": "hello world"}]
        spec = _spec(
            dataset_rows=rows,
            metrics=[("accuracy", 0.8), ("non_empty_response", 1.0)],
        )
        result = await orchestrator.run(spec)
        names = [m.metric_name for m in result.metric_outputs]
        assert "accuracy" in names
        assert "non_empty_response" in names


class TestOrchestratorMandatoryGates:
    @pytest.mark.asyncio
    async def test_failed_metric_reported_in_mandatory_gate_failures(self, orchestrator):
        # accuracy=0.5 fails threshold=0.9
        rows = [
            {"prediction": "A", "reference": "A"},
            {"prediction": "B", "reference": "A"},
        ]
        spec = _spec(dataset_rows=rows, metrics=[("accuracy", 0.9)])
        result = await orchestrator.run(spec)
        # passed=False means it's a gate failure
        failures = result.mandatory_gate_failures
        assert len(failures) >= 1

    @pytest.mark.asyncio
    async def test_has_critical_failure_when_gate_fails(self, orchestrator):
        rows = [{"prediction": "wrong", "reference": "right"}]
        spec = _spec(dataset_rows=rows, metrics=[("accuracy", 0.9)])
        result = await orchestrator.run(spec)
        assert result.has_critical_failure is True

    @pytest.mark.asyncio
    async def test_no_critical_failure_when_all_pass(self, orchestrator):
        rows = [{"prediction": "A", "reference": "A"}]
        spec = _spec(dataset_rows=rows, metrics=[("accuracy", 0.5)])
        result = await orchestrator.run(spec)
        assert result.has_critical_failure is False

    @pytest.mark.asyncio
    async def test_overall_status_completed_when_all_pass(self, orchestrator):
        rows = [{"prediction": "X", "reference": "X"}]
        spec = _spec(dataset_rows=rows, metrics=[("accuracy", 0.5)])
        result = await orchestrator.run(spec)
        assert result.overall_status == "completed"

    @pytest.mark.asyncio
    async def test_overall_status_completed_with_failures(self, orchestrator):
        rows = [{"prediction": "wrong", "reference": "right"}]
        spec = _spec(dataset_rows=rows, metrics=[("accuracy", 0.9)])
        result = await orchestrator.run(spec)
        assert result.overall_status == "completed_with_failures"


class TestOrchestratorProviderFailure:
    @pytest.mark.asyncio
    async def test_failed_provider_calls_counted(self):
        failing_provider = MockProviderAdapter(fail_on_calls=[0, 1])
        registry = build_default_registry()
        orch = EvaluationOrchestrator(registry=registry, provider=failing_provider)
        rows = [{"input": "test"}, {"input": "test2"}]
        spec = _spec(dataset_rows=rows, metrics=[("non_empty_response", 1.0)])
        result = await orch.run(spec)
        assert result.failed_rows == 2
        assert result.completed_rows == 0
        assert len(result.errors) == 2


class TestRunResultSpec:
    @pytest.mark.asyncio
    async def test_spec_preserved_in_result(self, orchestrator):
        spec = _spec(name="Evidence Test Run", executed_by="governance.officer")
        result = await orchestrator.run(spec)
        assert result.spec.name == "Evidence Test Run"
        assert result.spec.executed_by == "governance.officer"
        assert result.spec.provider_name == "mock"
        assert result.spec.model_name == "mock-gpt-v1"
