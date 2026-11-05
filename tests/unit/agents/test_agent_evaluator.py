"""
Unit tests for the agent evaluator.

Covers:
- AgentStep / AgentTrace domain model helpers (via evaluator inputs)
- _normalise_input, _tool_use_accuracy, _loop_detected, _step_efficiency,
  _error_rate, _goal_completion_heuristic, _infer_goal_status (all internal helpers)
- AgentEvaluator.evaluate: overall_score, passed property, loop detection,
  goal status inference, batch evaluation
"""

from __future__ import annotations

import pytest

from aieval360.agents.evaluator import (
    AgentEvaluator,
    AgentEvaluatorConfig,
    _error_rate,
    _goal_completion_heuristic,
    _infer_goal_status,
    _loop_detected,
    _normalise_input,
    _step_efficiency,
    _tool_use_accuracy,
)
from aieval360.agents.models import (
    AgentEvaluationResult,
    AgentStep,
    AgentStepType,
    AgentTrace,
    GoalStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_step(
    step_id: str,
    step_type: AgentStepType = AgentStepType.TOOL_CALL,
    tool_name: str | None = "search",
    input: object = "query",
    error: str | None = None,
) -> AgentStep:
    return AgentStep(
        step_id=step_id,
        step_type=step_type,
        tool_name=tool_name,
        input=input,
        error=error,
    )


def make_trace(
    goal: str = "Find the capital of France.",
    steps: list[AgentStep] | None = None,
    final_answer: str | None = "The capital of France is Paris.",
    trace_id: str = "trace_001",
) -> AgentTrace:
    return AgentTrace(
        trace_id=trace_id,
        goal=goal,
        steps=steps or [],
        final_answer=final_answer,
    )


# ---------------------------------------------------------------------------
# _normalise_input
# ---------------------------------------------------------------------------

class TestNormaliseInput:
    def test_string_passthrough(self):
        result = _normalise_input("hello world")
        assert isinstance(result, str)

    def test_dict_sorted_keys(self):
        a = _normalise_input({"b": 2, "a": 1})
        b = _normalise_input({"a": 1, "b": 2})
        assert a == b

    def test_none_value(self):
        result = _normalise_input(None)
        assert result == "null"

    def test_list_value(self):
        result = _normalise_input([1, 2, 3])
        assert "1" in result

    def test_non_serialisable_falls_back_to_str(self):
        class Custom:
            pass
        result = _normalise_input(Custom())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _tool_use_accuracy
# ---------------------------------------------------------------------------

class TestToolUseAccuracy:
    def test_no_tool_calls_returns_none(self):
        trace = make_trace(steps=[])
        assert _tool_use_accuracy(trace) is None

    def test_all_succeeded(self):
        steps = [make_step(f"s{i}") for i in range(4)]
        trace = make_trace(steps=steps)
        assert _tool_use_accuracy(trace) == pytest.approx(1.0)

    def test_half_failed(self):
        ok = make_step("s1")
        err = make_step("s2", error="failed")
        trace = make_trace(steps=[ok, err])
        assert _tool_use_accuracy(trace) == pytest.approx(0.5)

    def test_all_failed(self):
        steps = [make_step(f"s{i}", error="fail") for i in range(3)]
        trace = make_trace(steps=steps)
        assert _tool_use_accuracy(trace) == pytest.approx(0.0)

    def test_ignores_non_tool_call_steps(self):
        thought = make_step("t1", step_type=AgentStepType.THOUGHT, tool_name=None)
        call = make_step("c1")
        trace = make_trace(steps=[thought, call])
        # Only 1 tool call, it succeeded → 1.0
        assert _tool_use_accuracy(trace) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _loop_detected
# ---------------------------------------------------------------------------

class TestLoopDetected:
    def test_no_loop_distinct_calls(self):
        steps = [
            make_step("s1", input="query A"),
            make_step("s2", input="query B"),
        ]
        found, details = _loop_detected(make_trace(steps=steps))
        assert found is False
        assert details["total_duplicate_pairs"] == 0

    def test_simple_loop_detected(self):
        steps = [
            make_step("s1", input="same"),
            make_step("s2", input="same"),
        ]
        found, details = _loop_detected(make_trace(steps=steps))
        assert found is True
        assert details["total_duplicate_pairs"] >= 1

    def test_different_tools_same_input_no_loop(self):
        s1 = make_step("s1", tool_name="search", input="q")
        s2 = make_step("s2", tool_name="fetch", input="q")
        found, _ = _loop_detected(make_trace(steps=[s1, s2]))
        assert found is False

    def test_same_tool_different_inputs_no_loop(self):
        s1 = make_step("s1", tool_name="search", input="q1")
        s2 = make_step("s2", tool_name="search", input="q2")
        found, _ = _loop_detected(make_trace(steps=[s1, s2]))
        assert found is False

    def test_dict_input_order_independent(self):
        s1 = make_step("s1", input={"b": 2, "a": 1})
        s2 = make_step("s2", input={"a": 1, "b": 2})
        found, _ = _loop_detected(make_trace(steps=[s1, s2]))
        assert found is True  # same normalised content

    def test_empty_trace_no_loop(self):
        found, details = _loop_detected(make_trace(steps=[]))
        assert found is False


# ---------------------------------------------------------------------------
# _step_efficiency
# ---------------------------------------------------------------------------

class TestStepEfficiency:
    def test_no_tool_calls_returns_one(self):
        trace = make_trace(steps=[])
        assert _step_efficiency(trace) == pytest.approx(1.0)

    def test_all_unique_calls_returns_one(self):
        steps = [make_step(f"s{i}", input=f"q{i}") for i in range(5)]
        trace = make_trace(steps=steps)
        assert _step_efficiency(trace) == pytest.approx(1.0)

    def test_half_duplicate_returns_half(self):
        steps = [
            make_step("s1", input="same"),
            make_step("s2", input="same"),
            make_step("s3", input="unique"),
            make_step("s4", input="unique2"),
        ]
        trace = make_trace(steps=steps)
        # 3 unique keys out of 4 calls
        assert _step_efficiency(trace) == pytest.approx(0.75)

    def test_all_duplicates_returns_low_value(self):
        steps = [make_step(f"s{i}", input="same") for i in range(4)]
        trace = make_trace(steps=steps)
        # 1 unique / 4 calls
        assert _step_efficiency(trace) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# _error_rate
# ---------------------------------------------------------------------------

class TestErrorRate:
    def test_no_steps_returns_zero(self):
        trace = make_trace(steps=[])
        assert _error_rate(trace) == pytest.approx(0.0)

    def test_no_errors_returns_zero(self):
        steps = [make_step(f"s{i}") for i in range(3)]
        trace = make_trace(steps=steps)
        assert _error_rate(trace) == pytest.approx(0.0)

    def test_half_errors(self):
        ok = make_step("ok")
        err = make_step("err", step_type=AgentStepType.ERROR, tool_name=None)
        trace = make_trace(steps=[ok, err])
        assert _error_rate(trace) == pytest.approx(0.5)

    def test_step_with_error_field_counts(self):
        err_step = make_step("e1", error="something broke")
        trace = make_trace(steps=[err_step])
        assert _error_rate(trace) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _goal_completion_heuristic
# ---------------------------------------------------------------------------

class TestGoalCompletionHeuristic:
    def test_no_answer_returns_zero(self):
        assert _goal_completion_heuristic("find capital of France", None) == pytest.approx(0.0)

    def test_exact_match(self):
        goal = "find capital France"
        answer = "find capital France"
        score = _goal_completion_heuristic(goal, answer)
        assert score == pytest.approx(1.0)

    def test_partial_match(self):
        goal = "find the capital of France"
        answer = "The capital is Paris."
        score = _goal_completion_heuristic(goal, answer)
        assert 0.0 < score < 1.0

    def test_no_overlap_returns_low(self):
        goal = "compute derivative integral"
        answer = "The weather is sunny today."
        score = _goal_completion_heuristic(goal, answer)
        assert score < 0.3

    def test_empty_answer_returns_zero(self):
        assert _goal_completion_heuristic("some goal", "") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _infer_goal_status
# ---------------------------------------------------------------------------

class TestInferGoalStatus:
    def test_high_completion_no_loop_achieved(self):
        assert _infer_goal_status(0.9, False) == GoalStatus.ACHIEVED

    def test_high_completion_with_loop_partial(self):
        assert _infer_goal_status(0.9, True) == GoalStatus.PARTIAL

    def test_mid_completion_partial(self):
        assert _infer_goal_status(0.5, False) == GoalStatus.PARTIAL

    def test_mid_completion_with_loop_partial(self):
        assert _infer_goal_status(0.5, True) == GoalStatus.PARTIAL

    def test_low_completion_not_achieved(self):
        assert _infer_goal_status(0.2, False) == GoalStatus.NOT_ACHIEVED

    def test_exactly_at_boundary_achieved(self):
        assert _infer_goal_status(0.8, False) == GoalStatus.ACHIEVED

    def test_exactly_at_partial_boundary(self):
        assert _infer_goal_status(0.4, False) == GoalStatus.PARTIAL


# ---------------------------------------------------------------------------
# AgentEvaluator.evaluate
# ---------------------------------------------------------------------------

class TestAgentEvaluatorEvaluate:
    def test_basic_success_trace(self):
        steps = [make_step("s1", input="capital of France")]
        trace = make_trace(
            goal="Find capital of France",
            steps=steps,
            final_answer="The capital of France is Paris.",
        )
        evaluator = AgentEvaluator()
        result = evaluator.evaluate(trace)
        assert isinstance(result, AgentEvaluationResult)
        assert result.trace_id == "trace_001"

    def test_no_tool_calls_tua_is_none(self):
        trace = make_trace(steps=[], final_answer="Paris.")
        evaluator = AgentEvaluator()
        result = evaluator.evaluate(trace)
        assert result.tool_use_accuracy is None

    def test_loop_detected_in_result(self):
        steps = [
            make_step("s1", input="same query"),
            make_step("s2", input="same query"),
        ]
        trace = make_trace(steps=steps)
        evaluator = AgentEvaluator()
        result = evaluator.evaluate(trace)
        assert result.loop_detected is True

    def test_no_loop_in_clean_trace(self):
        steps = [
            make_step("s1", input="query one"),
            make_step("s2", input="query two"),
        ]
        trace = make_trace(steps=steps)
        evaluator = AgentEvaluator()
        result = evaluator.evaluate(trace)
        assert result.loop_detected is False

    def test_overall_score_range(self):
        trace = make_trace()
        result = AgentEvaluator().evaluate(trace)
        assert 0.0 <= result.overall_score <= 1.0

    def test_passed_true_for_achieved_no_loop_no_error(self):
        # Need high goal completion (≥0.8) and no loop and no error
        trace = make_trace(
            goal="capital France",
            final_answer="capital France",  # exact match → 1.0
            steps=[make_step("s1", input="q")],
        )
        result = AgentEvaluator().evaluate(trace)
        assert result.goal_status == GoalStatus.ACHIEVED
        assert result.passed is True

    def test_passed_false_with_loop(self):
        steps = [make_step("s1", input="x"), make_step("s2", input="x")]
        trace = make_trace(
            goal="capital France",
            final_answer="capital France",
            steps=steps,
        )
        result = AgentEvaluator().evaluate(trace)
        assert result.loop_detected is True
        assert result.passed is False

    def test_overall_score_formula(self):
        result = AgentEvaluationResult(
            trace_id="x",
            goal_completion=1.0,
            step_efficiency=1.0,
            error_rate=0.0,
            goal_status=GoalStatus.ACHIEVED,
        )
        assert result.overall_score == pytest.approx(0.5 + 0.3 + 0.2)

    def test_custom_goal_completion_fn(self):
        config = AgentEvaluatorConfig(goal_completion_fn=lambda g, a: 0.99)
        trace = make_trace()
        result = AgentEvaluator(config).evaluate(trace)
        assert result.goal_completion == pytest.approx(0.99)

    def test_goal_completion_clamped_above_one(self):
        config = AgentEvaluatorConfig(goal_completion_fn=lambda g, a: 5.0)
        trace = make_trace()
        result = AgentEvaluator(config).evaluate(trace)
        assert result.goal_completion == pytest.approx(1.0)

    def test_goal_completion_clamped_below_zero(self):
        config = AgentEvaluatorConfig(goal_completion_fn=lambda g, a: -2.0)
        trace = make_trace()
        result = AgentEvaluator(config).evaluate(trace)
        assert result.goal_completion == pytest.approx(0.0)

    def test_details_dict_populated(self):
        trace = make_trace(steps=[make_step("s1")])
        result = AgentEvaluator().evaluate(trace)
        assert "total_steps" in result.details
        assert "total_tool_calls" in result.details
        assert "unique_tools" in result.details

    def test_total_steps_count(self):
        steps = [make_step(f"s{i}") for i in range(6)]
        trace = make_trace(steps=steps)
        result = AgentEvaluator().evaluate(trace)
        assert result.total_steps == 6

    def test_total_tool_calls_count(self):
        tool_steps = [make_step(f"tc{i}") for i in range(3)]
        answer_step = make_step("a1", step_type=AgentStepType.ANSWER, tool_name=None)
        trace = make_trace(steps=tool_steps + [answer_step])
        result = AgentEvaluator().evaluate(trace)
        assert result.total_tool_calls == 3


# ---------------------------------------------------------------------------
# AgentEvaluator.evaluate_many
# ---------------------------------------------------------------------------

class TestEvaluateMany:
    def test_returns_one_result_per_trace(self):
        traces = [make_trace(trace_id=f"t{i}") for i in range(5)]
        results = AgentEvaluator().evaluate_many(traces)
        assert len(results) == 5

    def test_trace_ids_match(self):
        traces = [make_trace(trace_id=f"trace_{i}") for i in range(3)]
        results = AgentEvaluator().evaluate_many(traces)
        ids = [r.trace_id for r in results]
        assert ids == ["trace_0", "trace_1", "trace_2"]

    def test_empty_list_returns_empty(self):
        results = AgentEvaluator().evaluate_many([])
        assert results == []
