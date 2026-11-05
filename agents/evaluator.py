"""
Agent evaluator for AIEval360.

Analyses an AgentTrace and computes five evaluation metrics:

  1. tool_use_accuracy   — fraction of tool calls that succeeded
  2. goal_completion     — numeric score for goal achievement (heuristic)
  3. loop_detected       — boolean; True if the agent repeated itself
  4. step_efficiency     — ratio of unique / total steps (1.0 = no redundancy)
  5. error_rate          — fraction of steps that errored

Design decisions
----------------
- Pure synchronous evaluation: traces are already captured, so async
  is not needed here.
- Goal completion uses a simple keyword-overlap heuristic between the
  goal statement and the final_answer.  Callers who have an LLM judge
  can override by passing goal_completion_fn to the config.
- Loop detection compares (tool_name, normalised_input) tuples; a pair
  that appears more than once flags a loop.
- All logic is documented so it can be reproduced in a governance audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from aieval360.agents.models import (
    AgentEvaluationResult,
    AgentStep,
    AgentStepType,
    AgentTrace,
    GoalStatus,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_input(value: Any) -> str:
    """
    Convert a step input to a stable string for loop detection.

    Uses JSON serialisation when possible so that dict / list inputs
    are compared structurally rather than by identity.
    """
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _tool_use_accuracy(trace: AgentTrace) -> Optional[float]:
    """
    Fraction of TOOL_CALL steps that completed without error.

    Returns None if the trace has no tool calls (metric is not applicable).
    """
    calls = trace.tool_calls
    if not calls:
        return None
    succeeded = sum(1 for s in calls if not s.is_error)
    return succeeded / len(calls)


def _loop_detected(trace: AgentTrace) -> tuple[bool, dict[str, Any]]:
    """
    Detect repeated (tool_name, input) pairs in TOOL_CALL steps.

    Returns (loop_found, details_dict).
    """
    seen: dict[str, int] = {}
    duplicates: list[dict[str, Any]] = []

    for step in trace.tool_calls:
        key = f"{step.tool_name}::{_normalise_input(step.input)}"
        # Use an 8-char hash so the key fits neatly in a report
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:8]
        seen[key_hash] = seen.get(key_hash, 0) + 1
        if seen[key_hash] == 2:
            duplicates.append({"tool": step.tool_name, "step_id": step.step_id})

    found = len(duplicates) > 0
    return found, {"repeated_calls": duplicates, "total_duplicate_pairs": len(duplicates)}


def _step_efficiency(trace: AgentTrace) -> float:
    """
    Ratio of unique tool+input pairs to total tool calls.

    1.0 means every tool call was unique (no redundancy).
    <1.0 means the agent repeated at least one call.
    0.0 is never returned here (division by zero is guarded).
    """
    calls = trace.tool_calls
    if not calls:
        return 1.0
    keys = {f"{s.tool_name}::{_normalise_input(s.input)}" for s in calls}
    return len(keys) / len(calls)


def _error_rate(trace: AgentTrace) -> float:
    """
    Fraction of all steps that raised an error.

    Uses the step's is_error property (covers both ERROR type and
    steps with a non-None error field).
    """
    if not trace.steps:
        return 0.0
    errors = sum(1 for s in trace.steps if s.is_error)
    return errors / len(trace.steps)


def _goal_completion_heuristic(goal: str, final_answer: Optional[str]) -> float:
    """
    Simple Jaccard-overlap score between goal tokens and answer tokens.

    Returns 0.0 if there is no final_answer.
    Returns 1.0 if the answer contains every word from the goal.
    """
    if not final_answer:
        return 0.0
    goal_tokens = set(goal.lower().split())
    answer_tokens = set(final_answer.lower().split())
    if not goal_tokens:
        return 0.0
    overlap = goal_tokens & answer_tokens
    return len(overlap) / len(goal_tokens)


def _infer_goal_status(completion: float, loop: bool) -> GoalStatus:
    """
    Map a numeric completion score to a GoalStatus enum value.

    Thresholds (conservative by design):
      >= 0.8 and no loop  → ACHIEVED
      >= 0.4              → PARTIAL
      else                → NOT_ACHIEVED
    """
    if completion >= 0.8 and not loop:
        return GoalStatus.ACHIEVED
    if completion >= 0.4:
        return GoalStatus.PARTIAL
    return GoalStatus.NOT_ACHIEVED


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentEvaluatorConfig:
    """
    Configuration for AgentEvaluator.

    goal_completion_fn   Optional async-or-sync callable
                         ``(goal: str, final_answer: str | None) -> float``
                         If None, the heuristic Jaccard scorer is used.
    metadata             Arbitrary key-value context stored in the result.
    """

    goal_completion_fn: Optional[Callable[[str, Optional[str]], float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AgentEvaluator
# ---------------------------------------------------------------------------

class AgentEvaluator:
    """
    Analyses an AgentTrace and returns an AgentEvaluationResult.

    Usage::

        trace = AgentTrace(
            trace_id="run_001",
            goal="Find the capital of France and return it.",
            steps=[...],
            final_answer="The capital of France is Paris.",
        )
        evaluator = AgentEvaluator()
        result = evaluator.evaluate(trace)
        print(result.overall_score)
    """

    def __init__(self, config: Optional[AgentEvaluatorConfig] = None) -> None:
        self.config = config or AgentEvaluatorConfig()

    def evaluate(self, trace: AgentTrace) -> AgentEvaluationResult:
        """
        Run all five metrics on the trace and return a result object.

        Parameters
        ----------
        trace : AgentTrace

        Returns
        -------
        AgentEvaluationResult
        """
        cfg = self.config

        # 1. Tool-use accuracy
        tua = _tool_use_accuracy(trace)

        # 2. Goal completion
        if cfg.goal_completion_fn is not None:
            gc = cfg.goal_completion_fn(trace.goal, trace.final_answer)
        else:
            gc = _goal_completion_heuristic(trace.goal, trace.final_answer)
        gc = float(max(0.0, min(1.0, gc)))  # clamp to [0, 1]

        # 3. Loop detection
        loop, loop_details = _loop_detected(trace)

        # 4. Step efficiency
        efficiency = _step_efficiency(trace)

        # 5. Error rate
        err_rate = _error_rate(trace)

        # 6. Infer goal status
        goal_status = _infer_goal_status(gc, loop)

        details: dict[str, Any] = {
            "tool_use_accuracy": tua,
            "goal_completion": gc,
            "loop_details": loop_details,
            "step_efficiency": efficiency,
            "error_rate": err_rate,
            "total_steps": len(trace.steps),
            "total_tool_calls": len(trace.tool_calls),
            "unique_tools": sorted(trace.unique_tools_used),
            "error_step_ids": [s.step_id for s in trace.error_steps],
            **cfg.metadata,
        }

        return AgentEvaluationResult(
            trace_id=trace.trace_id,
            goal_status=goal_status,
            tool_use_accuracy=tua,
            goal_completion=gc,
            loop_detected=loop,
            step_efficiency=efficiency,
            error_rate=err_rate,
            total_steps=len(trace.steps),
            total_tool_calls=len(trace.tool_calls),
            details=details,
        )

    def evaluate_many(self, traces: list[AgentTrace]) -> list[AgentEvaluationResult]:
        """Evaluate a batch of traces and return one result per trace."""
        return [self.evaluate(t) for t in traces]
