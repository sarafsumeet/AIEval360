"""
Agent evaluation domain models for AIEval360.

Defines the data structures for evaluating AI agents that:
- Use tools / function calls
- Pursue a goal across multiple steps
- May fall into loops or inefficient patterns

Key design decisions
--------------------
- AgentStep captures one discrete action (tool call, thought, or answer).
- AgentTrace is the full execution record for one agent run.
- AgentEvaluationResult holds all computed metrics for one trace.
- Timestamps are stored as float (Unix epoch) so traces from any
  source can be compared without timezone complexity.
- loop_detected is a boolean property so callers don't need to
  re-implement the cycle-detection logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AgentStepType(str, Enum):
    """The kind of action an agent took in a single step."""
    TOOL_CALL   = "tool_call"    # the agent invoked a tool / function
    TOOL_RESULT = "tool_result"  # the tool returned a result
    THOUGHT     = "thought"      # chain-of-thought / reasoning step
    ANSWER      = "answer"       # the agent produced its final answer
    ERROR       = "error"        # a step that failed with an exception


class GoalStatus(str, Enum):
    """Whether the agent achieved the stated goal."""
    ACHIEVED    = "achieved"
    PARTIAL     = "partial"
    NOT_ACHIEVED = "not_achieved"
    UNKNOWN     = "unknown"


# ---------------------------------------------------------------------------
# Step and Trace
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """
    A single step in an agent's execution trace.

    step_id      Unique identifier within the trace (e.g. "step_001").
    step_type    AgentStepType enum.
    tool_name    Name of the tool invoked (TOOL_CALL / TOOL_RESULT only).
    input        The input passed to the tool or the thought text.
    output       The output returned by the tool or the answer text.
    timestamp    Unix epoch float when this step started.
    duration_ms  How long the step took in milliseconds (None if unknown).
    error        Exception message if the step failed.
    metadata     Any extra context (model, temperature, tokens used, etc.).
    """

    step_id:     str
    step_type:   AgentStepType
    input:       Any = None
    output:      Any = None
    tool_name:   Optional[str] = None
    timestamp:   float = 0.0
    duration_ms: Optional[float] = None
    error:       Optional[str] = None
    metadata:    dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.step_id:
            raise ValueError("step_id must not be empty")

    @property
    def is_tool_call(self) -> bool:
        return self.step_type == AgentStepType.TOOL_CALL

    @property
    def is_error(self) -> bool:
        return self.step_type == AgentStepType.ERROR or self.error is not None


@dataclass
class AgentTrace:
    """
    Complete execution record for one agent run.

    trace_id      Unique identifier for this run.
    goal          Natural-language statement of what the agent was asked to do.
    steps         Ordered list of AgentStep objects.
    final_answer  The agent's final answer or output (may be None if failed).
    metadata      Run-level context (model, provider, session ID, etc.).
    """

    trace_id:     str
    goal:         str
    steps:        list[AgentStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    metadata:     dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id must not be empty")
        if not self.goal:
            raise ValueError("goal must not be empty")

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def tool_calls(self) -> list[AgentStep]:
        """All TOOL_CALL steps in order."""
        return [s for s in self.steps if s.step_type == AgentStepType.TOOL_CALL]

    @property
    def error_steps(self) -> list[AgentStep]:
        """All steps that raised an error."""
        return [s for s in self.steps if s.is_error]

    @property
    def unique_tools_used(self) -> set[str]:
        """Set of distinct tool names invoked."""
        return {s.tool_name for s in self.tool_calls if s.tool_name}

    @property
    def total_duration_ms(self) -> Optional[float]:
        """Sum of all step durations (None if any step is missing duration)."""
        durations = [s.duration_ms for s in self.steps]
        if any(d is None for d in durations):
            return None
        return sum(d for d in durations if d is not None)

    def steps_by_type(self, step_type: AgentStepType) -> list[AgentStep]:
        return [s for s in self.steps if s.step_type == step_type]


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class AgentEvaluationResult:
    """
    Computed evaluation metrics for one AgentTrace.

    trace_id            ID of the trace that was evaluated.
    goal_status         Whether the agent achieved its goal.
    tool_use_accuracy   Fraction of tool calls that succeeded (0.0–1.0).
    goal_completion     Numeric score of goal achievement (0.0–1.0).
    loop_detected       True if a repeated tool/input sequence was found.
    step_efficiency     1.0 when no redundant steps; <1.0 otherwise.
    error_rate          Fraction of steps that raised an error (0.0–1.0).
    total_steps         Total number of steps in the trace.
    total_tool_calls    Total number of TOOL_CALL steps.
    details             Breakdown dict for audit / governance reporting.
    """

    trace_id:          str
    goal_status:       GoalStatus = GoalStatus.UNKNOWN
    tool_use_accuracy: Optional[float] = None
    goal_completion:   float = 0.0
    loop_detected:     bool = False
    step_efficiency:   float = 1.0
    error_rate:        float = 0.0
    total_steps:       int = 0
    total_tool_calls:  int = 0
    details:           dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Convenience gate: achieved goal with no loop and no errors."""
        return (
            self.goal_status == GoalStatus.ACHIEVED
            and not self.loop_detected
            and self.error_rate == 0.0
        )

    @property
    def overall_score(self) -> float:
        """
        Simple composite score (0.0–1.0) for dashboard / trend charts.

        Components and weights:
          - goal_completion  : 50 %
          - step_efficiency  : 30 %
          - (1 – error_rate) : 20 %
        """
        return (
            0.5 * self.goal_completion
            + 0.3 * self.step_efficiency
            + 0.2 * (1.0 - self.error_rate)
        )
