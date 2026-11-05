"""
Generation quality metrics for RAG evaluation.

These metrics assess how well the LLM answer uses the retrieved context.

Metric taxonomy
---------------
HEURISTIC   — no LLM call, computable from text overlap
LLM_JUDGED  — consume a pre-computed judge_response (JSON blob)
              The orchestrator is responsible for the judge LLM call;
              these calculators just parse the response.

Implemented
-----------
- AnswerRelevanceHeuristic   HEURISTIC  — word-overlap between query and answer
- AnswerFaithfulnessScaffold LLM_JUDGED — is the answer grounded in context?
- ContextUtilisationHeuristic HEURISTIC — fraction of retrieved docs cited/used
- AnswerCompletenessScaffold  LLM_JUDGED — does the answer cover the question fully?
"""

from __future__ import annotations

import json
import re
from typing import Optional

from aieval360.evaluation.rag.models import RAGInput, RAGMetricOutput


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_output(
    metric_name: str,
    value: Optional[float],
    threshold: Optional[float] = None,
    details: Optional[dict] = None,
    reasoning: Optional[str] = None,
    error: Optional[str] = None,
) -> RAGMetricOutput:
    return RAGMetricOutput(
        metric_name=metric_name,
        value=value,
        threshold=threshold,
        details=details or {},
        reasoning=reasoning,
        error=error,
    )


def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _parse_judge(judge_response: Optional[str], metric_name: str) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Parse a judge JSON blob.

    Expected format: {"score": 0.85, "reasoning": "...", "passed": true}
    Returns (value, reasoning, error).
    """
    if not judge_response:
        return None, None, f"{metric_name}: judge_response is empty"
    try:
        data = json.loads(judge_response)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, None, f"{metric_name}: malformed judge JSON — {exc}"
    score = data.get("score")
    if score is None:
        return None, None, f"{metric_name}: judge JSON missing 'score' field"
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None, None, f"{metric_name}: 'score' is not numeric"
    reasoning = data.get("reasoning", "")
    return value, reasoning, None


# ---------------------------------------------------------------------------
# AnswerRelevanceHeuristic  (HEURISTIC)
# ---------------------------------------------------------------------------

def answer_relevance_heuristic(rag_input: RAGInput, threshold: Optional[float] = None) -> RAGMetricOutput:
    """
    Heuristic answer relevance: Jaccard overlap between query tokens and answer tokens.

    Score = |query_tokens ∩ answer_tokens| / |query_tokens ∪ answer_tokens|

    Quick proxy for relevance without an LLM judge call.  Should be used
    alongside an LLM-judged metric for governance-grade assessments.
    """
    if not rag_input.query:
        return _make_output("answer_relevance_heuristic", None, threshold, error="query is empty")
    if not rag_input.generated_answer:
        return _make_output("answer_relevance_heuristic", 0.0, threshold,
                            details={"query_tokens": 0, "answer_tokens": 0})

    q_tokens = _tokenise(rag_input.query)
    a_tokens = _tokenise(rag_input.generated_answer)
    union = q_tokens | a_tokens
    if not union:
        return _make_output("answer_relevance_heuristic", 0.0, threshold)

    intersection = q_tokens & a_tokens
    value = len(intersection) / len(union)
    return _make_output(
        "answer_relevance_heuristic", value, threshold,
        details={
            "query_tokens": len(q_tokens),
            "answer_tokens": len(a_tokens),
            "intersection": len(intersection),
            "union": len(union),
        },
    )


# ---------------------------------------------------------------------------
# AnswerFaithfulnessScaffold  (LLM_JUDGED)
# ---------------------------------------------------------------------------

def answer_faithfulness_scaffold(rag_input: RAGInput, threshold: Optional[float] = None) -> RAGMetricOutput:
    """
    LLM-judged faithfulness: is every claim in the answer supported by the
    retrieved context?

    Consumes a pre-computed judge_response JSON:
        {"score": 0.85, "reasoning": "All claims trace to doc-3 and doc-7."}

    The orchestrator is responsible for calling the judge LLM and storing
    its response in RAGInput.judge_response before invoking this function.
    This keeps the metric stateless and independently testable.
    """
    value, reasoning, error = _parse_judge(rag_input.judge_response, "answer_faithfulness")
    if error:
        return _make_output("answer_faithfulness", None, threshold, error=error)
    return _make_output(
        "answer_faithfulness", value, threshold,
        details={"judge_response_length": len(rag_input.judge_response or "")},
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# ContextUtilisationHeuristic  (HEURISTIC)
# ---------------------------------------------------------------------------

def context_utilisation_heuristic(rag_input: RAGInput, threshold: Optional[float] = None) -> RAGMetricOutput:
    """
    Heuristic context utilisation: fraction of retrieved doc content words
    that appear in the generated answer.

    Score = |context_tokens ∩ answer_tokens| / |context_tokens|

    Measures whether the LLM actually drew on the retrieved context or
    ignored it and hallucinated.  A low score despite high context recall
    is a red flag for hallucination.
    """
    if not rag_input.retrieved_docs:
        return _make_output("context_utilisation_heuristic", None, threshold, error="no retrieved_docs")
    if not rag_input.generated_answer:
        return _make_output("context_utilisation_heuristic", 0.0, threshold,
                            details={"context_tokens": 0, "answer_tokens": 0})

    context_text = " ".join(d.content for d in rag_input.retrieved_docs)
    ctx_tokens = _tokenise(context_text)
    ans_tokens = _tokenise(rag_input.generated_answer)

    if not ctx_tokens:
        return _make_output("context_utilisation_heuristic", 0.0, threshold,
                            details={"context_tokens": 0})

    overlap = ctx_tokens & ans_tokens
    value = len(overlap) / len(ctx_tokens)
    return _make_output(
        "context_utilisation_heuristic", value, threshold,
        details={
            "context_tokens": len(ctx_tokens),
            "answer_tokens": len(ans_tokens),
            "overlap_tokens": len(overlap),
            "num_docs": len(rag_input.retrieved_docs),
        },
    )


# ---------------------------------------------------------------------------
# AnswerCompletenessScaffold  (LLM_JUDGED)
# ---------------------------------------------------------------------------

def answer_completeness_scaffold(rag_input: RAGInput, threshold: Optional[float] = None) -> RAGMetricOutput:
    """
    LLM-judged completeness: does the answer fully address all aspects of
    the query, given the available context?

    Consumes a pre-computed judge_response JSON:
        {"score": 0.70, "reasoning": "Answer omits the regulatory section."}
    """
    value, reasoning, error = _parse_judge(rag_input.judge_response, "answer_completeness")
    if error:
        return _make_output("answer_completeness", None, threshold, error=error)
    return _make_output(
        "answer_completeness", value, threshold,
        details={"judge_response_length": len(rag_input.judge_response or "")},
        reasoning=reasoning,
    )
