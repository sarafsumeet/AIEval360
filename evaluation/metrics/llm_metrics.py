"""
LLM evaluation metrics: BLEU, ROUGE, heuristic quality, and faithfulness scaffold.

MetricType breakdown:
  REFERENCE_BASED: BLEU, ROUGE (need ground-truth reference texts)
  HEURISTIC:       response_length, non_empty, lexical_diversity
  LLM_JUDGED:      faithfulness_scaffold (placeholder — real scorer calls judge LLM)

None of these make external network calls; LLM_JUDGED metrics consume
a pre-computed judge_response from MetricInput.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from aieval360.core.models.evaluation import MetricType

from .base import MetricCalculator, MetricInput, MetricOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Simple whitespace+punctuation tokeniser."""
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------------------
# BLEU (reference-based)
# ---------------------------------------------------------------------------

class BleuCalculator(MetricCalculator):
    """
    Corpus-level BLEU-4 (modified precision, brevity penalty).

    Uses sacrebleu when available; falls back to a minimal pure-Python
    implementation that is correct for single-reference evaluation.
    """

    @property
    def metric_name(self) -> str:
        return "bleu"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.REFERENCE_BASED

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        hyps = inputs.predictions if inputs.predictions else (
            [inputs.generated_text] if inputs.generated_text else []
        )
        refs = inputs.references if inputs.references else (
            [inputs.reference_text] if inputs.reference_text else []
        )
        if not hyps or not refs:
            return self._make_output(None, threshold, error="predictions and references required for BLEU")
        if len(hyps) != len(refs):
            return self._make_output(None, threshold, error="predictions and references must be same length")

        try:
            import sacrebleu  # type: ignore
            result = sacrebleu.corpus_bleu(hyps, [refs])
            value = result.score / 100.0  # sacrebleu returns 0-100
            return self._make_output(value, threshold, details={"sacrebleu_score": result.score})
        except ImportError:
            pass

        # Pure-Python fallback: 4-gram BLEU
        value = self._pure_bleu(hyps, refs)
        return self._make_output(value, threshold)

    @staticmethod
    def _pure_bleu(hyps: list[str], refs: list[str], max_n: int = 4) -> float:
        import math

        total_hyp_len = 0
        total_ref_len = 0
        clipped_counts = [0] * max_n
        total_counts = [0] * max_n

        for hyp, ref in zip(hyps, refs):
            hyp_tokens = _tokenise(str(hyp))
            ref_tokens = _tokenise(str(ref))
            total_hyp_len += len(hyp_tokens)
            total_ref_len += len(ref_tokens)
            for n in range(1, max_n + 1):
                hyp_ngrams = _ngrams(hyp_tokens, n)
                ref_ngrams = _ngrams(ref_tokens, n)
                clipped = {ng: min(cnt, ref_ngrams[ng]) for ng, cnt in hyp_ngrams.items()}
                clipped_counts[n - 1] += sum(clipped.values())
                total_counts[n - 1] += sum(hyp_ngrams.values())

        precisions = []
        for c, t in zip(clipped_counts, total_counts):
            if t == 0:
                precisions.append(0.0)
            else:
                precisions.append(c / t)

        if any(p == 0 for p in precisions):
            return 0.0

        log_avg = sum(math.log(p) for p in precisions) / max_n
        bp = min(1.0, math.exp(1 - total_ref_len / total_hyp_len)) if total_hyp_len > 0 else 0.0
        return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# ROUGE (reference-based)
# ---------------------------------------------------------------------------

class RougeCalculator(MetricCalculator):
    """
    ROUGE-1, ROUGE-2, ROUGE-L (F1 scores).

    Returns rouge_1 as the primary value; full breakdown in details.
    Uses rouge-score when available; pure-Python fallback otherwise.
    """

    @property
    def metric_name(self) -> str:
        return "rouge"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.REFERENCE_BASED

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        hyps = inputs.predictions if inputs.predictions else (
            [inputs.generated_text] if inputs.generated_text else []
        )
        refs = inputs.references if inputs.references else (
            [inputs.reference_text] if inputs.reference_text else []
        )
        if not hyps or not refs or len(hyps) != len(refs):
            return self._make_output(None, threshold, error="predictions and references required for ROUGE")

        try:
            from rouge_score import rouge_scorer  # type: ignore
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
            agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            for h, r in zip(hyps, refs):
                scores = scorer.score(str(r), str(h))
                agg["rouge1"] += scores["rouge1"].fmeasure
                agg["rouge2"] += scores["rouge2"].fmeasure
                agg["rougeL"] += scores["rougeL"].fmeasure
            n = len(hyps)
            details = {k: v / n for k, v in agg.items()}
            return self._make_output(details["rouge1"], threshold, details=details)
        except ImportError:
            pass

        # Pure-Python fallback: ROUGE-1 only
        r1_scores = []
        for h, r in zip(hyps, refs):
            h_toks = set(_tokenise(str(h)))
            r_toks = set(_tokenise(str(r)))
            overlap = h_toks & r_toks
            prec = len(overlap) / len(h_toks) if h_toks else 0.0
            rec = len(overlap) / len(r_toks) if r_toks else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            r1_scores.append(f1)
        value = sum(r1_scores) / len(r1_scores)
        return self._make_output(value, threshold, details={"rouge1": value, "note": "pure-Python fallback"})


# ---------------------------------------------------------------------------
# Heuristic quality metrics
# ---------------------------------------------------------------------------

class ResponseLengthCalculator(MetricCalculator):
    """
    Token count of generated text, normalised to [0, 1] by a max_tokens cap.

    Not a quality signal by itself, but useful as a gate:
    "response must be at least N tokens" or "must not exceed M tokens".
    primary value = raw token count (as float for consistency)
    """

    def __init__(self, max_tokens: int = 2000) -> None:
        self.max_tokens = max_tokens

    @property
    def metric_name(self) -> str:
        return "response_length"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.HEURISTIC

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        text = inputs.generated_text or (inputs.predictions[0] if inputs.predictions else "")
        if not text:
            return self._make_output(0.0, threshold, details={"token_count": 0})
        tokens = _tokenise(str(text))
        count = len(tokens)
        return self._make_output(
            float(count), threshold,
            details={"token_count": count, "max_tokens": self.max_tokens},
        )


class NonEmptyResponseCalculator(MetricCalculator):
    """
    Binary check: 1.0 if generated text is non-empty, 0.0 otherwise.
    Useful as a mandatory gate to catch truncated or failed responses.
    """

    @property
    def metric_name(self) -> str:
        return "non_empty_response"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.HEURISTIC

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        text = inputs.generated_text or (inputs.predictions[0] if inputs.predictions else "")
        value = 1.0 if str(text).strip() else 0.0
        return self._make_output(value, threshold)


class LexicalDiversityCalculator(MetricCalculator):
    """
    Type-Token Ratio (TTR): unique tokens / total tokens.
    Range [0, 1]; higher = more diverse vocabulary.
    """

    @property
    def metric_name(self) -> str:
        return "lexical_diversity"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.HEURISTIC

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        text = inputs.generated_text or (inputs.predictions[0] if inputs.predictions else "")
        tokens = _tokenise(str(text))
        if not tokens:
            return self._make_output(0.0, threshold, details={"unique": 0, "total": 0})
        value = len(set(tokens)) / len(tokens)
        return self._make_output(
            value, threshold,
            details={"unique": len(set(tokens)), "total": len(tokens)},
        )


# ---------------------------------------------------------------------------
# LLM-judged metrics (scaffold — judge call is external)
# ---------------------------------------------------------------------------

class FaithfulnessScaffoldCalculator(MetricCalculator):
    """
    Faithfulness of a generated answer to a source context.

    This calculator *consumes* a pre-computed judge response (JSON string)
    rather than calling the judge LLM itself.  The evaluation runner is
    responsible for calling the judge provider and storing the response in
    MetricInput.judge_response before invoking this calculator.

    Expected judge_response JSON format::

        {"score": 0.85, "reasoning": "...", "passed": true}

    Returns MetricType.LLM_JUDGED so downstream evidence records clearly
    distinguish AI-judged scores from deterministic ones.
    """

    @property
    def metric_name(self) -> str:
        return "faithfulness"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.LLM_JUDGED

    def calculate(self, inputs: MetricInput, threshold: Optional[float] = None) -> MetricOutput:
        import json

        raw = inputs.judge_response
        if not raw:
            return self._make_output(
                None, threshold,
                error="judge_response required for faithfulness; populate via LLM judge before calling",
            )
        try:
            parsed = json.loads(str(raw))
            score = float(parsed["score"])
            reasoning = parsed.get("reasoning", "")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return self._make_output(None, threshold, error=f"Could not parse judge_response: {exc}")
        return self._make_output(score, threshold, reasoning=reasoning, details={"raw": raw})


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def build_llm_registry() -> dict[str, MetricCalculator]:
    """Return a dict mapping metric name → calculator for all LLM metrics."""
    calculators: list[MetricCalculator] = [
        BleuCalculator(),
        RougeCalculator(),
        ResponseLengthCalculator(),
        NonEmptyResponseCalculator(),
        LexicalDiversityCalculator(),
        FaithfulnessScaffoldCalculator(),
    ]
    return {c.metric_name: c for c in calculators}
