"""
Unit tests for RAGEvaluationPipeline and RAGMetricRegistry.

Covers:
- Pipeline runs with retrieval-only, generation-only, and mixed configs
- Aggregation (mean across examples)
- LLM-judge integration via mock judge_fn
- Error handling for unknown metrics and failing functions
- Empty example list behaviour
- RAGMetricRegistry: register, get, list, contains, by_type, by_tag
- RAGPipelineConfig defaults
"""

from __future__ import annotations

import json
import pytest

from aieval360.evaluation.rag.models import RAGInput, RetrievedDocument
from aieval360.evaluation.rag.pipeline import RAGEvaluationPipeline, RAGPipelineConfig
from aieval360.evaluation.rag.registry import (
    RAGMetricRegistry,
    RAGMetricRegistration,
    build_default_rag_registry,
    default_rag_registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc(doc_id: str, content: str = "", rank: int = 1) -> RetrievedDocument:
    return RetrievedDocument(doc_id=doc_id, content=content, rank=rank)


def make_example(
    retrieved: list[tuple[str, str]],  # (doc_id, content)
    relevant: set[str],
    query: str = "test query",
    answer: str = "test answer",
    judge_response: str | None = None,
) -> RAGInput:
    docs = [
        RetrievedDocument(doc_id=d, content=c, rank=i + 1)
        for i, (d, c) in enumerate(retrieved)
    ]
    return RAGInput(
        query=query,
        retrieved_docs=docs,
        generated_answer=answer,
        relevant_doc_ids=relevant,
        judge_response=judge_response,
    )


async def mock_judge(query: str, context: str, answer: str) -> str:
    return json.dumps({"score": 0.8, "reasoning": "mock judge response"})


async def failing_judge(query: str, context: str, answer: str) -> str:
    raise RuntimeError("judge LLM unavailable")


# ---------------------------------------------------------------------------
# RAGPipelineConfig
# ---------------------------------------------------------------------------

class TestRAGPipelineConfig:
    def test_default_retrieval_metrics(self):
        cfg = RAGPipelineConfig()
        assert "precision_at_k" in cfg.retrieval_metrics
        assert "mrr" in cfg.retrieval_metrics
        assert len(cfg.retrieval_metrics) == 7

    def test_default_generation_metrics(self):
        cfg = RAGPipelineConfig()
        assert "answer_relevance_heuristic" in cfg.generation_metrics
        assert len(cfg.generation_metrics) == 4

    def test_default_k(self):
        cfg = RAGPipelineConfig()
        assert cfg.retrieval_k == 5

    def test_no_judge_fn_by_default(self):
        cfg = RAGPipelineConfig()
        assert cfg.judge_fn is None

    def test_custom_thresholds(self):
        cfg = RAGPipelineConfig(thresholds={"precision_at_k": 0.8})
        assert cfg.thresholds["precision_at_k"] == 0.8


# ---------------------------------------------------------------------------
# RAGEvaluationPipeline — retrieval only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPipelineRetrievalOnly:
    async def test_single_example_all_relevant(self):
        example = make_example([("a", ""), ("b", ""), ("c", "")], {"a", "b", "c"})
        cfg = RAGPipelineConfig(
            retrieval_metrics=["precision_at_k", "recall_at_k"],
            generation_metrics=[],
            retrieval_k=3,
        )
        pipeline = RAGEvaluationPipeline(cfg)
        result = await pipeline.run([example])
        assert result.total_examples == 1
        assert len(result.metric_outputs) == 2
        pk = next(m for m in result.metric_outputs if m.metric_name == "precision_at_k")
        assert pk.value == pytest.approx(1.0)

    async def test_multiple_examples_aggregated(self):
        # Example 1: 1/2 relevant; Example 2: 2/2 relevant
        ex1 = make_example([("a", ""), ("x", "")], {"a"})
        ex2 = make_example([("b", ""), ("c", "")], {"b", "c"})
        cfg = RAGPipelineConfig(
            retrieval_metrics=["precision_at_k"],
            generation_metrics=[],
            retrieval_k=2,
        )
        pipeline = RAGEvaluationPipeline(cfg)
        result = await pipeline.run([ex1, ex2])
        pk = next(m for m in result.metric_outputs if m.metric_name == "precision_at_k")
        # Ex1: 1/2=0.5, Ex2: 2/2=1.0, mean = 0.75
        assert pk.value == pytest.approx(0.75)

    async def test_mrr_no_k_argument(self):
        example = make_example([("x", ""), ("a", "")], {"a"})
        cfg = RAGPipelineConfig(
            retrieval_metrics=["mrr"],
            generation_metrics=[],
        )
        pipeline = RAGEvaluationPipeline(cfg)
        result = await pipeline.run([example])
        mrr = next(m for m in result.metric_outputs if m.metric_name == "mrr")
        assert mrr.value == pytest.approx(0.5)

    async def test_elapsed_time_positive(self):
        example = make_example([("a", "")], {"a"})
        cfg = RAGPipelineConfig(retrieval_metrics=["hit_at_k"], generation_metrics=[])
        result = await RAGEvaluationPipeline(cfg).run([example])
        assert result.elapsed_seconds >= 0.0

    async def test_per_example_scores_recorded(self):
        example = make_example([("a", "")], {"a"})
        cfg = RAGPipelineConfig(retrieval_metrics=["hit_at_k"], generation_metrics=[])
        result = await RAGEvaluationPipeline(cfg).run([example])
        assert len(result.per_example) == 1
        assert "hit_at_k" in result.per_example[0]["scores"]

    async def test_threshold_applied_to_aggregated(self):
        example = make_example([("a", ""), ("x", "")], {"a"})
        cfg = RAGPipelineConfig(
            retrieval_metrics=["precision_at_k"],
            generation_metrics=[],
            retrieval_k=2,
            thresholds={"precision_at_k": 0.4},
        )
        result = await RAGEvaluationPipeline(cfg).run([example])
        pk = next(m for m in result.metric_outputs if m.metric_name == "precision_at_k")
        assert pk.passed is True


# ---------------------------------------------------------------------------
# RAGEvaluationPipeline — generation only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPipelineGenerationOnly:
    async def test_heuristic_metrics_no_judge(self):
        example = make_example(
            [("d1", "machine learning is a subfield of AI")],
            relevant=set(),
            query="what is machine learning",
            answer="machine learning helps AI systems learn from data",
        )
        cfg = RAGPipelineConfig(
            retrieval_metrics=[],
            generation_metrics=["answer_relevance_heuristic", "context_utilisation_heuristic"],
        )
        result = await RAGEvaluationPipeline(cfg).run([example])
        names = {m.metric_name for m in result.metric_outputs}
        assert "answer_relevance_heuristic" in names
        assert "context_utilisation_heuristic" in names

    async def test_llm_judged_metrics_with_judge_fn(self):
        example = make_example(
            [("d1", "neural network context")],
            relevant=set(),
            query="explain neural networks",
            answer="neural networks are inspired by the brain",
        )
        cfg = RAGPipelineConfig(
            retrieval_metrics=[],
            generation_metrics=["answer_faithfulness", "answer_completeness"],
            judge_fn=mock_judge,
        )
        result = await RAGEvaluationPipeline(cfg).run([example])
        faith = next(m for m in result.metric_outputs if m.metric_name == "answer_faithfulness")
        comp = next(m for m in result.metric_outputs if m.metric_name == "answer_completeness")
        assert faith.value == pytest.approx(0.8)
        assert comp.value == pytest.approx(0.8)

    async def test_llm_judged_no_judge_fn_returns_error(self):
        example = make_example([("d1", "ctx")], set(), answer="ans")
        cfg = RAGPipelineConfig(
            retrieval_metrics=[],
            generation_metrics=["answer_faithfulness"],
            judge_fn=None,
        )
        result = await RAGEvaluationPipeline(cfg).run([example])
        faith = next(m for m in result.metric_outputs if m.metric_name == "answer_faithfulness")
        # Without judge_fn, judge_response is None → error
        assert faith.value is None

    async def test_failing_judge_fn_records_error(self):
        example = make_example(
            [("d1", "some context")],
            set(),
            query="q",
            answer="a",
        )
        cfg = RAGPipelineConfig(
            retrieval_metrics=[],
            generation_metrics=["answer_faithfulness"],
            judge_fn=failing_judge,
        )
        result = await RAGEvaluationPipeline(cfg).run([example])
        # Error recorded in result.errors
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# RAGEvaluationPipeline — mixed config
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPipelineMixed:
    async def test_full_run_all_metrics(self):
        examples = [
            make_example(
                [("a", "machine learning context"), ("b", "deep learning context")],
                relevant={"a"},
                query="what is machine learning",
                answer="machine learning is a type of AI",
                judge_response=json.dumps({"score": 0.75, "reasoning": "good"}),
            ),
            make_example(
                [("c", "neural networks are layers"), ("d", "backpropagation")],
                relevant={"c", "d"},
                query="explain neural networks",
                answer="neural networks use layers and backpropagation",
                judge_response=json.dumps({"score": 0.9, "reasoning": "complete"}),
            ),
        ]
        cfg = RAGPipelineConfig(
            retrieval_metrics=["precision_at_k", "mrr"],
            generation_metrics=["answer_relevance_heuristic", "answer_faithfulness"],
        )
        result = await RAGEvaluationPipeline(cfg).run(examples)
        assert result.total_examples == 2
        assert len(result.metric_outputs) == 4
        metric_names = {m.metric_name for m in result.metric_outputs}
        assert "precision_at_k" in metric_names
        assert "mrr" in metric_names
        assert "answer_relevance_heuristic" in metric_names
        assert "answer_faithfulness" in metric_names

    async def test_overall_status_completed(self):
        example = make_example([("a", "")], {"a"}, answer="answer")
        cfg = RAGPipelineConfig(
            retrieval_metrics=["hit_at_k"],
            generation_metrics=[],
            thresholds={"hit_at_k": 0.5},
        )
        result = await RAGEvaluationPipeline(cfg).run([example])
        assert result.overall_status == "completed"

    async def test_overall_status_with_failures(self):
        example = make_example([("x", "")], {"a"}, answer="answer")
        cfg = RAGPipelineConfig(
            retrieval_metrics=["hit_at_k"],
            generation_metrics=[],
            thresholds={"hit_at_k": 1.0},
        )
        result = await RAGEvaluationPipeline(cfg).run([example])
        assert result.overall_status == "completed_with_failures"
        assert result.has_failures is True


# ---------------------------------------------------------------------------
# RAGEvaluationPipeline — edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPipelineEdgeCases:
    async def test_empty_example_list(self):
        cfg = RAGPipelineConfig()
        result = await RAGEvaluationPipeline(cfg).run([])
        assert result.total_examples == 0
        # All metrics have None value (no data to aggregate)
        for m in result.metric_outputs:
            assert m.value is None

    async def test_unknown_retrieval_metric_records_error(self):
        example = make_example([("a", "")], {"a"})
        cfg = RAGPipelineConfig(
            retrieval_metrics=["nonexistent_metric"],
            generation_metrics=[],
        )
        result = await RAGEvaluationPipeline(cfg).run([example])
        assert len(result.errors) > 0

    async def test_unknown_generation_metric_records_error(self):
        example = make_example([("a", "")], {"a"}, answer="ans")
        cfg = RAGPipelineConfig(
            retrieval_metrics=[],
            generation_metrics=["fake_generation_metric"],
        )
        result = await RAGEvaluationPipeline(cfg).run([example])
        assert len(result.errors) > 0

    async def test_default_config_pipeline(self):
        # Default config should run without raising
        example = make_example(
            [("a", "some context"), ("b", "more context")],
            {"a"},
            answer="the answer",
            judge_response=json.dumps({"score": 0.7, "reasoning": "ok"}),
        )
        result = await RAGEvaluationPipeline().run([example])
        assert result.total_examples == 1

    async def test_per_example_error_list_populated(self):
        example = make_example([("a", "")], {"a"})
        cfg = RAGPipelineConfig(
            retrieval_metrics=["bad_metric"],
            generation_metrics=[],
        )
        result = await RAGEvaluationPipeline(cfg).run([example])
        assert len(result.per_example) == 1
        assert len(result.per_example[0]["errors"]) > 0


# ---------------------------------------------------------------------------
# RAGMetricRegistry
# ---------------------------------------------------------------------------

class TestRAGMetricRegistry:
    def test_default_registry_has_11_metrics(self):
        registry = build_default_rag_registry()
        assert len(registry) == 11

    def test_get_known_metric(self):
        registry = build_default_rag_registry()
        reg = registry.get("precision_at_k")
        assert reg.name == "precision_at_k"
        assert reg.metric_type == "DETERMINISTIC"

    def test_get_unknown_raises_key_error(self):
        registry = build_default_rag_registry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_contains(self):
        registry = build_default_rag_registry()
        assert "mrr" in registry
        assert "fake" not in registry

    def test_list_all_sorted(self):
        registry = build_default_rag_registry()
        names = registry.list_all()
        assert names == sorted(names)
        assert len(names) == 11

    def test_list_by_type_deterministic(self):
        registry = build_default_rag_registry()
        names = registry.list_by_type("DETERMINISTIC")
        assert len(names) == 7
        assert all("_at_k" in n or n in ("mrr", "context_precision", "context_recall") for n in names)

    def test_list_by_type_heuristic(self):
        registry = build_default_rag_registry()
        names = registry.list_by_type("HEURISTIC")
        assert len(names) == 2
        assert "answer_relevance_heuristic" in names
        assert "context_utilisation_heuristic" in names

    def test_list_by_type_llm_judged(self):
        registry = build_default_rag_registry()
        names = registry.list_by_type("LLM_JUDGED")
        assert len(names) == 2
        assert "answer_faithfulness" in names
        assert "answer_completeness" in names

    def test_list_retrieval(self):
        registry = build_default_rag_registry()
        names = registry.list_retrieval()
        assert len(names) == 7

    def test_list_generation(self):
        registry = build_default_rag_registry()
        names = registry.list_generation()
        assert len(names) == 4

    def test_requires_judge_set_on_llm_metrics(self):
        registry = build_default_rag_registry()
        assert registry.get("answer_faithfulness").requires_judge is True
        assert registry.get("answer_completeness").requires_judge is True
        assert registry.get("precision_at_k").requires_judge is False

    def test_custom_registration(self):
        from aieval360.evaluation.rag.models import RAGMetricOutput

        def my_metric(rag_input, threshold=None):
            return RAGMetricOutput(metric_name="my_metric", value=1.0, threshold=threshold)

        registry = RAGMetricRegistry()
        registry.register(RAGMetricRegistration(
            name="my_metric",
            fn=my_metric,
            metric_type="DETERMINISTIC",
            description="Custom test metric",
        ))
        assert "my_metric" in registry
        reg = registry.get("my_metric")
        assert reg.fn is my_metric

    def test_module_level_default_registry(self):
        # default_rag_registry is pre-built at import time
        assert len(default_rag_registry) == 11
        assert "precision_at_k" in default_rag_registry
