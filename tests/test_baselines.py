"""Tests for :mod:`crosslingual_rewrite.baselines` (the shared runner).

These tests use the smoke fixtures plus small synthetic examples so they run
quickly without needing any HuggingFace model download.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._helpers import ensure_src_on_path, make_config

ensure_src_on_path()

from crosslingual_rewrite.baselines import (  # noqa: E402
    ExampleOutcome,
    MethodError,
    MethodRunResult,
    generate_queries,
    run_method,
)
from crosslingual_rewrite.data import (  # noqa: E402
    CorpusDocument,
    RewriteExample,
    load_corpus,
    load_dataset,
)
from crosslingual_rewrite.modeling import MockRewriteModel  # noqa: E402
from crosslingual_rewrite.training import train_retrieval_aware, train_supervised  # noqa: E402


def _toy_examples() -> list[RewriteExample]:
    return [
        RewriteExample(
            question_ko="파이썬 dataclass란?",
            positive_doc_id="doc-python",
            negative_doc_id="doc-cooking",
            target_query="python dataclass generated methods",
            example_id="ex-1",
            query_type="mixed_ko_en",
        ),
        RewriteExample(
            question_ko="서울에서 제일 높은 산은?",
            positive_doc_id="doc-bukhansan",
            negative_doc_id="doc-python",
            target_query="highest mountain seoul",
            example_id="ex-2",
            query_type="pure_ko",
        ),
    ]


def _toy_corpus() -> list[CorpusDocument]:
    return [
        CorpusDocument(
            doc_id="doc-python",
            title="Python dataclass",
            text="python dataclass generates init repr methods",
            source_language="en",
        ),
        CorpusDocument(
            doc_id="doc-bukhansan",
            title="Bukhansan",
            text="bukhansan highest mountain seoul granite peaks",
            source_language="en",
        ),
        CorpusDocument(
            doc_id="doc-cooking",
            title="Cooking tips",
            text="basic cooking tips knives and sauces",
            source_language="en",
        ),
    ]


class QueryGenerationTests(unittest.TestCase):
    def test_raw_returns_korean_question(self) -> None:
        queries = generate_queries("raw", _toy_examples())
        self.assertEqual(queries[0], "파이썬 dataclass란?")

    def test_translate_uses_target_query(self) -> None:
        queries = generate_queries("translate", _toy_examples())
        self.assertEqual(queries[0], "python dataclass generated methods")

    def test_translate_raises_when_target_missing(self) -> None:
        broken = [
            RewriteExample(
                question_ko="질문",
                positive_doc_id="d1",
                negative_doc_id="d2",
                target_query=None,
                example_id="ex-missing",
            )
        ]
        from crosslingual_rewrite.data import DataValidationError

        with self.assertRaises(DataValidationError):
            generate_queries("translate", broken)

    def test_supervised_requires_model(self) -> None:
        with self.assertRaises(MethodError):
            generate_queries("supervised", _toy_examples(), model=None)

    def test_supervised_delegates_to_model(self) -> None:
        model = MockRewriteModel(variant="test")
        queries = generate_queries("supervised", _toy_examples(), model=model)
        self.assertEqual(queries[0], "python dataclass generated methods")

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(MethodError):
            generate_queries("bogus", _toy_examples())


class RunMethodTests(unittest.TestCase):
    def _cfg(self, tmp: Path, *, retrieval_loss_weight: float = 0.5) -> object:
        runs_dir = tmp / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        return make_config(
            runs_dir=runs_dir,
            top_k=3,
            retrieval_loss_weight=retrieval_loss_weight,
        )

    def test_raw_run_produces_outcomes_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(Path(tmp))
            examples = _toy_examples()
            result = run_method(
                cfg,
                "raw",
                examples=examples,
                corpus=_toy_corpus(),
            )
            self.assertIsInstance(result, MethodRunResult)
            self.assertEqual(result.example_count, 2)
            self.assertEqual(result.method, "raw")
            self.assertEqual(len(result.outcomes), 2)
            for outcome in result.outcomes:
                self.assertIsInstance(outcome, ExampleOutcome)

    def test_translate_achieves_perfect_retrieval_on_toy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(Path(tmp))
            result = run_method(
                cfg,
                "translate",
                examples=_toy_examples(),
                corpus=_toy_corpus(),
            )
            self.assertEqual(result.error_count, 0)
            self.assertAlmostEqual(result.aggregated["recall_at_k"], 1.0, places=6)
            self.assertAlmostEqual(result.aggregated["mrr"], 1.0, places=6)

    def test_supervised_with_mock_uses_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(Path(tmp))
            examples = load_dataset(cfg.data.dataset_path)
            corpus = load_corpus(cfg.data.corpus_path)

            training_result = train_supervised(cfg, examples)
            self.assertEqual(training_result.mode, "mock")
            result = run_method(
                cfg,
                "supervised",
                examples=examples,
                corpus=corpus,
                checkpoint_dir=training_result.checkpoint_dir,
            )
            self.assertEqual(result.error_count, 0)
            self.assertGreaterEqual(result.aggregated["recall_at_k"], 0.5)

    def test_retrieval_aware_with_mock_trains_and_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(Path(tmp))
            examples = load_dataset(cfg.data.dataset_path)
            corpus = load_corpus(cfg.data.corpus_path)

            training_result = train_retrieval_aware(cfg, examples, corpus)
            self.assertEqual(training_result.mode, "mock")
            self.assertGreater(training_result.total_steps, 0)
            result = run_method(
                cfg,
                "retrieval_aware",
                examples=examples,
                corpus=corpus,
                checkpoint_dir=training_result.checkpoint_dir,
            )
            self.assertEqual(result.error_count, 0)
            self.assertGreaterEqual(result.aggregated["recall_at_k"], 0.5)

    def test_limit_truncates_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(Path(tmp))
            result = run_method(
                cfg,
                "raw",
                examples=_toy_examples(),
                corpus=_toy_corpus(),
                limit=1,
            )
            self.assertEqual(result.example_count, 1)

    def test_translate_missing_target_surfaces_as_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(Path(tmp))
            broken = _toy_examples()
            broken[0] = RewriteExample(
                question_ko="no target",
                positive_doc_id="doc-python",
                negative_doc_id="doc-cooking",
                target_query=None,
                example_id="ex-broken",
            )
            result = run_method(
                cfg,
                "translate",
                examples=broken,
                corpus=_toy_corpus(),
            )
            self.assertEqual(result.error_count, len(broken))


if __name__ == "__main__":
    unittest.main()
