"""Tests for :mod:`crosslingual_rewrite.evaluation`."""

from __future__ import annotations

import math
import unittest

from tests._helpers import ensure_src_on_path

ensure_src_on_path()

from crosslingual_rewrite.data import RetrievedDocument  # noqa: E402
from crosslingual_rewrite.evaluation import (  # noqa: E402
    aggregate_metrics,
    compute_metrics,
    english_source_ratio,
    faithfulness_score,
    korean_source_ratio,
    mean_reciprocal_rank,
    ndcg,
    recall_at_k,
    source_diversity,
)


def _retrieved(entries: list[tuple[str, int, str | None, bool]]) -> list[RetrievedDocument]:
    docs: list[RetrievedDocument] = []
    for doc_id, rank, lang, is_pos in entries:
        docs.append(
            RetrievedDocument(
                doc_id=doc_id,
                text="text",
                rank=rank,
                score=float(-rank),
                source_language=lang,
                is_positive=is_pos,
            )
        )
    return docs


class RetrievalMetricsTests(unittest.TestCase):
    def test_recall_at_k_hit(self) -> None:
        docs = _retrieved([("d1", 1, "en", False), ("d2", 2, "en", True), ("d3", 3, "en", False)])
        self.assertEqual(recall_at_k(docs, "d2", k=3), 1.0)

    def test_recall_at_k_miss_when_positive_below_k(self) -> None:
        docs = _retrieved([("d1", 1, "en", False), ("d2", 2, "en", False), ("d3", 3, "en", True)])
        self.assertEqual(recall_at_k(docs, "d3", k=2), 0.0)

    def test_mrr_uses_rank_of_positive(self) -> None:
        docs = _retrieved([("d1", 1, "en", False), ("d2", 2, "en", True), ("d3", 3, "en", False)])
        self.assertAlmostEqual(mean_reciprocal_rank(docs, "d2"), 0.5)

    def test_mrr_zero_when_positive_not_retrieved(self) -> None:
        docs = _retrieved([("d1", 1, "en", False), ("d3", 2, "en", False)])
        self.assertEqual(mean_reciprocal_rank(docs, "d2"), 0.0)

    def test_ndcg_binary_single_relevant(self) -> None:
        docs = _retrieved([("d1", 1, "en", False), ("d2", 2, "en", True), ("d3", 3, "en", False)])
        expected = 1.0 / math.log2(3)
        self.assertAlmostEqual(ndcg(docs, "d2"), expected)

    def test_ndcg_zero_when_missing(self) -> None:
        docs = _retrieved([("d1", 1, "en", False)])
        self.assertEqual(ndcg(docs, "missing"), 0.0)


class SourceDiversityAndRatiosTests(unittest.TestCase):
    def test_source_diversity(self) -> None:
        docs = _retrieved([("d1", 1, "en", False), ("d2", 2, "ko", True), ("d3", 3, "en", False)])
        self.assertAlmostEqual(source_diversity(docs), 2 / 3)

    def test_english_ratio(self) -> None:
        docs = _retrieved([("d1", 1, "en", False), ("d2", 2, "ko", False)])
        self.assertEqual(english_source_ratio(docs), 0.5)

    def test_korean_ratio(self) -> None:
        docs = _retrieved([("d1", 1, "en", False), ("d2", 2, "ko", False), ("d3", 3, "ko", False)])
        self.assertAlmostEqual(korean_source_ratio(docs), 2 / 3)

    def test_empty_retrieval_returns_zero_ratios(self) -> None:
        self.assertEqual(source_diversity([]), 0.0)
        self.assertEqual(english_source_ratio([]), 0.0)
        self.assertEqual(korean_source_ratio([]), 0.0)


class FaithfulnessTests(unittest.TestCase):
    def test_perfect_overlap_is_one(self) -> None:
        self.assertAlmostEqual(faithfulness_score("python dataclass", "python dataclass"), 1.0)

    def test_partial_overlap(self) -> None:
        score = faithfulness_score("python dataclass methods", "python dataclass")
        self.assertGreater(score, 0.5)
        self.assertLess(score, 1.0)

    def test_zero_when_no_target(self) -> None:
        self.assertEqual(faithfulness_score("python", None), 0.0)

    def test_score_in_unit_interval(self) -> None:
        score = faithfulness_score("alpha beta", "gamma delta")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class AggregateAndBundleTests(unittest.TestCase):
    def test_compute_metrics_populates_all_fields(self) -> None:
        docs = _retrieved([("d1", 1, "en", False), ("d2", 2, "ko", True)])
        bundle = compute_metrics(
            docs,
            positive_doc_id="d2",
            top_k=2,
            generated_query="python dataclass",
            target_query="python dataclass",
        ).to_dict()
        for key in (
            "recall_at_k",
            "mrr",
            "ndcg",
            "source_diversity",
            "english_source_ratio",
            "korean_source_ratio",
            "faithfulness",
        ):
            self.assertIn(key, bundle)

    def test_aggregate_averages(self) -> None:
        per = [
            {"recall_at_k": 1.0, "mrr": 1.0, "ndcg": 1.0},
            {"recall_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0},
        ]
        avg = aggregate_metrics(per)
        self.assertAlmostEqual(avg["recall_at_k"], 0.5)
        self.assertAlmostEqual(avg["mrr"], 0.5)
        self.assertAlmostEqual(avg["ndcg"], 0.5)
        self.assertEqual(avg["source_diversity"], 0.0)

    def test_aggregate_empty_returns_zeros(self) -> None:
        avg = aggregate_metrics([])
        self.assertEqual(avg["recall_at_k"], 0.0)


if __name__ == "__main__":
    unittest.main()
