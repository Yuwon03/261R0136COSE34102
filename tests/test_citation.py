"""Tests for citation-aware planning and scoring utilities."""

from __future__ import annotations

import unittest

from tests._helpers import ensure_src_on_path

ensure_src_on_path()

from crosslingual_rewrite.citation import (  # noqa: E402
    CitationCandidate,
    CitationCandidateRecord,
    SearchPlan,
    compute_citation_metrics,
    normalize_support_label,
    score_candidate_record,
)
from crosslingual_rewrite.citation_retrieval import reciprocal_rank_fusion  # noqa: E402
from crosslingual_rewrite.data import CorpusDocument  # noqa: E402
from crosslingual_rewrite.search_planner import parse_search_plan_output  # noqa: E402


class CitationUtilityTests(unittest.TestCase):
    def test_support_label_normalization(self) -> None:
        self.assertEqual(normalize_support_label("good"), "supported")
        self.assertEqual(normalize_support_label("bad"), "unsupported")
        self.assertEqual(normalize_support_label("partially_supported"), "partial")

    def test_citation_metrics_count_support(self) -> None:
        citations = [
            CitationCandidate(
                doc_id="d1",
                chunk_id="d1",
                title="One",
                url=None,
                language="en",
                text="answer text",
                rank=1,
                support_label="supported",
            ),
            CitationCandidate(
                doc_id="d2",
                chunk_id="d2",
                title="Two",
                url=None,
                language="ko",
                text="other",
                rank=2,
                support_label="unsupported",
            ),
        ]
        metrics = compute_citation_metrics(citations, positive_doc_id="d1")
        self.assertEqual(metrics["recall_at_10"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertAlmostEqual(metrics["citation_precision"], 0.5)
        self.assertEqual(metrics["cross_lingual_success_rate"], 1.0)

    def test_score_candidate_record_uses_positive_doc(self) -> None:
        record = CitationCandidateRecord(
            question_id="q1",
            question="질문",
            query_type="pure_ko",
            candidate_id="q1:raw",
            search_plan=SearchPlan(method="raw", queries=["질문"]),
            positive_doc_id="d1",
            citations=[
                CitationCandidate(
                    doc_id="d1",
                    chunk_id="d1",
                    title="One",
                    url=None,
                    language="en",
                    text="text",
                    rank=1,
                )
            ],
        )
        scored = score_candidate_record(record)
        self.assertGreater(scored.candidate_score, 0.0)
        self.assertEqual(scored.citations[0].support_label, "supported")


class FusionTests(unittest.TestCase):
    def test_rrf_is_deterministic(self) -> None:
        docs = [
            CorpusDocument(doc_id="a", title="A", text="alpha"),
            CorpusDocument(doc_id="b", title="B", text="beta"),
            CorpusDocument(doc_id="c", title="C", text="gamma"),
        ]
        fused = reciprocal_rank_fusion(
            {
                "bm25": [(docs[0], 2.0), (docs[1], 1.0)],
                "dense": [(docs[1], 3.0), (docs[2], 1.0)],
            },
            top_k=3,
        )
        self.assertEqual([hit.doc.doc_id for hit in fused], ["b", "a", "c"])


class SearchPlannerParsingTests(unittest.TestCase):
    def test_parse_search_plan_json_block(self) -> None:
        plan = parse_search_plan_output(
            'Here is the plan: {"queries": ["Lakers playoffs history"], "entities": ["Lakers"], '
            '"answer_type": "date", "preferred_source_languages": ["en"], '
            '"source_priority": ["official"]}',
            question="언제 레이커스가 마지막으로 플레이오프에 진출했나요",
        )
        self.assertEqual(plan.method, "citation_planner")
        self.assertEqual(plan.queries, ["Lakers playoffs history"])
        self.assertEqual(plan.entities, ["Lakers"])

    def test_parse_search_plan_falls_back_on_invalid_output(self) -> None:
        plan = parse_search_plan_output("not json", question="질문")
        self.assertEqual(plan.method, "citation_planner")
        self.assertEqual(plan.queries, ["질문"])
        self.assertEqual(plan.metadata["parse_error"], "missing_json")


if __name__ == "__main__":
    unittest.main()
