"""Tests for :mod:`crosslingual_rewrite.retriever`."""

from __future__ import annotations

import unittest

from tests._helpers import ensure_src_on_path

ensure_src_on_path()

from crosslingual_rewrite.data import CorpusDocument  # noqa: E402
from crosslingual_rewrite.retriever import BM25Retriever, tokenize  # noqa: E402


def _corpus() -> list[CorpusDocument]:
    return [
        CorpusDocument(
            doc_id="doc-python",
            title="Python",
            text="python dataclass generates init and repr methods for classes",
            source_language="en",
        ),
        CorpusDocument(
            doc_id="doc-cooking",
            title="Cooking",
            text="cooking tips for beginners knives seasoning and sauces",
            source_language="en",
        ),
        CorpusDocument(
            doc_id="doc-python-ko",
            title="파이썬",
            text="파이썬 dataclass는 클래스에 자동으로 메서드를 생성합니다",
            source_language="ko",
        ),
        CorpusDocument(
            doc_id="doc-tie-a",
            title="tie a",
            text="same content sample",
            source_language="en",
        ),
        CorpusDocument(
            doc_id="doc-tie-b",
            title="tie b",
            text="same content sample",
            source_language="en",
        ),
    ]


class TokenizerTests(unittest.TestCase):
    def test_tokenize_english_lowercases_and_splits(self) -> None:
        tokens = tokenize("Python Dataclass BM25")
        self.assertIn("python", tokens)
        self.assertIn("dataclass", tokens)
        self.assertIn("bm25", tokens)

    def test_tokenize_korean_adds_unigrams_and_bigrams(self) -> None:
        tokens = tokenize("파이썬")
        self.assertIn("파", tokens)
        self.assertIn("이", tokens)
        self.assertIn("썬", tokens)
        self.assertIn("파이", tokens)
        self.assertIn("이썬", tokens)

    def test_tokenize_mixed_text(self) -> None:
        tokens = tokenize("파이썬 dataclass는?")
        self.assertIn("dataclass", tokens)
        self.assertIn("파", tokens)
        self.assertIn("썬", tokens)
        self.assertIn("파이", tokens)

    def test_empty_text_returns_empty_list(self) -> None:
        self.assertEqual(tokenize(""), [])


class BM25RetrieverTests(unittest.TestCase):
    def test_top_k_respected(self) -> None:
        retriever = BM25Retriever(_corpus())
        results = retriever.retrieve(
            "python dataclass", top_k=2, positive_doc_id="doc-python"
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[1].rank, 2)

    def test_positive_doc_ranked_first_for_matching_query(self) -> None:
        retriever = BM25Retriever(_corpus())
        results = retriever.retrieve(
            "python dataclass init repr methods",
            top_k=3,
            positive_doc_id="doc-python",
        )
        self.assertEqual(results[0].doc_id, "doc-python")
        self.assertTrue(results[0].is_positive)

    def test_tie_broken_by_doc_id(self) -> None:
        retriever = BM25Retriever(_corpus())
        results = retriever.retrieve(
            "same content sample",
            top_k=5,
            positive_doc_id="doc-tie-a",
        )
        tied_ids = [r.doc_id for r in results if r.doc_id.startswith("doc-tie-")]
        self.assertEqual(tied_ids, sorted(tied_ids))

    def test_korean_query_matches_korean_document(self) -> None:
        retriever = BM25Retriever(_corpus())
        results = retriever.retrieve("파이썬", top_k=3, positive_doc_id="doc-python-ko")
        self.assertTrue(any(r.doc_id == "doc-python-ko" for r in results))

    def test_empty_corpus_returns_empty_results(self) -> None:
        retriever = BM25Retriever([])
        self.assertEqual(retriever.retrieve("any", top_k=3, positive_doc_id="x"), [])

    def test_top_k_must_be_positive(self) -> None:
        retriever = BM25Retriever(_corpus())
        with self.assertRaises(ValueError):
            retriever.retrieve("python", top_k=0, positive_doc_id="doc-python")

    def test_score_is_stable_across_calls(self) -> None:
        retriever = BM25Retriever(_corpus())
        a = retriever.retrieve("python dataclass", top_k=3, positive_doc_id="doc-python")
        b = retriever.retrieve("python dataclass", top_k=3, positive_doc_id="doc-python")
        self.assertEqual([(r.doc_id, r.rank) for r in a], [(r.doc_id, r.rank) for r in b])

    def test_is_positive_flag_set_correctly(self) -> None:
        retriever = BM25Retriever(_corpus())
        results = retriever.retrieve(
            "python dataclass", top_k=5, positive_doc_id="doc-python"
        )
        positives = [r for r in results if r.is_positive]
        self.assertEqual(len(positives), 1)
        self.assertEqual(positives[0].doc_id, "doc-python")


if __name__ == "__main__":
    unittest.main()
