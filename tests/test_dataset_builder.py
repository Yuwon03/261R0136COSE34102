from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crosslingual_rewrite.data import CorpusDocument
from crosslingual_rewrite.dataset_builder import (
    build_dataset_from_records,
    doc_contains_answer,
    normalize_source_record,
)


class DatasetBuilderTests(unittest.TestCase):
    def test_normalize_mkqa_record_uses_korean_question_and_english_query(self) -> None:
        record = {
            "example_id": 123,
            "query": "who created Python",
            "queries": {"ko": "파이썬은 누가 만들었나요?", "en": "who created Python"},
            "answers": {
                "en": [{"type": "entity", "text": "Guido van Rossum", "aliases": ["Rossum"]}]
            },
        }

        source = normalize_source_record(record, source="mkqa")

        self.assertIsNotNone(source)
        assert source is not None
        self.assertEqual(source.question_ko, "파이썬은 누가 만들었나요?")
        self.assertEqual(source.target_query, "who created Python")
        self.assertIn("Guido van Rossum", source.answers)
        self.assertIn("Rossum", source.answers)

    def test_normalize_xor_record_requires_external_target_query(self) -> None:
        record = {
            "id": "ko-1",
            "lang": "ko",
            "question": "파이썬은 누가 만들었나요?",
            "answers": ["Guido van Rossum"],
            "split": "train",
        }

        source = normalize_source_record(record, source="xor_tydi")

        self.assertIsNotNone(source)
        assert source is not None
        self.assertEqual(source.target_query, "")
        self.assertTrue(source.metadata["requires_external_target_query"])

    def test_normalize_xor_record_accepts_classlabel_language_id(self) -> None:
        record = {
            "id": "ko-1",
            "lang": 4,
            "question": "파이썬은 누가 만들었나요?",
            "answers": "Guido van Rossum",
            "split": "train",
        }

        source = normalize_source_record(record, source="xor_tydi")

        self.assertIsNotNone(source)
        assert source is not None
        self.assertEqual(source.question_ko, "파이썬은 누가 만들었나요?")
        self.assertEqual(source.answers, ("Guido van Rossum",))

    def test_doc_contains_answer_normalizes_case_and_punctuation(self) -> None:
        doc = CorpusDocument(
            doc_id="doc-python",
            title="Python",
            text="Python was created by Guido van Rossum.",
            source_language="en",
        )

        self.assertTrue(doc_contains_answer(doc, ["guido van rossum"]))
        self.assertFalse(doc_contains_answer(doc, ["James Gosling"]))

    def test_build_dataset_from_records_writes_positive_and_hard_negative(self) -> None:
        records = [
            {
                "example_id": "123",
                "query": "who created Python",
                "queries": {
                    "ko": "파이썬은 누가 만들었나요?",
                    "en": "who created Python",
                },
                "answers": {
                    "en": [
                        {
                            "type": "entity",
                            "text": "Guido van Rossum",
                            "aliases": [],
                        }
                    ]
                },
            }
        ]
        corpus = [
            CorpusDocument(
                doc_id="doc-positive",
                title="Python programming language",
                text="Python was created by Guido van Rossum and is widely used.",
                source_language="en",
            ),
            CorpusDocument(
                doc_id="doc-negative",
                title="Python snake",
                text="Python snakes are large reptiles, not programming language creators.",
                source_language="en",
            ),
            CorpusDocument(
                doc_id="doc-unrelated",
                title="Coffee",
                text="Coffee is a brewed drink.",
                source_language="en",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset.jsonl"
            stats = build_dataset_from_records(
                records,
                source="mkqa",
                corpus=corpus,
                output_path=output,
                top_k_positive=3,
                top_k_negative=3,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(stats.written, 1)
        self.assertEqual(rows[0]["question_ko"], "파이썬은 누가 만들었나요?")
        self.assertEqual(rows[0]["positive_doc_id"], "doc-positive")
        self.assertEqual(rows[0]["negative_doc_id"], "doc-negative")
        self.assertEqual(rows[0]["metadata"]["labeling_rule"], "retriever_candidates_answer_containment")


if __name__ == "__main__":
    unittest.main()
