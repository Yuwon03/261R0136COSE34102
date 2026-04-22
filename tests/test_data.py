"""Tests for :mod:`crosslingual_rewrite.data`."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._helpers import SMOKE_CORPUS, SMOKE_DATASET, ensure_src_on_path, write_jsonl

ensure_src_on_path()

from crosslingual_rewrite.data import (  # noqa: E402
    DataValidationError,
    classify_query_type,
    example_from_record,
    load_corpus,
    load_dataset,
)


class QueryTypeClassifierTests(unittest.TestCase):
    def test_pure_korean(self) -> None:
        self.assertEqual(classify_query_type("파이썬은 무엇입니까?"), "pure_ko")

    def test_mixed_korean_english(self) -> None:
        self.assertEqual(
            classify_query_type("파이썬 dataclass는 무엇인가?"), "mixed_ko_en"
        )

    def test_english_only(self) -> None:
        self.assertEqual(classify_query_type("what is bm25"), "non_ko")

    def test_other_for_digits_and_symbols(self) -> None:
        self.assertEqual(classify_query_type("12345 ???"), "other")

    def test_empty_string(self) -> None:
        self.assertEqual(classify_query_type(""), "other")


class DatasetLoaderTests(unittest.TestCase):
    def test_load_smoke_dataset(self) -> None:
        examples = load_dataset(SMOKE_DATASET)
        self.assertGreaterEqual(len(examples), 6)
        pure_ko = [ex for ex in examples if ex.query_type == "pure_ko"]
        mixed = [ex for ex in examples if ex.query_type == "mixed_ko_en"]
        self.assertGreaterEqual(len(pure_ko), 3)
        self.assertGreaterEqual(len(mixed), 2)
        for ex in examples:
            self.assertTrue(ex.positive_doc_id)
            self.assertTrue(ex.negative_doc_id)
            self.assertTrue(ex.question_ko)

    def test_missing_required_field_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "example_id": "ex-1",
                        "question_ko": "질문",
                        "positive_doc_id": "d1",
                    }
                ],
            )
            with self.assertRaises(DataValidationError):
                load_dataset(path)

    def test_query_type_inferred_when_missing(self) -> None:
        record = {
            "question_ko": "안녕 world",
            "positive_doc_id": "d1",
            "negative_doc_id": "d2",
        }
        example = example_from_record(record)
        self.assertEqual(example.query_type, "mixed_ko_en")

    def test_require_target_query_error_message(self) -> None:
        record = {
            "question_ko": "질문",
            "positive_doc_id": "d1",
            "negative_doc_id": "d2",
        }
        example = example_from_record(record)
        with self.assertRaises(DataValidationError) as ctx:
            example.require_target_query(context="translate")
        self.assertIn("translate", str(ctx.exception))

    def test_invalid_json_line_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text('{"question_ko": "ok"}\n{not json\n', encoding="utf-8")
            with self.assertRaises(DataValidationError):
                load_dataset(path)


class CorpusLoaderTests(unittest.TestCase):
    def test_load_smoke_corpus(self) -> None:
        docs = load_corpus(SMOKE_CORPUS)
        self.assertGreaterEqual(len(docs), 12)
        ids = {d.doc_id for d in docs}
        self.assertIn("doc-python-dataclass", ids)
        self.assertIn("doc-cooking", ids)

    def test_missing_required_field_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            write_jsonl(path, [{"doc_id": "d1"}])
            with self.assertRaises(DataValidationError):
                load_corpus(path)

    def test_duplicate_doc_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dup.jsonl"
            write_jsonl(
                path,
                [
                    {"doc_id": "x", "text": "a"},
                    {"doc_id": "x", "text": "b"},
                ],
            )
            with self.assertRaises(DataValidationError):
                load_corpus(path)


if __name__ == "__main__":
    unittest.main()
