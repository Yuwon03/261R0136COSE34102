from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crosslingual_rewrite.xor_translation_prep import (
    TranslationAlignmentError,
    merge_translated_queries,
    prepare_xor_translation_files,
    split_query_list_file,
    validate_question_mark_query_counts,
)


class XorTranslationPrepTests(unittest.TestCase):
    def test_prepare_writes_attribute_jsonl_and_comma_question_list(self) -> None:
        records = [
            {
                "id": "ko-1",
                "lang": 4,
                "question": "서울, 수도는 어디인가요?",
                "answers": "서울",
            },
            {
                "id": "en-1",
                "lang": "en",
                "question": "ignored",
                "answers": ["ignored"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "xor_ko.jsonl"
            queries = Path(tmp) / "xor_ko_queries.txt"
            stats = prepare_xor_translation_files(
                records,
                jsonl_output=jsonl,
                query_list_output=queries,
            )
            rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
            query_text = queries.read_text(encoding="utf-8")

        self.assertEqual(stats.written, 1)
        self.assertEqual(rows[0]["question"], "서울, 수도는 어디인가요?")
        self.assertEqual(rows[0]["answers"], ["서울"])
        self.assertEqual(query_text, "서울 수도는 어디인가요?")

    def test_prepare_normalizes_query_mark_counts(self) -> None:
        records = [
            {
                "id": "ko-1",
                "lang": "ko",
                "question": "한국에서 가장 많은 성씨는 무엇인가",
                "answers": ["김"],
            },
            {
                "id": "ko-2",
                "lang": "ko",
                "question": "성리학의 어원은 무엇인가요??",
                "answers": ["성명과 의리"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            queries = Path(tmp) / "xor_ko_queries.txt"
            prepare_xor_translation_files(
                records,
                jsonl_output=Path(tmp) / "xor_ko.jsonl",
                query_list_output=queries,
            )
            query_text = queries.read_text(encoding="utf-8")

        self.assertEqual(query_text, "한국에서 가장 많은 성씨는 무엇인가?,성리학의 어원은 무엇인가요?")
        self.assertEqual(len([part for part in query_text.split("?") if part.strip(" ,")]), 2)

    def test_split_query_list_file_writes_small_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "queries.txt"
            chunks = Path(tmp) / "chunks"
            source.write_text("질문1?,질문2?,질문3?", encoding="utf-8")

            chunks_written = split_query_list_file(source, output_dir=chunks, chunk_size=2)
            files = sorted(chunks.glob("*.txt"))
            names = [path.name for path in files]
            first_chunk = files[0].read_text(encoding="utf-8")
            second_chunk = files[1].read_text(encoding="utf-8")

        self.assertEqual(chunks_written, 2)
        self.assertEqual(names, ["queries_0001.txt", "queries_0002.txt"])
        self.assertEqual(first_chunk, "질문1?,질문2?")
        self.assertEqual(second_chunk, "질문3?")

    def test_validate_question_mark_query_counts_accepts_files_and_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ko_dir = Path(tmp) / "ko"
            en_dir = Path(tmp) / "en"
            ko_dir.mkdir()
            en_dir.mkdir()
            (ko_dir / "ko_0001.txt").write_text("질문1?,질문2?", encoding="utf-8")
            (ko_dir / "ko_0002.txt").write_text("질문3?", encoding="utf-8")
            (en_dir / "en_0001.txt").write_text("question one?,question two?", encoding="utf-8")
            (en_dir / "en_0002.txt").write_text("question three?", encoding="utf-8")

            matching = validate_question_mark_query_counts(
                korean_query_list=ko_dir,
                english_query_list=en_dir,
            )
            (en_dir / "en_0002.txt").write_text("", encoding="utf-8")
            mismatched = validate_question_mark_query_counts(
                korean_query_list=ko_dir,
                english_query_list=en_dir,
            )

        self.assertTrue(matching.matches)
        self.assertEqual(matching.korean_count, 3)
        self.assertEqual(matching.english_count, 3)
        self.assertFalse(mismatched.matches)
        self.assertEqual(mismatched.english_count, 2)

    def test_merge_adds_target_query_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "xor_ko.jsonl"
            translations = Path(tmp) / "english.txt"
            output = Path(tmp) / "xor_ko_with_target.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "1", "question": "질문1", "answers": ["a"]}, ensure_ascii=False),
                        json.dumps({"id": "2", "question": "질문2", "answers": ["b"]}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            translations.write_text("english query one,english query two", encoding="utf-8")

            stats = merge_translated_queries(
                jsonl_input=source,
                english_query_list=translations,
                output_jsonl=output,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(stats.merged, 2)
        self.assertEqual(rows[0]["target_query"], "english query one")
        self.assertEqual(rows[1]["target_query"], "english query two")

    def test_merge_rejects_count_mismatch_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "xor_ko.jsonl"
            translations = Path(tmp) / "english.txt"
            output = Path(tmp) / "xor_ko_with_target.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "1", "question": "질문1", "answers": ["a"]}, ensure_ascii=False),
                        json.dumps({"id": "2", "question": "질문2", "answers": ["b"]}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            translations.write_text("english query one", encoding="utf-8")

            with self.assertRaises(TranslationAlignmentError):
                merge_translated_queries(
                    jsonl_input=source,
                    english_query_list=translations,
                    output_jsonl=output,
                )


if __name__ == "__main__":
    unittest.main()
