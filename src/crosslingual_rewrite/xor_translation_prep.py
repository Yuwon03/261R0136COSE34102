"""Prepare XOR-TyDi Korean questions for external translation.

This module deliberately does not call translation APIs. It creates:

* a JSONL file with Korean questions, answers, and source attributes;
* a comma-only text file containing the Korean questions for an external AI
  translation workflow, optionally split into smaller chunk files;
* a merge helper that attaches comma-separated English queries back to JSONL.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .dataset_builder import (
    _XOR_LANG_LABELS,
    _answer_texts_from_xor,
    _clean_text,
    has_korean,
    load_hf_records,
)


_COMMA_RE = re.compile(r"[,，、]+")
_QUESTION_RE = re.compile(r"[?？]+")
_SPACE_RE = re.compile(r"\s+")


@dataclass
class XorPrepStats:
    source_seen: int = 0
    written: int = 0
    skipped_non_korean: int = 0
    skipped_no_question: int = 0
    skipped_no_answers: int = 0
    skipped_duplicate_question: int = 0
    merged: int = 0
    skipped_no_translation: int = 0
    chunks_written: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class QueryCountValidation:
    korean_count: int
    english_count: int
    matches: bool

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


class TranslationAlignmentError(ValueError):
    """Raised when translated query files do not align with source rows."""


def _csv_safe_query(text: str) -> str:
    """Make one question safe for a comma-only list and ``?`` counting."""

    cleaned = _clean_text(text)
    cleaned = _COMMA_RE.sub(" ", cleaned)
    cleaned = _QUESTION_RE.sub(" ", cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned)
    cleaned = cleaned.strip(" .。")
    return f"{cleaned}?" if cleaned else ""


def _dedupe_key(text: str) -> str:
    return _SPACE_RE.sub(" ", _clean_text(text).casefold())


def _normalize_xor_lang(value: Any) -> str:
    if isinstance(value, int) and 0 <= value < len(_XOR_LANG_LABELS):
        return _XOR_LANG_LABELS[value]
    return _clean_text(value).lower()


def normalize_xor_export_record(record: Mapping[str, Any], *, split_name: str) -> dict[str, Any] | None:
    """Return a stable local JSONL row for a Korean XOR-TyDi item."""

    lang_raw = record.get("lang") if record.get("lang") is not None else record.get("language")
    lang = _normalize_xor_lang(lang_raw)
    if lang and lang != "ko":
        return None
    question = _clean_text(record.get("question") or record.get("query"))
    if not question or not has_korean(question):
        return None
    answers = _answer_texts_from_xor(record.get("answers") or record.get("answer"))
    if not answers:
        return None
    source_id = _clean_text(record.get("id") or record.get("example_id"))
    return {
        "id": source_id,
        "lang": "ko",
        "question": question,
        "answers": list(answers),
        "split": _clean_text(record.get("split") or split_name) or split_name,
        "metadata": {
            "source": "xor_tydi",
            "original_id": source_id,
        },
    }


def prepare_xor_translation_files(
    records: Iterable[Mapping[str, Any]],
    *,
    jsonl_output: str | Path,
    query_list_output: str | Path,
    query_chunks_dir: str | Path | None = None,
    chunk_size: int = 100,
    split_name: str = "train",
    limit: int | None = None,
    dedupe_questions: bool = True,
) -> XorPrepStats:
    """Write XOR Korean attribute JSONL plus comma-separated question list."""

    stats = XorPrepStats()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    queries: list[str] = []

    for raw in records:
        if limit is not None and stats.source_seen >= limit:
            break
        stats.source_seen += 1
        row = normalize_xor_export_record(raw, split_name=split_name)
        if row is None:
            lang_raw = raw.get("lang") if raw.get("lang") is not None else raw.get("language")
            lang = _normalize_xor_lang(lang_raw)
            question = _clean_text(raw.get("question") or raw.get("query"))
            answers = _answer_texts_from_xor(raw.get("answers") or raw.get("answer"))
            if lang and lang != "ko":
                stats.skipped_non_korean += 1
            elif not question or not has_korean(question):
                stats.skipped_no_question += 1
            elif not answers:
                stats.skipped_no_answers += 1
            else:
                stats.skipped_no_question += 1
            continue
        key = _dedupe_key(str(row["question"]))
        if dedupe_questions and key in seen:
            stats.skipped_duplicate_question += 1
            continue
        seen.add(key)
        rows.append(row)
        queries.append(_csv_safe_query(str(row["question"])))
        stats.written += 1

    jsonl_path = Path(jsonl_output)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    query_path = Path(query_list_output)
    query_path.parent.mkdir(parents=True, exist_ok=True)
    query_path.write_text(",".join(queries), encoding="utf-8")
    if query_chunks_dir is not None:
        stats.chunks_written = write_query_chunks(
            queries,
            output_dir=query_chunks_dir,
            chunk_size=chunk_size,
            prefix=query_path.stem,
        )
    return stats


def _read_comma_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    reader = csv.reader([text])
    return [_clean_text(item) for item in next(reader)]


def _read_comma_list(path: str | Path) -> list[str]:
    query_path = Path(path)
    if query_path.is_dir():
        queries: list[str] = []
        for child in sorted(query_path.glob("*.txt")):
            queries.extend(_read_comma_text(child.read_text(encoding="utf-8")))
        return queries
    return _read_comma_text(query_path.read_text(encoding="utf-8"))


def write_query_chunks(
    queries: Iterable[str],
    *,
    output_dir: str | Path,
    chunk_size: int = 100,
    prefix: str = "queries",
) -> int:
    """Write comma-separated query chunks and return the number of files."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    cleaned_queries = []
    for query in queries:
        cleaned = _csv_safe_query(query)
        if cleaned:
            cleaned_queries.append(cleaned)
    chunk_path = Path(output_dir)
    chunk_path.mkdir(parents=True, exist_ok=True)
    chunks_written = 0
    for start in range(0, len(cleaned_queries), chunk_size):
        chunks_written += 1
        chunk = cleaned_queries[start : start + chunk_size]
        path = chunk_path / f"{prefix}_{chunks_written:04d}.txt"
        path.write_text(",".join(chunk), encoding="utf-8")
    return chunks_written


def split_query_list_file(
    query_list_input: str | Path,
    *,
    output_dir: str | Path,
    chunk_size: int = 100,
    prefix: str | None = None,
) -> int:
    """Split an existing comma-separated query list into smaller files."""

    source = Path(query_list_input)
    return write_query_chunks(
        _read_comma_list(source),
        output_dir=output_dir,
        chunk_size=chunk_size,
        prefix=prefix or source.stem,
    )


def count_question_mark_queries(path: str | Path) -> int:
    """Count non-empty queries by splitting one file or a chunk directory on ``?``."""

    query_path = Path(path)
    if query_path.is_dir():
        return sum(count_question_mark_queries(child) for child in sorted(query_path.glob("*.txt")))
    text = query_path.read_text(encoding="utf-8")
    return len([part for part in text.split("?") if part.strip(" \t\r\n,")])


def validate_question_mark_query_counts(
    *,
    korean_query_list: str | Path,
    english_query_list: str | Path,
) -> QueryCountValidation:
    """Validate that Korean and English query files contain the same ``?`` count."""

    korean_count = count_question_mark_queries(korean_query_list)
    english_count = count_question_mark_queries(english_query_list)
    return QueryCountValidation(
        korean_count=korean_count,
        english_count=english_count,
        matches=korean_count == english_count,
    )


def merge_translated_queries(
    *,
    jsonl_input: str | Path,
    english_query_list: str | Path,
    output_jsonl: str | Path,
    field_name: str = "target_query",
    require_exact_count: bool = True,
) -> XorPrepStats:
    """Attach comma-separated English queries to prepared XOR JSONL."""

    stats = XorPrepStats()
    translations = _read_comma_list(english_query_list)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict[str, Any]] = []
    with Path(jsonl_input).open("r", encoding="utf-8") as src:
        for line in src:
            stripped = line.strip()
            if stripped:
                source_rows.append(json.loads(stripped))
    stats.source_seen = len(source_rows)

    if require_exact_count and len(source_rows) != len(translations):
        raise TranslationAlignmentError(
            "Translated query count does not match source row count: "
            f"source_rows={len(source_rows)} translations={len(translations)}"
        )

    with out_path.open("w", encoding="utf-8") as dst:
        for index, row in enumerate(source_rows):
            translation = translations[index] if index < len(translations) else ""
            if not translation:
                stats.skipped_no_translation += 1
                if require_exact_count:
                    raise TranslationAlignmentError(
                        f"Missing translated query at 0-based index {index}"
                    )
                continue
            row[field_name] = translation
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats.merged += 1
            stats.written += 1
    return stats


def prepare_xor_from_hf(
    *,
    jsonl_output: str | Path,
    query_list_output: str | Path,
    query_chunks_dir: str | Path | None = None,
    chunk_size: int = 100,
    split: str = "train",
    dataset_name: str | None = None,
    config_name: str | None = None,
    trust_remote_code: bool = False,
    limit: int | None = None,
) -> XorPrepStats:
    records = load_hf_records(
        "xor_tydi",
        split=split,
        dataset_name=dataset_name,
        config_name=config_name,
        trust_remote_code=trust_remote_code,
    )
    return prepare_xor_translation_files(
        records,
        jsonl_output=jsonl_output,
        query_list_output=query_list_output,
        query_chunks_dir=query_chunks_dir,
        chunk_size=chunk_size,
        split_name=split,
        limit=limit,
    )


__all__ = [
    "XorPrepStats",
    "QueryCountValidation",
    "TranslationAlignmentError",
    "count_question_mark_queries",
    "normalize_xor_export_record",
    "prepare_xor_translation_files",
    "split_query_list_file",
    "validate_question_mark_query_counts",
    "write_query_chunks",
    "merge_translated_queries",
    "prepare_xor_from_hf",
]
