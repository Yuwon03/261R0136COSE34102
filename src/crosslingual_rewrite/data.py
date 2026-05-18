"""Canonical example + corpus dataclasses and JSONL loaders.

Every dataset example used in this project is normalized into a single
``RewriteExample`` object. Downstream code must not depend on raw dataset
field names; all access goes through this class.

Similarly, every retrievable document is normalized into a ``CorpusDocument``.

This module also contains a deterministic Korean/English query-type
classifier used for slicing evaluation metrics by language mix.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping


class DataValidationError(ValueError):
    """Raised when a dataset or corpus record is missing required fields."""


QueryType = Literal["pure_ko", "mixed_ko_en", "non_ko", "other"]

_VALID_QUERY_TYPES: frozenset[str] = frozenset({"pure_ko", "mixed_ko_en", "non_ko", "other"})


@dataclass
class RewriteExample:
    """Canonical form of one question in the retrieval-rewrite corpus.

    Required fields: ``question_ko``, ``positive_doc_id``, ``negative_doc_id``.

    ``target_query`` is required for supervised training and the gold-query
    upper-bound baseline. It may be ``None`` for raw-only or automatic
    translation inference settings. Downstream code that needs it should
    check and raise explicitly.
    """

    question_ko: str
    positive_doc_id: str
    negative_doc_id: str
    target_query: str | None = None
    example_id: str | None = None
    dataset_name: str | None = None
    split_name: str | None = None
    query_type: QueryType = "other"
    source_language: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def require_target_query(self, *, context: str) -> str:
        """Return ``target_query`` or raise ``DataValidationError``."""

        if self.target_query is None or not self.target_query.strip():
            raise DataValidationError(
                f"{context} requires target_query for example {self.example_id!r}"
            )
        return self.target_query


@dataclass
class CorpusDocument:
    """Canonical form of one retrievable document."""

    doc_id: str
    text: str
    source_language: str | None = None
    title: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedDocument:
    """One document returned by the retriever at query time."""

    doc_id: str
    text: str
    rank: int
    score: float
    source_language: str | None = None
    is_positive: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_serializable(self) -> dict[str, Any]:
        """Return a JSON-ready dict matching the per-example output schema."""

        return {
            "doc_id": self.doc_id,
            "rank": int(self.rank),
            "score": float(self.score),
            "source_language": self.source_language,
            "is_positive": bool(self.is_positive),
            "text": self.text,
        }


def _iter_jsonl(path: str | os.PathLike[str]) -> Iterable[tuple[int, dict[str, Any]]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"Invalid JSON on line {lineno} of {file_path}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise DataValidationError(
                    f"JSONL records must be objects. Line {lineno} of {file_path}: {record!r}"
                )
            yield lineno, record


_REQUIRED_EXAMPLE_FIELDS: tuple[str, ...] = (
    "question_ko",
    "positive_doc_id",
    "negative_doc_id",
)

_REQUIRED_CORPUS_FIELDS: tuple[str, ...] = ("doc_id", "text")


def classify_query_type(text: str) -> QueryType:
    """Deterministically label a query by its script mix.

    Rules
    -----
    * A character is Korean when it falls in the Hangul Syllables block
      (U+AC00..U+D7A3) or Hangul Jamo (U+1100..U+11FF, U+3130..U+318F).
    * A character is English when it is ASCII ``[A-Za-z]``.
    * ``pure_ko``: contains Korean characters and no English letters.
    * ``mixed_ko_en``: contains both Korean and English letters.
    * ``non_ko``: contains English letters but no Korean characters.
    * ``other``: has neither Korean nor English letters.
    """

    if text is None:
        return "other"
    has_korean = False
    has_english = False
    for ch in text:
        codepoint = ord(ch)
        if 0xAC00 <= codepoint <= 0xD7A3:
            has_korean = True
        elif 0x1100 <= codepoint <= 0x11FF:
            has_korean = True
        elif 0x3130 <= codepoint <= 0x318F:
            has_korean = True
        elif "A" <= ch <= "Z" or "a" <= ch <= "z":
            has_english = True
    if has_korean and has_english:
        return "mixed_ko_en"
    if has_korean:
        return "pure_ko"
    if has_english:
        return "non_ko"
    return "other"


def _ensure_required(record: Mapping[str, Any], required: Iterable[str], *, lineno: int, path: Path) -> None:
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise DataValidationError(
            f"Missing required fields {missing} on line {lineno} of {path}"
        )


def example_from_record(record: Mapping[str, Any]) -> RewriteExample:
    """Create a :class:`RewriteExample` from a dictionary record.

    Callers must ensure required fields are present. This helper fills in
    defaults for optional fields and classifies ``query_type`` if missing.
    """

    question_ko = str(record["question_ko"])
    target_raw = record.get("target_query")
    target_query = str(target_raw) if target_raw not in (None, "") else None

    declared_type = record.get("query_type")
    if declared_type in _VALID_QUERY_TYPES:
        query_type: QueryType = declared_type  # type: ignore[assignment]
    else:
        query_type = classify_query_type(question_ko)

    metadata_raw = record.get("metadata") or {}
    if not isinstance(metadata_raw, Mapping):
        metadata_raw = {}

    return RewriteExample(
        question_ko=question_ko,
        positive_doc_id=str(record["positive_doc_id"]),
        negative_doc_id=str(record["negative_doc_id"]),
        target_query=target_query,
        example_id=str(record["example_id"]) if record.get("example_id") is not None else None,
        dataset_name=record.get("dataset_name"),
        split_name=record.get("split_name"),
        query_type=query_type,
        source_language=record.get("source_language"),
        metadata=dict(metadata_raw),
    )


def corpus_document_from_record(record: Mapping[str, Any]) -> CorpusDocument:
    """Create a :class:`CorpusDocument` from a dictionary record."""

    metadata_raw = record.get("metadata") or {}
    if not isinstance(metadata_raw, Mapping):
        metadata_raw = {}
    return CorpusDocument(
        doc_id=str(record["doc_id"]),
        text=str(record["text"]),
        source_language=record.get("source_language"),
        title=record.get("title"),
        metadata=dict(metadata_raw),
    )


def load_dataset(path: str | os.PathLike[str]) -> list[RewriteExample]:
    """Load a JSONL dataset file into a list of :class:`RewriteExample`."""

    file_path = Path(path)
    examples: list[RewriteExample] = []
    for lineno, record in _iter_jsonl(file_path):
        _ensure_required(record, _REQUIRED_EXAMPLE_FIELDS, lineno=lineno, path=file_path)
        examples.append(example_from_record(record))
    return examples


def load_corpus(path: str | os.PathLike[str]) -> list[CorpusDocument]:
    """Load a JSONL corpus file into a list of :class:`CorpusDocument`."""

    file_path = Path(path)
    docs: list[CorpusDocument] = []
    seen_ids: set[str] = set()
    for lineno, record in _iter_jsonl(file_path):
        _ensure_required(record, _REQUIRED_CORPUS_FIELDS, lineno=lineno, path=file_path)
        doc = corpus_document_from_record(record)
        if doc.doc_id in seen_ids:
            raise DataValidationError(
                f"Duplicate doc_id {doc.doc_id!r} at line {lineno} of {file_path}"
            )
        seen_ids.add(doc.doc_id)
        docs.append(doc)
    return docs


__all__ = [
    "DataValidationError",
    "QueryType",
    "RewriteExample",
    "CorpusDocument",
    "RetrievedDocument",
    "classify_query_type",
    "example_from_record",
    "corpus_document_from_record",
    "load_dataset",
    "load_corpus",
]
