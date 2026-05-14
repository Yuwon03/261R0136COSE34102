"""Build training JSONL examples from Korean QA datasets.

The builder turns Korean questions from MKQA/XOR-style records into the
project's ``RewriteExample`` JSONL format. It uses the fixed BM25 retriever to
create candidate documents, then labels positives/negatives with explicit
answer-containment checks so retriever mistakes do not silently become labels.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from .data import CorpusDocument, classify_query_type, load_corpus
from .retriever import BM25Retriever


SourceName = Literal["mkqa", "xor_tydi"]

_XOR_LANG_LABELS: tuple[str, ...] = ("ar", "bn", "fi", "ja", "ko", "ru", "te")


@dataclass(frozen=True)
class SourceQuestion:
    """Normalized Korean QA item before document labeling."""

    source: SourceName
    source_id: str
    question_ko: str
    target_query: str
    answers: tuple[str, ...]
    split_name: str = "train"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuiltExample:
    """One generated dataset row plus candidate provenance."""

    row: dict[str, Any]
    positive_candidates: tuple[str, ...]
    negative_candidates: tuple[str, ...]


@dataclass
class BuildStats:
    """Counters returned by dataset-building helpers."""

    source_seen: int = 0
    normalized: int = 0
    written: int = 0
    skipped_no_korean_question: int = 0
    skipped_no_target_query: int = 0
    skipped_no_answers: int = 0
    skipped_no_positive: int = 0
    skipped_no_negative: int = 0
    skipped_duplicate_question: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


_HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")
_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^0-9a-z가-힣]+")


def has_korean(text: str) -> bool:
    return bool(_HANGUL_RE.search(text or ""))


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return _SPACE_RE.sub(" ", str(value)).strip()


def _normalize_for_match(text: str) -> str:
    lowered = _clean_text(text).lower()
    return _NON_WORD_RE.sub(" ", lowered)


def _dedupe_keep_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for raw in values:
        value = _clean_text(raw)
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        output.append(value)
    return tuple(output)


def _stable_suffix(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _answer_texts_from_mkqa(raw_answers: Any) -> tuple[str, ...]:
    values: list[str] = []
    if not isinstance(raw_answers, Sequence) or isinstance(raw_answers, (str, bytes)):
        return ()
    for answer in raw_answers:
        if not isinstance(answer, Mapping):
            values.append(_clean_text(answer))
            continue
        answer_type = _clean_text(answer.get("type")).lower()
        if answer_type == "unanswerable":
            continue
        values.append(_clean_text(answer.get("text")))
        aliases = answer.get("aliases") or []
        if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
            values.extend(_clean_text(alias) for alias in aliases)
    return _dedupe_keep_order(values)


def _answer_texts_from_xor(raw_answers: Any) -> tuple[str, ...]:
    if isinstance(raw_answers, str):
        return _dedupe_keep_order([raw_answers])
    if isinstance(raw_answers, Sequence) and not isinstance(raw_answers, (str, bytes)):
        return _dedupe_keep_order(_clean_text(item) for item in raw_answers)
    return _dedupe_keep_order([_clean_text(raw_answers)])


def _mkqa_source_question(record: Mapping[str, Any], *, split_name: str) -> SourceQuestion | None:
    queries = record.get("queries") or {}
    if not isinstance(queries, Mapping):
        return None
    question_ko = _clean_text(queries.get("ko"))
    target_query = _clean_text(record.get("query") or queries.get("en"))
    answers_by_lang = record.get("answers") or {}
    if not isinstance(answers_by_lang, Mapping):
        answers_by_lang = {}
    answers = _answer_texts_from_mkqa(answers_by_lang.get("en") or answers_by_lang.get("ko"))
    source_id = _clean_text(record.get("example_id") or record.get("id"))
    return SourceQuestion(
        source="mkqa",
        source_id=source_id or f"mkqa-{_stable_suffix(question_ko)}",
        question_ko=question_ko,
        target_query=target_query,
        answers=answers,
        split_name=split_name,
        metadata={
            "answer_count": len(answers),
            "original_query": _clean_text(record.get("query")),
        },
    )


def _xor_target_query(record: Mapping[str, Any]) -> str:
    for key in ("target_query", "question_en", "query_en", "english_question", "translated_question"):
        value = _clean_text(record.get(key))
        if value:
            return value
    return ""


def _xor_source_question(record: Mapping[str, Any], *, split_name: str) -> SourceQuestion | None:
    lang_raw = record.get("lang") if record.get("lang") is not None else record.get("language")
    if isinstance(lang_raw, int) and 0 <= lang_raw < len(_XOR_LANG_LABELS):
        lang = _XOR_LANG_LABELS[lang_raw]
    else:
        lang = _clean_text(lang_raw).lower()
    if lang and lang != "ko":
        return None
    question_ko = _clean_text(record.get("question") or record.get("query"))
    target_query = _xor_target_query(record)
    answers = _answer_texts_from_xor(record.get("answers") or record.get("answer"))
    source_id = _clean_text(record.get("id") or record.get("example_id"))
    return SourceQuestion(
        source="xor_tydi",
        source_id=source_id or f"xor-tydi-{_stable_suffix(question_ko)}",
        question_ko=question_ko,
        target_query=target_query,
        answers=answers,
        split_name=_clean_text(record.get("split") or split_name) or split_name,
        metadata={
            "answer_count": len(answers),
            "requires_external_target_query": not bool(target_query),
        },
    )


def normalize_source_record(
    record: Mapping[str, Any],
    *,
    source: SourceName,
    split_name: str = "train",
) -> SourceQuestion | None:
    """Convert a raw MKQA/XOR record into a normalized Korean question."""

    if source == "mkqa":
        return _mkqa_source_question(record, split_name=split_name)
    if source == "xor_tydi":
        return _xor_source_question(record, split_name=split_name)
    raise ValueError(f"Unsupported source: {source!r}")


def doc_contains_answer(doc: CorpusDocument, answers: Sequence[str]) -> bool:
    """Return true when any normalized answer string appears in title/text."""

    haystack = _normalize_for_match(" ".join(filter(None, [doc.title, doc.text])))
    if not haystack:
        return False
    for answer in answers:
        needle = _normalize_for_match(answer)
        if not needle:
            continue
        if needle in haystack:
            return True
    return False


def build_example_from_source(
    source_question: SourceQuestion,
    *,
    retriever: BM25Retriever,
    documents_by_id: Mapping[str, CorpusDocument],
    top_k_positive: int = 100,
    top_k_negative: int = 20,
) -> BuiltExample | None:
    """Create one project-schema row from a normalized source question."""

    retrieved = retriever.retrieve(source_question.target_query, top_k=max(top_k_positive, top_k_negative))
    positive_ids: list[str] = []
    negative_ids: list[str] = []

    for hit in retrieved[:top_k_positive]:
        doc = documents_by_id.get(hit.doc_id)
        if doc is not None and doc_contains_answer(doc, source_question.answers):
            positive_ids.append(hit.doc_id)

    positive_set = set(positive_ids)
    for hit in retrieved[:top_k_negative]:
        if hit.doc_id in positive_set:
            continue
        doc = documents_by_id.get(hit.doc_id)
        if doc is None:
            continue
        if doc_contains_answer(doc, source_question.answers):
            continue
        negative_ids.append(hit.doc_id)

    if not positive_ids or not negative_ids:
        return None

    example_id = f"{source_question.source}-{source_question.source_id}"
    row: dict[str, Any] = {
        "example_id": example_id,
        "question_ko": source_question.question_ko,
        "target_query": source_question.target_query,
        "positive_doc_id": positive_ids[0],
        "negative_doc_id": negative_ids[0],
        "dataset_name": source_question.source,
        "split_name": source_question.split_name,
        "query_type": classify_query_type(source_question.question_ko),
        "metadata": {
            **dict(source_question.metadata),
            "source_id": source_question.source_id,
            "answers": list(source_question.answers),
            "positive_candidates": positive_ids,
            "negative_candidates": negative_ids,
            "labeling_rule": "retriever_candidates_answer_containment",
        },
    }
    return BuiltExample(
        row=row,
        positive_candidates=tuple(positive_ids),
        negative_candidates=tuple(negative_ids),
    )


def iter_jsonl_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if isinstance(record, dict):
                yield record


def build_dataset_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    source: SourceName,
    corpus: Sequence[CorpusDocument],
    output_path: str | Path,
    split_name: str = "train",
    top_k_positive: int = 100,
    top_k_negative: int = 20,
    limit: int | None = None,
    dedupe_questions: bool = True,
) -> BuildStats:
    """Build and write a dataset JSONL file from source records."""

    stats = BuildStats()
    retriever = BM25Retriever(corpus)
    documents_by_id = {doc.doc_id: doc for doc in corpus}
    seen_questions: set[str] = set()
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with dest.open("w", encoding="utf-8") as fh:
        for raw in records:
            if limit is not None and stats.source_seen >= limit:
                break
            stats.source_seen += 1
            source_question = normalize_source_record(raw, source=source, split_name=split_name)
            if source_question is None or not has_korean(source_question.question_ko):
                stats.skipped_no_korean_question += 1
                continue
            if dedupe_questions:
                dedupe_key = _normalize_for_match(source_question.question_ko)
                if dedupe_key in seen_questions:
                    stats.skipped_duplicate_question += 1
                    continue
                seen_questions.add(dedupe_key)
            if not source_question.target_query:
                stats.skipped_no_target_query += 1
                continue
            if not source_question.answers:
                stats.skipped_no_answers += 1
                continue
            stats.normalized += 1
            built = build_example_from_source(
                source_question,
                retriever=retriever,
                documents_by_id=documents_by_id,
                top_k_positive=top_k_positive,
                top_k_negative=top_k_negative,
            )
            if built is None:
                retrieved = retriever.retrieve(source_question.target_query, top_k=top_k_positive)
                has_positive = any(
                    doc_contains_answer(documents_by_id[hit.doc_id], source_question.answers)
                    for hit in retrieved
                    if hit.doc_id in documents_by_id
                )
                if has_positive:
                    stats.skipped_no_negative += 1
                else:
                    stats.skipped_no_positive += 1
                continue
            fh.write(json.dumps(built.row, ensure_ascii=False) + "\n")
            stats.written += 1
    return stats


def load_hf_records(
    source: SourceName,
    *,
    split: str = "train",
    dataset_name: str | None = None,
    config_name: str | None = None,
    trust_remote_code: bool = False,
) -> Iterable[Mapping[str, Any]]:
    """Load records from HuggingFace datasets.

    This function is imported lazily so local tests and offline smoke runs do
    not require network access.
    """

    from datasets import load_dataset

    if dataset_name is None:
        dataset_name = "apple/mkqa" if source == "mkqa" else "akariasai/xor_tydi_qa"
    if config_name:
        dataset = load_dataset(
            dataset_name,
            config_name,
            split=split,
            trust_remote_code=trust_remote_code,
        )
    else:
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=trust_remote_code)
    for record in dataset:
        if isinstance(record, Mapping):
            yield record


def build_dataset_from_hf(
    *,
    source: SourceName,
    corpus_path: str | Path,
    output_path: str | Path,
    split: str = "train",
    dataset_name: str | None = None,
    config_name: str | None = None,
    trust_remote_code: bool = False,
    top_k_positive: int = 100,
    top_k_negative: int = 20,
    limit: int | None = None,
) -> BuildStats:
    """Load a source dataset from HuggingFace and write project JSONL."""

    corpus = load_corpus(corpus_path)
    records = load_hf_records(
        source,
        split=split,
        dataset_name=dataset_name,
        config_name=config_name,
        trust_remote_code=trust_remote_code,
    )
    return build_dataset_from_records(
        records,
        source=source,
        corpus=corpus,
        output_path=output_path,
        split_name=split,
        top_k_positive=top_k_positive,
        top_k_negative=top_k_negative,
        limit=limit,
    )


__all__ = [
    "SourceName",
    "SourceQuestion",
    "BuiltExample",
    "BuildStats",
    "has_korean",
    "normalize_source_record",
    "doc_contains_answer",
    "build_example_from_source",
    "iter_jsonl_records",
    "build_dataset_from_records",
    "load_hf_records",
    "build_dataset_from_hf",
]
