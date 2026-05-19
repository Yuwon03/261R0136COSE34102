"""Citation-aware search-plan data structures and scoring utilities."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SUPPORT_LABELS: tuple[str, ...] = (
    "supported",
    "partial",
    "unsupported",
    "contradicted",
    "unlabeled",
)

SUPPORT_VALUES: dict[str, float] = {
    "supported": 1.0,
    "partial": 0.5,
    "unlabeled": 0.0,
    "unsupported": 0.0,
    "contradicted": -1.0,
}

ANSWER_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    ("언제", "date"),
    ("몇", "number"),
    ("누구", "person"),
    ("어디", "place"),
    ("when", "date"),
    ("how many", "number"),
    ("who", "person"),
    ("where", "place"),
    ("what", "definition"),
)

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")
_HANGUL_RE = re.compile(r"[가-힣]{2,}")


@dataclass(frozen=True)
class SearchPlan:
    """A citation-seeking search strategy produced from one question."""

    method: str
    queries: list[str]
    entities: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    answer_type: str = "unknown"
    preferred_source_languages: list[str] = field(default_factory=lambda: ["en", "ko"])
    source_priority: list[str] = field(default_factory=lambda: ["encyclopedic", "official", "news"])
    metadata: dict[str, Any] = field(default_factory=dict)

    def primary_query(self) -> str:
        for query in self.queries:
            cleaned = query.strip()
            if cleaned:
                return cleaned
        return ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchPlan":
        method = str(payload.get("method") or "unknown")
        raw_queries = payload.get("queries") or []
        if isinstance(raw_queries, str):
            raw_queries = [raw_queries]
        queries = _dedupe_keep_order(str(item).strip() for item in raw_queries if str(item).strip())
        if not queries:
            query = str(payload.get("query") or "").strip()
            queries = [query] if query else []
        return cls(
            method=method,
            queries=queries,
            entities=_as_string_list(payload.get("entities")),
            aliases=_as_string_list(payload.get("aliases")),
            answer_type=str(payload.get("answer_type") or "unknown"),
            preferred_source_languages=_as_string_list(
                payload.get("preferred_source_languages"), default=["en", "ko"]
            ),
            source_priority=_as_string_list(
                payload.get("source_priority"), default=["encyclopedic", "official", "news"]
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class CitationCandidate:
    """One retrieved citation candidate for a search plan."""

    doc_id: str
    chunk_id: str
    title: str | None
    url: str | None
    language: str | None
    text: str
    rank: int
    retriever_scores: dict[str, float] = field(default_factory=dict)
    rerank_score: float | None = None
    support_label: str = "unlabeled"
    support_score: float = 0.0
    source_quality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CitationCandidate":
        return cls(
            doc_id=str(payload.get("doc_id") or ""),
            chunk_id=str(payload.get("chunk_id") or payload.get("doc_id") or ""),
            title=str(payload["title"]) if payload.get("title") is not None else None,
            url=str(payload["url"]) if payload.get("url") is not None else None,
            language=str(payload["language"]) if payload.get("language") is not None else None,
            text=str(payload.get("text") or ""),
            rank=int(payload.get("rank") or 0),
            retriever_scores={k: float(v) for k, v in dict(payload.get("retriever_scores") or {}).items()},
            rerank_score=float(payload["rerank_score"]) if payload.get("rerank_score") is not None else None,
            support_label=normalize_support_label(str(payload.get("support_label") or "unlabeled")),
            support_score=float(payload.get("support_score") or 0.0),
            source_quality=float(payload.get("source_quality") or 0.0),
        )


@dataclass(frozen=True)
class CitationCandidateRecord:
    """One candidate search plan and its retrieved citations for a question."""

    question_id: str
    question: str
    query_type: str
    candidate_id: str
    search_plan: SearchPlan
    positive_doc_id: str | None = None
    negative_doc_id: str | None = None
    target_query: str | None = None
    answers: list[str] = field(default_factory=list)
    citations: list[CitationCandidate] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    candidate_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["search_plan"] = self.search_plan.to_dict()
        payload["citations"] = [citation.to_dict() for citation in self.citations]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CitationCandidateRecord":
        return cls(
            question_id=str(payload.get("question_id") or payload.get("example_id") or ""),
            question=str(payload.get("question") or payload.get("question_ko") or ""),
            query_type=str(payload.get("query_type") or "other"),
            candidate_id=str(payload.get("candidate_id") or ""),
            search_plan=SearchPlan.from_dict(payload.get("search_plan") or {}),
            positive_doc_id=str(payload["positive_doc_id"]) if payload.get("positive_doc_id") else None,
            negative_doc_id=str(payload["negative_doc_id"]) if payload.get("negative_doc_id") else None,
            target_query=str(payload["target_query"]) if payload.get("target_query") else None,
            answers=_as_string_list(payload.get("answers")),
            citations=[CitationCandidate.from_dict(item) for item in payload.get("citations") or []],
            metrics={k: float(v) for k, v in dict(payload.get("metrics") or {}).items()},
            candidate_score=float(payload.get("candidate_score") or 0.0),
            metadata=dict(payload.get("metadata") or {}),
        )


def _as_string_list(value: Any, *, default: Sequence[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [value] if value.strip() else list(default or [])
    if isinstance(value, Sequence):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned if cleaned else list(default or [])
    return list(default or [])


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return dest


def normalize_support_label(label: str) -> str:
    cleaned = label.strip().lower().replace("-", "_")
    aliases = {
        "good": "supported",
        "bad": "unsupported",
        "yes": "supported",
        "no": "unsupported",
        "support": "supported",
        "supports": "supported",
        "partially_supported": "partial",
        "partially": "partial",
        "contradict": "contradicted",
        "contradicts": "contradicted",
    }
    cleaned = aliases.get(cleaned, cleaned)
    return cleaned if cleaned in SUPPORT_LABELS else "unlabeled"


def answers_from_metadata(metadata: Mapping[str, Any]) -> list[str]:
    answers = metadata.get("answers")
    if isinstance(answers, str):
        return [answers] if answers.strip() else []
    if isinstance(answers, Sequence):
        return _dedupe_keep_order(str(answer).strip() for answer in answers if str(answer).strip())
    return []


def infer_answer_type(question: str) -> str:
    lower = question.lower()
    for hint, answer_type in ANSWER_TYPE_HINTS:
        if hint in lower or hint in question:
            return answer_type
    return "unknown"


def extract_surface_entities(text: str, *, max_entities: int = 8) -> list[str]:
    english = [token for token in _WORD_RE.findall(text) if len(token) > 2]
    korean = _HANGUL_RE.findall(text)
    return _dedupe_keep_order([*english, *korean])[:max_entities]


def compact_query_tokens(*texts: str | None, max_tokens: int = 32) -> str:
    tokens: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in _WORD_RE.finditer(text):
            token = match.group(0).strip("'").lower()
            if len(token) <= 1 or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) >= max_tokens:
                return " ".join(tokens)
    return " ".join(tokens)


def make_search_plan(
    *,
    method: str,
    question: str,
    queries: Sequence[str],
    target_query: str | None = None,
    answers: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SearchPlan:
    entities = extract_surface_entities(" ".join([question, target_query or "", " ".join(answers or [])]))
    return SearchPlan(
        method=method,
        queries=_dedupe_keep_order(queries),
        entities=entities,
        aliases=[],
        answer_type=infer_answer_type(question),
        preferred_source_languages=["en", "ko"],
        source_priority=["encyclopedic", "official", "news"],
        metadata=dict(metadata or {}),
    )


def support_score_from_label(label: str) -> float:
    return SUPPORT_VALUES.get(normalize_support_label(label), 0.0)


def source_quality_score(citation: CitationCandidate) -> float:
    score = 0.0
    if citation.title:
        score += 0.2
    if citation.url:
        score += 0.2
    if (citation.language or "").lower() == "en":
        score += 0.2
    if citation.rerank_score is not None:
        score += min(0.4, max(0.0, citation.rerank_score) / 10.0)
    return min(1.0, score)


def answer_containment_score(text: str, answers: Sequence[str]) -> float:
    if not answers:
        return 0.0
    lower_text = text.lower()
    hits = 0
    for answer in answers:
        cleaned = str(answer).strip().lower()
        if cleaned and cleaned in lower_text:
            hits += 1
    return hits / max(1, len(answers))


def label_citation_with_heuristics(
    citation: CitationCandidate,
    *,
    positive_doc_id: str | None,
    negative_doc_id: str | None,
    answers: Sequence[str],
) -> CitationCandidate:
    if citation.support_label != "unlabeled":
        return citation
    label = "unlabeled"
    score = 0.0
    if positive_doc_id and citation.doc_id == positive_doc_id:
        label = "supported"
        score = 1.0
    elif negative_doc_id and citation.doc_id == negative_doc_id:
        label = "unsupported"
        score = 0.0
    else:
        containment = answer_containment_score(citation.text, answers)
        if containment >= 0.75:
            label = "supported"
            score = containment
        elif containment > 0:
            label = "partial"
            score = containment
    return CitationCandidate(
        doc_id=citation.doc_id,
        chunk_id=citation.chunk_id,
        title=citation.title,
        url=citation.url,
        language=citation.language,
        text=citation.text,
        rank=citation.rank,
        retriever_scores=dict(citation.retriever_scores),
        rerank_score=citation.rerank_score,
        support_label=label,
        support_score=score,
        source_quality=source_quality_score(citation),
    )


def compute_citation_metrics(
    citations: Sequence[CitationCandidate],
    *,
    positive_doc_id: str | None,
    top_k_values: Sequence[int] = (5, 10, 20),
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    ranked = sorted(citations, key=lambda citation: citation.rank)
    for k in top_k_values:
        top = ranked[:k]
        metrics[f"recall_at_{k}"] = (
            1.0
            if positive_doc_id and any(citation.doc_id == positive_doc_id for citation in top)
            else 0.0
        )
    positive_rank = None
    if positive_doc_id:
        for citation in ranked:
            if citation.doc_id == positive_doc_id:
                positive_rank = citation.rank
                break
    metrics["mrr"] = 1.0 / positive_rank if positive_rank else 0.0
    metrics["ndcg_at_10"] = 1.0 / math.log2(positive_rank + 1) if positive_rank and positive_rank <= 10 else 0.0

    labels = [normalize_support_label(citation.support_label) for citation in ranked]
    supported = sum(1 for label in labels if label == "supported")
    partial = sum(1 for label in labels if label == "partial")
    contradicted = sum(1 for label in labels if label == "contradicted")
    judged = sum(1 for label in labels if label != "unlabeled")
    support_credit = supported + 0.5 * partial
    metrics["citation_precision"] = support_credit / max(1, judged)
    metrics["citation_recall"] = 1.0 if support_credit > 0 else 0.0
    precision = metrics["citation_precision"]
    recall = metrics["citation_recall"]
    metrics["citation_f1"] = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    metrics["unsupported_claim_rate"] = sum(1 for label in labels if label == "unsupported") / max(1, judged)
    metrics["contradicted_claim_rate"] = contradicted / max(1, judged)

    en = [citation for citation in ranked if (citation.language or "").lower() == "en"]
    ko = [citation for citation in ranked if (citation.language or "").lower() == "ko"]
    useful_en = [citation for citation in en if normalize_support_label(citation.support_label) in {"supported", "partial"}]
    useful_non_ko = [
        citation
        for citation in ranked
        if (citation.language or "").lower() != "ko"
        and normalize_support_label(citation.support_label) in {"supported", "partial"}
    ]
    metrics["english_citation_ratio"] = len(en) / max(1, len(ranked))
    metrics["korean_citation_ratio"] = len(ko) / max(1, len(ranked))
    metrics["useful_english_citation_ratio"] = len(useful_en) / max(1, len(ranked))
    metrics["cross_lingual_success_rate"] = 1.0 if useful_non_ko else 0.0
    metrics["context_precision"] = metrics["citation_precision"]
    metrics["context_recall"] = metrics["recall_at_20"]
    metrics["answer_faithfulness"] = metrics["citation_precision"]
    metrics["answer_relevance"] = 1.0 if support_credit > 0 else 0.0
    metrics["answer_contains_gold_answer"] = 1.0 if any(citation.support_score > 0 for citation in ranked) else 0.0
    return metrics


def aggregate_metric_dicts(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    rows_list = list(rows)
    if not rows_list:
        return {}
    keys = sorted({key for row in rows_list for key in row})
    return {key: sum(float(row.get(key, 0.0)) for row in rows_list) / len(rows_list) for key in keys}


def score_candidate_record(record: CitationCandidateRecord) -> CitationCandidateRecord:
    labeled = [
        label_citation_with_heuristics(
            citation,
            positive_doc_id=record.positive_doc_id,
            negative_doc_id=record.negative_doc_id,
            answers=record.answers,
        )
        for citation in record.citations
    ]
    metrics = compute_citation_metrics(labeled, positive_doc_id=record.positive_doc_id)
    candidate_score = (
        0.30 * metrics.get("recall_at_10", 0.0)
        + 0.20 * metrics.get("mrr", 0.0)
        + 0.20 * metrics.get("citation_precision", 0.0)
        + 0.15 * metrics.get("citation_recall", 0.0)
        + 0.10 * metrics.get("useful_english_citation_ratio", 0.0)
        + 0.05 * metrics.get("answer_faithfulness", 0.0)
    )
    return CitationCandidateRecord(
        question_id=record.question_id,
        question=record.question,
        query_type=record.query_type,
        candidate_id=record.candidate_id,
        search_plan=record.search_plan,
        positive_doc_id=record.positive_doc_id,
        negative_doc_id=record.negative_doc_id,
        target_query=record.target_query,
        answers=list(record.answers),
        citations=labeled,
        metrics=metrics,
        candidate_score=candidate_score,
        metadata=dict(record.metadata),
    )


__all__ = [
    "SUPPORT_LABELS",
    "SearchPlan",
    "CitationCandidate",
    "CitationCandidateRecord",
    "answers_from_metadata",
    "compact_query_tokens",
    "make_search_plan",
    "normalize_support_label",
    "read_jsonl",
    "write_jsonl",
    "support_score_from_label",
    "label_citation_with_heuristics",
    "compute_citation_metrics",
    "aggregate_metric_dicts",
    "score_candidate_record",
]
