"""Evaluation metrics over retrieved-document lists.

All metrics operate on a ``positive_doc_id`` plus a ranked list of
:class:`~crosslingual_rewrite.data.RetrievedDocument` objects and return a
float. The helper :func:`compute_metrics` bundles them into the dictionary
used throughout the pipeline.

The faithfulness metric is an explicit lexical proxy: we Jaccard-compare the
tokenized generated query against the tokenized target query. This is
clearly documented as a proxy in the README and final report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .data import RetrievedDocument
from .retriever import tokenize


_METRIC_KEYS: tuple[str, ...] = (
    "recall_at_k",
    "mrr",
    "ndcg",
    "source_diversity",
    "english_source_ratio",
    "korean_source_ratio",
    "faithfulness",
)


@dataclass(frozen=True)
class MetricBundle:
    """Convenience wrapper for the per-example metric dictionary."""

    recall_at_k: float
    mrr: float
    ndcg: float
    source_diversity: float
    english_source_ratio: float
    korean_source_ratio: float
    faithfulness: float

    def to_dict(self) -> dict[str, float]:
        return {
            "recall_at_k": float(self.recall_at_k),
            "mrr": float(self.mrr),
            "ndcg": float(self.ndcg),
            "source_diversity": float(self.source_diversity),
            "english_source_ratio": float(self.english_source_ratio),
            "korean_source_ratio": float(self.korean_source_ratio),
            "faithfulness": float(self.faithfulness),
        }


def _rank_of_positive(
    retrieved: Sequence[RetrievedDocument], positive_doc_id: str | None
) -> int | None:
    if positive_doc_id is None:
        return None
    for doc in retrieved:
        if doc.doc_id == positive_doc_id:
            return int(doc.rank)
    return None


def recall_at_k(
    retrieved: Sequence[RetrievedDocument],
    positive_doc_id: str,
    *,
    k: int,
) -> float:
    """Recall@k is 1.0 when the positive doc appears in the top-k."""

    if k <= 0:
        return 0.0
    for doc in retrieved[:k]:
        if doc.doc_id == positive_doc_id:
            return 1.0
    return 0.0


def mean_reciprocal_rank(
    retrieved: Sequence[RetrievedDocument],
    positive_doc_id: str,
) -> float:
    """MRR for a single query is ``1/rank`` of the positive doc, else 0."""

    rank = _rank_of_positive(retrieved, positive_doc_id)
    if rank is None:
        return 0.0
    return 1.0 / rank


def ndcg(
    retrieved: Sequence[RetrievedDocument],
    positive_doc_id: str,
    *,
    k: int | None = None,
) -> float:
    """Binary-relevance nDCG for a single query.

    With a single relevant document the ideal DCG is ``1.0``, so this reduces
    to ``1 / log2(rank + 1)`` when the positive doc is in the top-k and 0
    otherwise.
    """

    ranked = list(retrieved)
    if k is not None:
        ranked = ranked[:k]
    rank = _rank_of_positive(ranked, positive_doc_id)
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def source_diversity(retrieved: Sequence[RetrievedDocument]) -> float:
    """Fraction of distinct ``source_language`` values in the retrieved list."""

    if not retrieved:
        return 0.0
    unique: set[str] = set()
    for doc in retrieved:
        if doc.source_language is None:
            unique.add("__unknown__")
        else:
            unique.add(doc.source_language)
    return len(unique) / len(retrieved)


def _language_ratio(retrieved: Sequence[RetrievedDocument], target: str) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for doc in retrieved if (doc.source_language or "").lower() == target)
    return hits / len(retrieved)


def english_source_ratio(retrieved: Sequence[RetrievedDocument]) -> float:
    return _language_ratio(retrieved, "en")


def korean_source_ratio(retrieved: Sequence[RetrievedDocument]) -> float:
    return _language_ratio(retrieved, "ko")


def faithfulness_score(generated_query: str, target_query: str | None) -> float:
    """Lexical proxy for faithfulness.

    Computes a Jaccard similarity between the generated and target token
    sets. When no target query is available we return ``0.0``. The value is
    clamped to ``[0.0, 1.0]``.
    """

    if target_query is None or not target_query.strip():
        return 0.0
    generated = set(tokenize(generated_query or ""))
    target = set(tokenize(target_query))
    if not generated and not target:
        return 1.0
    if not generated or not target:
        return 0.0
    score = len(generated & target) / len(generated | target)
    return max(0.0, min(1.0, score))


def compute_metrics(
    retrieved: Sequence[RetrievedDocument],
    *,
    positive_doc_id: str,
    top_k: int,
    generated_query: str,
    target_query: str | None,
) -> MetricBundle:
    """Compute the canonical per-example metric bundle."""

    top = list(retrieved[:top_k])
    return MetricBundle(
        recall_at_k=recall_at_k(top, positive_doc_id, k=top_k),
        mrr=mean_reciprocal_rank(top, positive_doc_id),
        ndcg=ndcg(top, positive_doc_id, k=top_k),
        source_diversity=source_diversity(top),
        english_source_ratio=english_source_ratio(top),
        korean_source_ratio=korean_source_ratio(top),
        faithfulness=faithfulness_score(generated_query, target_query),
    )


def aggregate_metrics(per_example: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Average a list of per-example metric dictionaries.

    Missing keys are treated as zero so that every result row carries the same
    columns downstream.
    """

    per_example_list = list(per_example)
    if not per_example_list:
        return {key: 0.0 for key in _METRIC_KEYS}
    sums = {key: 0.0 for key in _METRIC_KEYS}
    for item in per_example_list:
        for key in _METRIC_KEYS:
            sums[key] += float(item.get(key, 0.0))
    count = len(per_example_list)
    return {key: sums[key] / count for key in _METRIC_KEYS}


def metric_keys() -> tuple[str, ...]:
    return _METRIC_KEYS


__all__ = [
    "MetricBundle",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg",
    "source_diversity",
    "english_source_ratio",
    "korean_source_ratio",
    "faithfulness_score",
    "compute_metrics",
    "aggregate_metrics",
    "metric_keys",
]
