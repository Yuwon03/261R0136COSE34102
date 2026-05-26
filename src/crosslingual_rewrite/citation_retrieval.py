"""Hybrid citation retrieval over search plans."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .citation import CitationCandidate, SearchPlan
from .data import CorpusDocument
from .retriever import BM25Retriever


@dataclass(frozen=True)
class FusedHit:
    doc: CorpusDocument
    rank: int
    scores: dict[str, float]
    fused_score: float


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[tuple[CorpusDocument, float]]],
    *,
    k: int = 60,
    top_k: int = 100,
) -> list[FusedHit]:
    """Fuse ranked document lists with deterministic reciprocal-rank fusion."""

    by_doc_id: dict[str, tuple[CorpusDocument, dict[str, float], float]] = {}
    for source_name, ranked in ranked_lists.items():
        for rank, (doc, score) in enumerate(ranked, start=1):
            current_doc, scores, fused = by_doc_id.get(doc.doc_id, (doc, {}, 0.0))
            scores[source_name] = float(score)
            fused += 1.0 / (k + rank)
            by_doc_id[doc.doc_id] = (current_doc, scores, fused)
    ordered = sorted(
        by_doc_id.values(),
        key=lambda item: (-item[2], item[0].doc_id),
    )[:top_k]
    return [
        FusedHit(doc=doc, rank=index, scores=scores, fused_score=fused)
        for index, (doc, scores, fused) in enumerate(ordered, start=1)
    ]


class DenseRetriever:
    """SentenceTransformer-based dense retriever with optional embedding cache."""

    def __init__(
        self,
        documents: Sequence[CorpusDocument],
        *,
        model_name: str = "BAAI/bge-m3",
        cache_dir: str | Path | None = None,
        batch_size: int = 16,
        max_seq_length: int = 256,
    ) -> None:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if sys.platform == "darwin":
            os.environ.setdefault("PYTORCH_SDP_DISABLE_FLASH_ATTENTION", "1")
            os.environ.setdefault("PYTORCH_SDP_DISABLE_MEM_EFFICIENT_ATTENTION", "1")
        self._np = np
        self._docs = list(documents)
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._model.max_seq_length = int(max_seq_length)
        self._batch_size = batch_size
        self._doc_ids = [doc.doc_id for doc in self._docs]
        cache_path = None
        if cache_dir is not None:
            cache_path = Path(cache_dir) / self._cache_name(model_name, len(self._docs))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path is not None and cache_path.exists():
            self._embeddings = np.load(cache_path)
        else:
            texts = [" ".join(part for part in (doc.title, doc.text) if part) for doc in self._docs]
            self._embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            if cache_path is not None:
                np.save(cache_path, self._embeddings)

    @staticmethod
    def _cache_name(model_name: str, doc_count: int) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in model_name.lower())
        return f"{cleaned}_{doc_count}_embeddings.npy"

    def retrieve(self, query: str, *, top_k: int) -> list[tuple[CorpusDocument, float]]:
        import numpy as np

        query_embedding = self._model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self._embeddings @ query_embedding
        top_indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        ordered = sorted(top_indices, key=lambda index: (-float(scores[index]), self._docs[int(index)].doc_id))
        return [(self._docs[int(index)], float(scores[index])) for index in ordered[:top_k]]


class CrossEncoderReranker:
    """CrossEncoder reranker wrapper."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, max_length=max_length)
        self._batch_size = batch_size

    def rerank(
        self,
        query: str,
        docs: Sequence[CorpusDocument],
        *,
        top_k: int,
    ) -> list[tuple[CorpusDocument, float]]:
        pairs = [[query, " ".join(part for part in (doc.title, doc.text) if part)] for doc in docs]
        scores = self._model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        ranked = sorted(zip(docs, scores), key=lambda item: (-float(item[1]), item[0].doc_id))
        return [(doc, float(score)) for doc, score in ranked[:top_k]]


def _bm25_ranked(
    retriever: BM25Retriever,
    docs_by_id: Mapping[str, CorpusDocument],
    query: str,
    *,
    top_k: int,
) -> list[tuple[CorpusDocument, float]]:
    retrieved = retriever.retrieve(query, top_k=top_k)
    ranked: list[tuple[CorpusDocument, float]] = []
    for hit in retrieved:
        doc = docs_by_id.get(hit.doc_id)
        if doc is not None:
            ranked.append((doc, float(hit.score)))
    return ranked


def retrieve_citations_for_plan(
    plan: SearchPlan,
    *,
    corpus: Sequence[CorpusDocument],
    bm25: BM25Retriever,
    dense: DenseRetriever | None = None,
    reranker: CrossEncoderReranker | None = None,
    bm25_top_k: int = 100,
    dense_top_k: int = 100,
    fusion_top_k: int = 50,
    final_top_k: int = 10,
) -> list[CitationCandidate]:
    """Retrieve and rerank citation candidates for a search plan."""

    docs_by_id = {doc.doc_id: doc for doc in corpus}
    per_query_hits: dict[str, list[tuple[CorpusDocument, float]]] = {}
    for query_index, query in enumerate(plan.queries, start=1):
        if not query.strip():
            continue
        per_query_hits[f"bm25_q{query_index}"] = _bm25_ranked(
            bm25,
            docs_by_id,
            query,
            top_k=bm25_top_k,
        )
        if dense is not None:
            per_query_hits[f"dense_q{query_index}"] = dense.retrieve(query, top_k=dense_top_k)
    fused = reciprocal_rank_fusion(per_query_hits, top_k=fusion_top_k)
    fused_docs = [hit.doc for hit in fused]
    fused_scores_by_id = {hit.doc.doc_id: {"rrf": hit.fused_score, **hit.scores} for hit in fused}
    primary_query = plan.primary_query()
    if reranker is not None and fused_docs:
        reranked = reranker.rerank(primary_query, fused_docs, top_k=final_top_k)
    else:
        reranked = [(hit.doc, hit.fused_score) for hit in fused[:final_top_k]]

    citations: list[CitationCandidate] = []
    for rank, (doc, rerank_score) in enumerate(reranked, start=1):
        metadata = dict(doc.metadata or {})
        url = metadata.get("url") or metadata.get("source_url")
        citations.append(
            CitationCandidate(
                doc_id=doc.doc_id,
                chunk_id=str(metadata.get("chunk_id") or doc.doc_id),
                title=doc.title,
                url=str(url) if url else None,
                language=doc.source_language,
                text=doc.text,
                rank=rank,
                retriever_scores=fused_scores_by_id.get(doc.doc_id, {}),
                rerank_score=float(rerank_score),
            )
        )
    return citations


__all__ = [
    "FusedHit",
    "DenseRetriever",
    "CrossEncoderReranker",
    "reciprocal_rank_fusion",
    "retrieve_citations_for_plan",
]
