"""Run BM25 + BGE-M3 + reranker citation retrieval for candidate search plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from crosslingual_rewrite.citation import CitationCandidateRecord  # noqa: E402
from crosslingual_rewrite.citation_retrieval import (  # noqa: E402
    CrossEncoderReranker,
    DenseRetriever,
    retrieve_citations_for_plan,
)
from crosslingual_rewrite.config import load_config, validate_config  # noqa: E402
from crosslingual_rewrite.data import load_corpus  # noqa: E402
from crosslingual_rewrite.retriever import BM25Retriever  # noqa: E402


def _iter_records(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield CitationCandidateRecord.from_dict(json.loads(stripped))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve citation candidates for search plans.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bm25-top-k", type=int, default=100)
    parser.add_argument("--dense-top-k", type=int, default=100)
    parser.add_argument("--fusion-top-k", type=int, default=50)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument("--dense-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--embedding-cache-dir", default="/opt/dlami/nvme/citation_index")
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    validate_config(cfg)
    corpus = load_corpus(cfg.data.corpus_path)
    records = list(_iter_records(Path(args.candidates)))
    if args.limit is not None and args.limit >= 0:
        records = records[: args.limit]

    bm25 = BM25Retriever(corpus)
    dense = None
    if not args.no_dense:
        dense = DenseRetriever(
            corpus,
            model_name=args.dense_model,
            cache_dir=args.embedding_cache_dir,
            batch_size=args.dense_batch_size,
        )
    reranker = None
    if not args.no_reranker:
        reranker = CrossEncoderReranker(model_name=args.reranker_model, batch_size=args.reranker_batch_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for index, record in enumerate(records, start=1):
            citations = retrieve_citations_for_plan(
                record.search_plan,
                corpus=corpus,
                bm25=bm25,
                dense=dense,
                reranker=reranker,
                bm25_top_k=args.bm25_top_k,
                dense_top_k=args.dense_top_k,
                fusion_top_k=args.fusion_top_k,
                final_top_k=args.final_top_k,
            )
            retrieved = CitationCandidateRecord(
                question_id=record.question_id,
                question=record.question,
                query_type=record.query_type,
                candidate_id=record.candidate_id,
                search_plan=record.search_plan,
                positive_doc_id=record.positive_doc_id,
                negative_doc_id=record.negative_doc_id,
                target_query=record.target_query,
                answers=record.answers,
                citations=citations,
                metadata=record.metadata,
            )
            out.write(json.dumps(retrieved.to_dict(), ensure_ascii=False) + "\n")
            if args.progress_every > 0 and (index == 1 or index == len(records) or index % args.progress_every == 0):
                pct = 100.0 * index / max(1, len(records))
                print(f"[citation_retrieval] progress {index}/{len(records)} ({pct:.1f}%)", flush=True)
    print(json.dumps({"output": str(output_path), "records": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
