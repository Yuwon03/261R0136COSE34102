"""Score citation candidates using human labels and automatic support proxies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from crosslingual_rewrite.citation import (  # noqa: E402
    CitationCandidate,
    CitationCandidateRecord,
    normalize_support_label,
    score_candidate_record,
)


def _load_label_file(path: str | None) -> dict[tuple[str, str], str]:
    if not path:
        return {}
    labels_path = Path(path)
    if not labels_path.exists():
        return {}
    labels: dict[tuple[str, str], str] = {}
    with labels_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("candidate_id")), str(row.get("chunk_id") or row.get("doc_id")))
            labels[key] = normalize_support_label(str(row.get("label") or "unlabeled"))
    return labels


def _load_labels(human_path: str | None, ai_path: str | None) -> dict[tuple[str, str], str]:
    labels = _load_label_file(ai_path)
    labels.update(_load_label_file(human_path))
    return labels


def _apply_labels(record: CitationCandidateRecord, labels: dict[tuple[str, str], str]) -> CitationCandidateRecord:
    citations: list[CitationCandidate] = []
    for citation in record.citations:
        key = (record.candidate_id, citation.chunk_id)
        label = labels.get(key)
        if label is None:
            key = (record.candidate_id, citation.doc_id)
            label = labels.get(key)
        if label is None:
            citations.append(citation)
            continue
        citations.append(
            CitationCandidate(
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
                support_score=1.0 if label == "supported" else 0.5 if label == "partial" else 0.0,
                source_quality=citation.source_quality,
            )
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
        answers=record.answers,
        citations=citations,
        metadata=record.metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score retrieved citation candidates.")
    parser.add_argument("--input", required=True, help="Retrieved candidate JSONL.")
    parser.add_argument("--output", required=True, help="Scored candidate JSONL.")
    parser.add_argument("--summary-output", required=True, help="Method summary CSV.")
    parser.add_argument("--labels", default=None, help="Optional human label JSONL.")
    parser.add_argument("--ai-labels", default=None, help="Optional AI label JSONL. Human labels override AI labels.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = _load_labels(args.labels, args.ai_labels)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_method_count: dict[str, int] = {}
    scores_by_method: dict[str, float] = {}
    metric_sums_by_method: dict[str, dict[str, float]] = {}
    metric_names: set[str] = set()
    total_records = 0

    with output_path.open("w", encoding="utf-8") as out:
        with Path(args.input).open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = CitationCandidateRecord.from_dict(json.loads(line))
                record = score_candidate_record(_apply_labels(record, labels))
                method = record.search_plan.method
                by_method_count[method] = by_method_count.get(method, 0) + 1
                scores_by_method[method] = scores_by_method.get(method, 0.0) + record.candidate_score
                metric_sums = metric_sums_by_method.setdefault(method, {})
                for name, value in record.metrics.items():
                    metric_names.add(name)
                    metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
                total_records += 1
                out.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "candidate_count", "candidate_score", *sorted(metric_names)])
        writer.writeheader()
        for method in sorted(by_method_count):
            count = by_method_count[method]
            metric_sums = metric_sums_by_method.get(method, {})
            writer.writerow(
                {
                    "method": method,
                    "candidate_count": count,
                    "candidate_score": scores_by_method.get(method, 0.0) / max(1, count),
                    **{name: metric_sums.get(name, 0.0) / max(1, count) for name in sorted(metric_names)},
                }
            )
    print(json.dumps({"output": str(output_path), "summary_output": str(summary_path), "records": total_records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
