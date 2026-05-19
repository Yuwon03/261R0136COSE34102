"""Create citation-aware comparison artifacts from scored candidate files."""

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

from crosslingual_rewrite.citation import CitationCandidateRecord, aggregate_metric_dicts  # noqa: E402


def _load_records(paths: list[str]) -> list[CitationCandidateRecord]:
    records: list[CitationCandidateRecord] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(CitationCandidateRecord.from_dict(json.loads(line)))
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare citation-aware scored runs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Scored candidate JSONL files.")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = _load_records(args.inputs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[CitationCandidateRecord]] = {}
    for record in records:
        grouped.setdefault(record.search_plan.method, []).append(record)
    metric_names = sorted({key for group in grouped.values() for record in group for key in record.metrics})
    comparison_path = output_dir / "comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "candidate_count", "candidate_score", *metric_names])
        writer.writeheader()
        for method, group in sorted(grouped.items()):
            aggregated = aggregate_metric_dicts(record.metrics for record in group)
            writer.writerow(
                {
                    "method": method,
                    "candidate_count": len(group),
                    "candidate_score": sum(record.candidate_score for record in group) / max(1, len(group)),
                    **{metric: aggregated.get(metric, 0.0) for metric in metric_names},
                }
            )
    report_path = output_dir / "report.md"
    lines = [
        "# Citation-Aware Cross-Lingual RAG Report",
        "",
        "This report compares search-plan methods by retrieval quality, citation support, and useful citation language mix.",
        "",
        "## Methods",
        "",
    ]
    for method, group in sorted(grouped.items()):
        aggregated = aggregate_metric_dicts(record.metrics for record in group)
        score = sum(record.candidate_score for record in group) / max(1, len(group))
        lines.append(
            f"- `{method}`: n={len(group)}, candidate_score={score:.4f}, "
            f"recall@10={aggregated.get('recall_at_10', 0.0):.4f}, "
            f"citation_f1={aggregated.get('citation_f1', 0.0):.4f}, "
            f"useful_en={aggregated.get('useful_english_citation_ratio', 0.0):.4f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`machine_translate` is the translation baseline. `gold_target` is an upper-bound reference, not a deployable baseline. "
            "The proposed planner should be judged by whether it improves citation support and useful citation retrieval over machine translation.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"comparison.csv -> {comparison_path}")
    print(f"report.md      -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
