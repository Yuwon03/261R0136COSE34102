"""Cross-run analysis: comparison CSV, slice breakdowns, failure cases, report.

:func:`analyze_runs` is the entry point used by ``scripts/compare_runs.py``
and by the tests. It reads every ``output/runs/<run_id>/`` directory
produced by :mod:`crosslingual_rewrite.results` and writes four artifacts to
the configured analysis directory.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .evaluation import metric_keys


FAILURE_CATEGORIES: tuple[str, ...] = (
    "positive_not_retrieved",
    "low_mrr",
    "low_source_diversity",
    "low_faithfulness",
    "empty_query",
    "runtime_error",
)


DEFAULT_THRESHOLDS: dict[str, float] = {
    "low_mrr": 0.2,
    "low_source_diversity": 0.1,
    "low_faithfulness": 0.2,
}


@dataclass
class RunRecord:
    """In-memory representation of a single run directory."""

    run_id: str
    run_dir: Path
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    example_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def method(self) -> str:
        if self.summary_rows:
            return str(self.summary_rows[0].get("method", ""))
        if self.example_rows:
            return str(self.example_rows[0].get("method", ""))
        return ""


@dataclass
class AnalysisArtifacts:
    comparison_path: Path
    slice_summary_path: Path
    failure_cases_path: Path
    report_path: Path


def _iter_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            converted = dict(row)
            for key in ("example_count", "error_count"):
                if converted.get(key) not in (None, ""):
                    converted[key] = int(converted[key])
            for key in metric_keys():
                if converted.get(key) not in (None, ""):
                    converted[key] = float(converted[key])
            rows.append(converted)
    return rows


def _iter_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def load_run_records(runs_dir: str | Path) -> list[RunRecord]:
    """Load every run directory under ``runs_dir``.

    Directories without a summary file are skipped so that partial runs do
    not crash comparison.
    """

    base = Path(runs_dir)
    if not base.exists():
        return []
    records: list[RunRecord] = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / f"summary_{run_dir.name}.csv"
        examples_path = run_dir / f"examples_{run_dir.name}.jsonl"
        if not summary_path.exists():
            continue
        record = RunRecord(run_id=run_dir.name, run_dir=run_dir)
        record.summary_rows = _iter_summary_rows(summary_path)
        if examples_path.exists():
            record.example_rows = _iter_jsonl_rows(examples_path)
        records.append(record)
    return records


def _comparison_header() -> list[str]:
    header = ["run_id", "method", "example_count", "error_count"]
    header.extend(metric_keys())
    return header


def write_comparison_csv(records: Sequence[RunRecord], destination: str | Path) -> Path:
    """Merge all run summaries into a single CSV."""

    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = _comparison_header()
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for record in records:
            for row in record.summary_rows:
                output: dict[str, Any] = {
                    "run_id": record.run_id,
                    "method": row.get("method", record.method),
                    "example_count": row.get("example_count", 0),
                    "error_count": row.get("error_count", 0),
                }
                for key in metric_keys():
                    output[key] = float(row.get(key, 0.0) or 0.0)
                writer.writerow(output)
    return dest


def _slice_summary_rows(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        for row in record.example_rows:
            key = (record.run_id, str(row.get("method", record.method)), str(row.get("query_type", "other")))
            bucket = buckets.setdefault(
                key,
                {
                    "run_id": record.run_id,
                    "method": key[1],
                    "query_type": key[2],
                    "example_count": 0,
                    **{metric: 0.0 for metric in metric_keys()},
                },
            )
            bucket["example_count"] += 1
            metrics = row.get("metrics") or {}
            for metric in metric_keys():
                bucket[metric] += float(metrics.get(metric, 0.0) or 0.0)

    finalized: list[dict[str, Any]] = []
    for bucket in buckets.values():
        count = max(1, int(bucket["example_count"]))
        averaged = dict(bucket)
        for metric in metric_keys():
            averaged[metric] = averaged[metric] / count
        finalized.append(averaged)
    finalized.sort(key=lambda b: (b["run_id"], b["method"], b["query_type"]))
    return finalized


def write_slice_summary(records: Sequence[RunRecord], destination: str | Path) -> Path:
    """Write the per-``query_type`` slice summary CSV."""

    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = ["run_id", "method", "query_type", "example_count", *metric_keys()]
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in _slice_summary_rows(records):
            writer.writerow(row)
    return dest


def classify_failure(
    row: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> list[str]:
    """Classify a per-example row into zero or more failure categories."""

    limits = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    categories: list[str] = []
    error = row.get("error")
    if error == "empty_query":
        categories.append("empty_query")
    if isinstance(error, str) and error.startswith("runtime_error"):
        categories.append("runtime_error")
    metrics = row.get("metrics") or {}
    if float(metrics.get("recall_at_k", 0.0) or 0.0) <= 0.0:
        categories.append("positive_not_retrieved")
    if float(metrics.get("mrr", 0.0) or 0.0) < float(limits["low_mrr"]):
        categories.append("low_mrr")
    if float(metrics.get("source_diversity", 0.0) or 0.0) < float(limits["low_source_diversity"]):
        categories.append("low_source_diversity")
    if float(metrics.get("faithfulness", 0.0) or 0.0) < float(limits["low_faithfulness"]):
        categories.append("low_faithfulness")
    return categories


def write_failure_cases(
    records: Sequence[RunRecord],
    destination: str | Path,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> Path:
    """Write the failure case JSONL file."""

    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for record in records:
            for row in record.example_rows:
                categories = classify_failure(row, thresholds=thresholds)
                if not categories:
                    continue
                payload = {
                    "run_id": record.run_id,
                    "method": row.get("method", record.method),
                    "example_id": row.get("example_id"),
                    "query_type": row.get("query_type"),
                    "query": row.get("query"),
                    "target_query": row.get("target_query"),
                    "failure_categories": categories,
                    "metrics": row.get("metrics") or {},
                    "error": row.get("error"),
                }
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return dest


def _format_metric_table(records: Sequence[RunRecord]) -> str:
    header = ["run_id", "method", "example_count", "error_count", *metric_keys()]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for record in records:
        for row in record.summary_rows:
            cells = [
                record.run_id,
                str(row.get("method", record.method)),
                str(row.get("example_count", 0)),
                str(row.get("error_count", 0)),
            ]
            for metric in metric_keys():
                cells.append(f"{float(row.get(metric, 0.0) or 0.0):.4f}")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_failure_counts(records: Sequence[RunRecord], thresholds: Mapping[str, float] | None) -> str:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        for row in record.example_rows:
            key = str(row.get("method", record.method))
            bucket = counts.setdefault(key, {c: 0 for c in FAILURE_CATEGORIES})
            for category in classify_failure(row, thresholds=thresholds):
                bucket[category] = bucket.get(category, 0) + 1
    if not counts:
        return "_No failure signals detected (or no example rows available)._"
    header = ["method", *FAILURE_CATEGORIES]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for method in sorted(counts):
        row = counts[method]
        cells = [method] + [str(row.get(category, 0)) for category in FAILURE_CATEGORIES]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    records: Sequence[RunRecord],
    destination: str | Path,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> Path:
    """Write the human-readable Markdown report."""

    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Cross-Lingual Query Rewriting - Comparison Report")
    lines.append("")
    lines.append(
        "This report is generated by `scripts/compare_runs.py`. It compares "
        "the four methods (`raw`, `translate`, `supervised`, `retrieval_aware`) "
        "against the same fixed retriever and corpus."
    )
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    if records:
        lines.append(_format_metric_table(records))
    else:
        lines.append("_No runs were discovered._")
    lines.append("")
    lines.append("## Failure cases by method")
    lines.append("")
    lines.append(_format_failure_counts(records, thresholds))
    lines.append("")
    lines.append("## Notes and limitations")
    lines.append("")
    lines.append(
        "* The smoke retriever is a fixed BM25-style index over the bundled JSONL corpus."
    )
    lines.append(
        "* `faithfulness` is a lexical Jaccard proxy between the generated query and the "
        "target query tokens. Replace it with a stronger semantic evaluator before drawing "
        "conclusions about faithfulness."
    )
    lines.append(
        "* Mock-model runs reuse the gold `target_query` when available, so supervised and "
        "retrieval-aware numbers in smoke mode will closely track the translate baseline."
    )
    lines.append("")
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def analyze_runs(
    runs_dir: str | Path,
    output_dir: str | Path,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> AnalysisArtifacts:
    """Produce every analysis artifact for a directory of runs."""

    records = load_run_records(runs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison = write_comparison_csv(records, output_path / "comparison.csv")
    slices = write_slice_summary(records, output_path / "slice_summary.csv")
    failures = write_failure_cases(records, output_path / "failure_cases.jsonl", thresholds=thresholds)
    report = write_report(records, output_path / "report.md", thresholds=thresholds)
    return AnalysisArtifacts(
        comparison_path=comparison,
        slice_summary_path=slices,
        failure_cases_path=failures,
        report_path=report,
    )


__all__ = [
    "FAILURE_CATEGORIES",
    "DEFAULT_THRESHOLDS",
    "RunRecord",
    "AnalysisArtifacts",
    "load_run_records",
    "write_comparison_csv",
    "write_slice_summary",
    "write_failure_cases",
    "write_report",
    "classify_failure",
    "analyze_runs",
]
