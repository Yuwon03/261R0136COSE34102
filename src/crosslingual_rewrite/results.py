"""Writers that enforce the shared result schema for every method run.

Each run directory gets the same four artifacts:

* ``config_<run_id>.yaml``: copy of the resolved experiment config.
* ``summary_<run_id>.csv``: one-row aggregate summary.
* ``examples_<run_id>.jsonl``: per-example rows following the schema below.
* ``errors_<run_id>.jsonl``: any per-example errors. Emitted even when empty
  so consumers can rely on the file existing.

The per-example schema:

.. code-block:: json

   {
     "run_id": "string",
     "method": "raw|machine_translate|translate|supervised|retrieval_aware",
     "example_id": "string",
     "question_ko": "string",
     "query": "string",
     "target_query": "string|null",
     "query_type": "pure_ko|mixed_ko_en|non_ko|other",
     "positive_doc_id": "string",
     "negative_doc_id": "string",
     "retrieved_docs": [...],
     "metrics": {...},
     "error": "string|null"
   }
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .baselines import ExampleOutcome, MethodRunResult
from .config import ExperimentConfig, build_run_paths, write_resolved_config
from .evaluation import metric_keys


SUMMARY_HEADER: tuple[str, ...] = (
    "run_id",
    "method",
    "example_count",
    "error_count",
    "recall_at_k",
    "mrr",
    "ndcg",
    "source_diversity",
    "english_source_ratio",
    "korean_source_ratio",
    "faithfulness",
)


@dataclass
class RunArtifacts:
    """Paths for the four files written by :func:`write_run`."""

    run_id: str
    run_dir: Path
    config_path: Path
    summary_path: Path
    examples_path: Path
    errors_path: Path


def _outcome_to_row(
    outcome: ExampleOutcome,
    *,
    run_id: str,
    method: str,
) -> dict[str, Any]:
    example = outcome.example
    retrieved = [doc.to_serializable() for doc in outcome.retrieved]
    metrics = (
        outcome.metrics.to_dict()
        if outcome.metrics is not None
        else {key: 0.0 for key in metric_keys()}
    )
    return {
        "run_id": run_id,
        "method": method,
        "example_id": example.example_id,
        "question_ko": example.question_ko,
        "query": outcome.query,
        "target_query": example.target_query,
        "query_type": example.query_type,
        "positive_doc_id": example.positive_doc_id,
        "negative_doc_id": example.negative_doc_id,
        "retrieved_docs": retrieved,
        "metrics": metrics,
        "error": outcome.error,
    }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_summary_csv(
    path: str | Path,
    *,
    run_id: str,
    method: str,
    example_count: int,
    error_count: int,
    aggregated: Mapping[str, float],
) -> Path:
    """Write the single-row summary CSV for a run."""

    dest = Path(path)
    _ensure_parent(dest)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_HEADER)
        writer.writeheader()
        row: dict[str, Any] = {
            "run_id": run_id,
            "method": method,
            "example_count": int(example_count),
            "error_count": int(error_count),
        }
        for key in metric_keys():
            row[key] = float(aggregated.get(key, 0.0))
        writer.writerow(row)
    return dest


def write_examples_jsonl(
    path: str | Path,
    *,
    run_id: str,
    method: str,
    outcomes: Sequence[ExampleOutcome],
) -> Path:
    dest = Path(path)
    _ensure_parent(dest)
    with dest.open("w", encoding="utf-8") as fh:
        for outcome in outcomes:
            row = _outcome_to_row(outcome, run_id=run_id, method=method)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dest


def write_errors_jsonl(
    path: str | Path,
    *,
    run_id: str,
    method: str,
    outcomes: Iterable[ExampleOutcome],
) -> Path:
    dest = Path(path)
    _ensure_parent(dest)
    errors = [o for o in outcomes if o.error]
    with dest.open("w", encoding="utf-8") as fh:
        for outcome in errors:
            example = outcome.example
            payload = {
                "run_id": run_id,
                "method": method,
                "example_id": example.example_id,
                "error": outcome.error,
                "query": outcome.query,
                "question_ko": example.question_ko,
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return dest


def write_run(
    cfg: ExperimentConfig,
    *,
    run_id: str,
    run_result: MethodRunResult,
) -> RunArtifacts:
    """Write the full set of artifacts for a run and return their paths."""

    paths = build_run_paths(cfg, run_id, create=True)
    write_resolved_config(cfg, paths["config"])
    write_summary_csv(
        paths["summary"],
        run_id=run_id,
        method=run_result.method,
        example_count=run_result.example_count,
        error_count=run_result.error_count,
        aggregated=run_result.aggregated,
    )
    write_examples_jsonl(
        paths["examples"],
        run_id=run_id,
        method=run_result.method,
        outcomes=run_result.outcomes,
    )
    write_errors_jsonl(
        paths["errors"],
        run_id=run_id,
        method=run_result.method,
        outcomes=run_result.outcomes,
    )
    return RunArtifacts(
        run_id=run_id,
        run_dir=paths["run_dir"],
        config_path=paths["config"],
        summary_path=paths["summary"],
        examples_path=paths["examples"],
        errors_path=paths["errors"],
    )


def read_summary_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a summary CSV into a list of dictionaries with typed values."""

    dest = Path(path)
    if not dest.exists():
        return []
    rows: list[dict[str, Any]] = []
    with dest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for key in ("example_count", "error_count"):
                if key in row and row[key] != "":
                    row[key] = int(row[key])
            for key in metric_keys():
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def read_examples_jsonl(path: str | Path) -> list[dict[str, Any]]:
    dest = Path(path)
    if not dest.exists():
        return []
    rows: list[dict[str, Any]] = []
    with dest.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


__all__ = [
    "SUMMARY_HEADER",
    "RunArtifacts",
    "write_summary_csv",
    "write_examples_jsonl",
    "write_errors_jsonl",
    "write_run",
    "read_summary_csv",
    "read_examples_jsonl",
]
