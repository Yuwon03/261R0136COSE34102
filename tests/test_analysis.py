"""Tests for :mod:`crosslingual_rewrite.analysis`."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tests._helpers import ensure_src_on_path, make_config

ensure_src_on_path()

from crosslingual_rewrite.analysis import (  # noqa: E402
    FAILURE_CATEGORIES,
    analyze_runs,
    classify_failure,
    load_run_records,
)
from crosslingual_rewrite.baselines import run_method  # noqa: E402
from crosslingual_rewrite.config import generate_run_id  # noqa: E402
from crosslingual_rewrite.data import load_corpus, load_dataset  # noqa: E402
from crosslingual_rewrite.results import write_run  # noqa: E402


def _seed_runs(tmp: Path, methods: list[str]) -> tuple[Path, Path]:
    runs_dir = tmp / "runs"
    analysis_dir = tmp / "analysis"
    runs_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_config(runs_dir=runs_dir, analysis_dir=analysis_dir, top_k=3)
    examples = load_dataset(cfg.data.dataset_path)
    corpus = load_corpus(cfg.data.corpus_path)
    from datetime import datetime, timedelta, timezone

    base_ts = datetime(2026, 4, 22, 9, 0, 0, tzinfo=timezone.utc)
    for idx, method in enumerate(methods):
        ts = base_ts + timedelta(minutes=idx)
        run_id = generate_run_id(cfg, method, timestamp=ts)
        result = run_method(cfg, method, examples=examples, corpus=corpus)
        write_run(cfg, run_id=run_id, run_result=result)
    return runs_dir, analysis_dir


class FailureClassifierTests(unittest.TestCase):
    def test_positive_not_retrieved(self) -> None:
        row = {
            "metrics": {"recall_at_k": 0.0, "mrr": 0.0, "source_diversity": 0.5, "faithfulness": 0.5},
            "error": None,
        }
        categories = classify_failure(row)
        self.assertIn("positive_not_retrieved", categories)

    def test_low_mrr_triggered_above_recall(self) -> None:
        row = {
            "metrics": {"recall_at_k": 1.0, "mrr": 0.1, "source_diversity": 0.5, "faithfulness": 0.8},
            "error": None,
        }
        categories = classify_failure(row)
        self.assertIn("low_mrr", categories)
        self.assertNotIn("positive_not_retrieved", categories)

    def test_empty_query_and_runtime_error(self) -> None:
        row_a = {"metrics": {}, "error": "empty_query"}
        row_b = {"metrics": {}, "error": "runtime_error: boom"}
        self.assertIn("empty_query", classify_failure(row_a))
        self.assertIn("runtime_error", classify_failure(row_b))

    def test_thresholds_can_be_overridden(self) -> None:
        row = {
            "metrics": {"recall_at_k": 1.0, "mrr": 0.5, "source_diversity": 0.4, "faithfulness": 0.4},
            "error": None,
        }
        categories = classify_failure(row, thresholds={"low_mrr": 0.9})
        self.assertIn("low_mrr", categories)


class AnalysisWriterTests(unittest.TestCase):
    def test_analyze_runs_produces_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir, analysis_dir = _seed_runs(Path(tmp), ["raw", "translate"])
            artifacts = analyze_runs(runs_dir, analysis_dir)

            self.assertTrue(artifacts.comparison_path.exists())
            self.assertTrue(artifacts.slice_summary_path.exists())
            self.assertTrue(artifacts.failure_cases_path.exists())
            self.assertTrue(artifacts.report_path.exists())

    def test_comparison_csv_contains_method_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir, analysis_dir = _seed_runs(Path(tmp), ["raw", "translate"])
            artifacts = analyze_runs(runs_dir, analysis_dir)
            with artifacts.comparison_path.open("r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            methods = sorted({row["method"] for row in rows})
            self.assertEqual(methods, ["raw", "translate"])
            for row in rows:
                self.assertIn("recall_at_k", row)

    def test_slice_summary_groups_by_query_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir, analysis_dir = _seed_runs(Path(tmp), ["raw"])
            artifacts = analyze_runs(runs_dir, analysis_dir)
            with artifacts.slice_summary_path.open("r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            query_types = {row["query_type"] for row in rows}
            self.assertTrue(query_types.issubset({"pure_ko", "mixed_ko_en", "non_ko", "other"}))
            self.assertTrue(any(row["method"] == "raw" for row in rows))

    def test_failure_cases_lists_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir, analysis_dir = _seed_runs(Path(tmp), ["raw"])
            artifacts = analyze_runs(runs_dir, analysis_dir)
            with artifacts.failure_cases_path.open("r", encoding="utf-8") as fh:
                lines = [line for line in fh if line.strip()]
            for line in lines:
                payload = json.loads(line)
                self.assertIn("failure_categories", payload)
                for category in payload["failure_categories"]:
                    self.assertIn(category, FAILURE_CATEGORIES)

    def test_report_markdown_contains_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir, analysis_dir = _seed_runs(Path(tmp), ["raw", "translate"])
            artifacts = analyze_runs(runs_dir, analysis_dir)
            text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Aggregate metrics", text)
            self.assertIn("Failure cases", text)
            self.assertIn("| method |", text)

    def test_load_run_records_skips_invalid_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            runs_dir.mkdir()
            (runs_dir / "empty").mkdir()
            (runs_dir / "also_empty").mkdir()
            records = load_run_records(runs_dir)
            self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
