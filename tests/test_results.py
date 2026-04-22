"""Tests for :mod:`crosslingual_rewrite.results`."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tests._helpers import ensure_src_on_path, make_config

ensure_src_on_path()

from crosslingual_rewrite.baselines import run_method  # noqa: E402
from crosslingual_rewrite.config import generate_run_id  # noqa: E402
from crosslingual_rewrite.data import load_corpus, load_dataset  # noqa: E402
from crosslingual_rewrite.evaluation import metric_keys  # noqa: E402
from crosslingual_rewrite.results import (  # noqa: E402
    SUMMARY_HEADER,
    read_examples_jsonl,
    read_summary_csv,
    write_run,
)


class ResultWriterTests(unittest.TestCase):
    def test_write_run_creates_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            cfg = make_config(runs_dir=runs_dir)
            examples = load_dataset(cfg.data.dataset_path)
            corpus = load_corpus(cfg.data.corpus_path)
            result = run_method(cfg, "raw", examples=examples, corpus=corpus)

            run_id = generate_run_id(cfg, "raw")
            artifacts = write_run(cfg, run_id=run_id, run_result=result)

            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.examples_path.exists())
            self.assertTrue(artifacts.errors_path.exists())

            self.assertEqual(artifacts.config_path.name, f"config_{run_id}.yaml")
            self.assertEqual(artifacts.summary_path.name, f"summary_{run_id}.csv")
            self.assertEqual(artifacts.examples_path.name, f"examples_{run_id}.jsonl")
            self.assertEqual(artifacts.errors_path.name, f"errors_{run_id}.jsonl")

    def test_summary_csv_has_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            cfg = make_config(runs_dir=runs_dir)
            examples = load_dataset(cfg.data.dataset_path)
            corpus = load_corpus(cfg.data.corpus_path)
            result = run_method(cfg, "translate", examples=examples, corpus=corpus)

            run_id = generate_run_id(cfg, "translate")
            artifacts = write_run(cfg, run_id=run_id, run_result=result)

            with artifacts.summary_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            for column in SUMMARY_HEADER:
                self.assertIn(column, rows[0])
            for metric in metric_keys():
                self.assertIn(metric, rows[0])

    def test_examples_jsonl_matches_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            cfg = make_config(runs_dir=runs_dir)
            examples = load_dataset(cfg.data.dataset_path)
            corpus = load_corpus(cfg.data.corpus_path)
            result = run_method(cfg, "raw", examples=examples, corpus=corpus)

            run_id = generate_run_id(cfg, "raw")
            artifacts = write_run(cfg, run_id=run_id, run_result=result)

            rows = read_examples_jsonl(artifacts.examples_path)
            self.assertEqual(len(rows), len(examples))
            for row in rows:
                self.assertIn("run_id", row)
                self.assertIn("method", row)
                self.assertIn("question_ko", row)
                self.assertIn("query", row)
                self.assertIn("target_query", row)
                self.assertIn("query_type", row)
                self.assertIn("positive_doc_id", row)
                self.assertIn("negative_doc_id", row)
                self.assertIn("retrieved_docs", row)
                self.assertIn("metrics", row)
                self.assertIn("error", row)
                self.assertIsInstance(row["retrieved_docs"], list)
                if row["retrieved_docs"]:
                    first_doc = row["retrieved_docs"][0]
                    for key in ("doc_id", "rank", "score", "source_language", "is_positive", "text"):
                        self.assertIn(key, first_doc)
                self.assertIsInstance(row["metrics"], dict)
                for metric in metric_keys():
                    self.assertIn(metric, row["metrics"])

    def test_errors_jsonl_contains_only_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            cfg = make_config(runs_dir=runs_dir)
            examples = load_dataset(cfg.data.dataset_path)
            corpus = load_corpus(cfg.data.corpus_path)
            result = run_method(cfg, "raw", examples=examples, corpus=corpus)
            run_id = generate_run_id(cfg, "raw")
            artifacts = write_run(cfg, run_id=run_id, run_result=result)

            with artifacts.errors_path.open("r", encoding="utf-8") as fh:
                lines = [line for line in fh if line.strip()]
            self.assertEqual(len(lines), result.error_count)

    def test_read_summary_csv_returns_typed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            cfg = make_config(runs_dir=runs_dir)
            examples = load_dataset(cfg.data.dataset_path)
            corpus = load_corpus(cfg.data.corpus_path)
            result = run_method(cfg, "translate", examples=examples, corpus=corpus)
            run_id = generate_run_id(cfg, "translate")
            artifacts = write_run(cfg, run_id=run_id, run_result=result)

            rows = read_summary_csv(artifacts.summary_path)
            self.assertEqual(len(rows), 1)
            self.assertIsInstance(rows[0]["example_count"], int)
            self.assertIsInstance(rows[0]["recall_at_k"], float)

    def test_config_copy_is_valid_yaml(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            cfg = make_config(runs_dir=runs_dir)
            examples = load_dataset(cfg.data.dataset_path)
            corpus = load_corpus(cfg.data.corpus_path)
            result = run_method(cfg, "raw", examples=examples, corpus=corpus)
            run_id = generate_run_id(cfg, "raw")
            artifacts = write_run(cfg, run_id=run_id, run_result=result)

            loaded = yaml.safe_load(artifacts.config_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["experiment"]["name"], cfg.experiment.name)
            self.assertEqual(loaded["retriever"]["top_k"], cfg.retriever.top_k)


if __name__ == "__main__":
    unittest.main()
