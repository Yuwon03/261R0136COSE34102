"""Tests for :mod:`crosslingual_rewrite.config`."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from tests._helpers import SMOKE_CORPUS, SMOKE_DATASET, ensure_src_on_path

ensure_src_on_path()

from crosslingual_rewrite.config import (  # noqa: E402
    ConfigError,
    build_run_paths,
    generate_run_id,
    load_config,
    parse_config,
    validate_config,
    write_resolved_config,
)


def _minimal_config_dict(tmp_path: Path) -> dict:
    return {
        "experiment": {"name": "cfg-test", "task_type": "evaluation", "seed": 42},
        "data": {
            "dataset_path": str(SMOKE_DATASET),
            "corpus_path": str(SMOKE_CORPUS),
            "split": "train",
        },
        "model": {
            "base_model": "google/mt5-small",
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "max_input_length": 128,
            "max_output_length": 32,
            "use_mock_model_for_smoke": True,
            "src_lang": "kor_Hang",
            "tgt_lang": "eng_Latn",
            "lora_enabled": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_target_modules": ["q_proj", "v_proj"],
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-4,
            "retrieval_loss_weight": 0.25,
            "fp16": False,
            "gradient_checkpointing": True,
            "save_each_epoch": True,
            "output_dir": str(tmp_path / "ckpt_root"),
        },
        "retriever": {"backend": "bm25", "top_k": 5},
        "evaluation": {"top_k": 5, "metrics": ["recall_at_k", "mrr"]},
        "output": {
            "runs_dir": str(tmp_path / "runs"),
            "analysis_dir": str(tmp_path / "analysis"),
        },
    }


class ConfigLoadingTests(unittest.TestCase):
    def test_parse_config_returns_dataclasses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dict = _minimal_config_dict(Path(tmp))
            cfg = parse_config(cfg_dict)
            self.assertEqual(cfg.experiment.name, "cfg-test")
            self.assertEqual(cfg.retriever.top_k, 5)
            self.assertEqual(cfg.training.retrieval_loss_weight, 0.25)
            self.assertEqual(cfg.training.gradient_accumulation_steps, 2)
            self.assertTrue(cfg.training.gradient_checkpointing)
            self.assertTrue(cfg.training.save_each_epoch)
            self.assertTrue(cfg.model.use_mock_model_for_smoke)
            self.assertEqual(cfg.model.src_lang, "kor_Hang")
            self.assertEqual(cfg.model.tgt_lang, "eng_Latn")
            self.assertTrue(cfg.model.lora_enabled)
            self.assertEqual(cfg.model.lora_target_modules, ["q_proj", "v_proj"])

    def test_parse_config_missing_section_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dict = _minimal_config_dict(Path(tmp))
            cfg_dict.pop("model")
            with self.assertRaises(ConfigError) as ctx:
                parse_config(cfg_dict)
            self.assertIn("model", str(ctx.exception))

    def test_parse_config_missing_multiple_sections_lists_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dict = _minimal_config_dict(Path(tmp))
            cfg_dict.pop("model")
            cfg_dict.pop("training")
            with self.assertRaises(ConfigError) as ctx:
                parse_config(cfg_dict)
            msg = str(ctx.exception)
            self.assertIn("model", msg)
            self.assertIn("training", msg)

    def test_load_config_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dict = _minimal_config_dict(Path(tmp))
            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
            cfg = load_config(cfg_path)
            self.assertEqual(cfg.experiment.name, "cfg-test")
            self.assertEqual(cfg.config_path, str(cfg_path))

    def test_load_config_missing_file_raises(self) -> None:
        with self.assertRaises(ConfigError):
            load_config("/nonexistent/config.yaml")


class ValidationTests(unittest.TestCase):
    def test_validate_requires_existing_dataset_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dict = _minimal_config_dict(Path(tmp))
            cfg_dict["data"]["dataset_path"] = str(Path(tmp) / "missing.jsonl")
            cfg = parse_config(cfg_dict)
            with self.assertRaises(ConfigError):
                validate_config(cfg, require_paths_exist=True)

    def test_validate_rejects_non_positive_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dict = _minimal_config_dict(Path(tmp))
            cfg_dict["retriever"]["top_k"] = 0
            cfg = parse_config(cfg_dict)
            with self.assertRaises(ConfigError):
                validate_config(cfg, require_paths_exist=False)

    def test_validate_passes_valid_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dict = _minimal_config_dict(Path(tmp))
            cfg = parse_config(cfg_dict)
            validate_config(cfg, require_paths_exist=True)


class RunPathTests(unittest.TestCase):
    def test_run_id_is_deterministic_but_timestamped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = parse_config(_minimal_config_dict(Path(tmp)))
            from datetime import datetime, timezone

            ts = datetime(2026, 4, 22, 9, 15, 0, tzinfo=timezone.utc)
            run_id_a = generate_run_id(cfg, "raw", timestamp=ts)
            run_id_b = generate_run_id(cfg, "raw", timestamp=ts)
            self.assertEqual(run_id_a, run_id_b)
            self.assertIn("cfg-test", run_id_a)
            self.assertIn("raw", run_id_a)
            self.assertIn("20260422T091500Z", run_id_a)

    def test_run_paths_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = parse_config(_minimal_config_dict(Path(tmp)))
            run_id = generate_run_id(cfg, "raw")
            paths = build_run_paths(cfg, run_id, create=True)
            self.assertTrue(paths["run_dir"].exists())
            self.assertEqual(paths["config"].name, f"config_{run_id}.yaml")
            self.assertEqual(paths["summary"].name, f"summary_{run_id}.csv")
            self.assertEqual(paths["examples"].name, f"examples_{run_id}.jsonl")
            self.assertEqual(paths["errors"].name, f"errors_{run_id}.jsonl")

    def test_generate_run_id_rejects_unknown_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = parse_config(_minimal_config_dict(Path(tmp)))
            with self.assertRaises(ConfigError):
                generate_run_id(cfg, "bogus")


class WriteResolvedConfigTests(unittest.TestCase):
    def test_roundtrip_config_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = parse_config(_minimal_config_dict(Path(tmp)))
            dest = Path(tmp) / "copy.yaml"
            write_resolved_config(cfg, dest)
            loaded = yaml.safe_load(dest.read_text(encoding="utf-8"))
            self.assertEqual(loaded["experiment"]["name"], "cfg-test")
            self.assertEqual(loaded["retriever"]["top_k"], 5)


if __name__ == "__main__":
    unittest.main()
