"""Shared helpers for the unit test suite.

Every test module uses these helpers to:

* make the ``src/`` package importable (in case ``PYTHONPATH=src`` was not
  set, for example when running through a test IDE);
* build a minimal valid :class:`ExperimentConfig` that points at the smoke
  JSONL fixtures bundled with the repo.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SMOKE_DATASET = PROJECT_ROOT / "data" / "smoke_dataset.jsonl"
SMOKE_CORPUS = PROJECT_ROOT / "data" / "smoke_corpus.jsonl"


def ensure_src_on_path() -> None:
    if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


ensure_src_on_path()


from crosslingual_rewrite.config import (  # noqa: E402  - sys.path must be set first
    DataSection,
    EvaluationSection,
    ExperimentConfig,
    ExperimentSection,
    ModelSection,
    OutputSection,
    RetrieverSection,
    TrainingSection,
)


def make_config(
    *,
    runs_dir: str | os.PathLike[str],
    dataset_path: str | os.PathLike[str] | None = None,
    corpus_path: str | os.PathLike[str] | None = None,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    training_output_dir: str | os.PathLike[str] | None = None,
    analysis_dir: str | os.PathLike[str] | None = None,
    experiment_name: str = "unit-test",
    retrieval_loss_weight: float = 0.5,
    top_k: int = 3,
) -> ExperimentConfig:
    """Build a valid config for tests using the smoke fixtures by default."""

    runs_dir_str = str(runs_dir)
    analysis_str = str(analysis_dir) if analysis_dir is not None else str(Path(runs_dir_str).parent / "analysis")
    ckpt_str = str(checkpoint_dir) if checkpoint_dir is not None else str(Path(runs_dir_str).parent / "checkpoints" / "supervised")
    train_output_str = str(training_output_dir) if training_output_dir is not None else str(Path(runs_dir_str).parent / "checkpoints")
    return ExperimentConfig(
        experiment=ExperimentSection(name=experiment_name, task_type="evaluation", seed=13),
        data=DataSection(
            dataset_path=str(dataset_path) if dataset_path is not None else str(SMOKE_DATASET),
            corpus_path=str(corpus_path) if corpus_path is not None else str(SMOKE_CORPUS),
            split="train",
        ),
        model=ModelSection(
            base_model="google/mt5-small",
            checkpoint_dir=ckpt_str,
            max_input_length=128,
            max_output_length=32,
            use_mock_model_for_smoke=True,
        ),
        training=TrainingSection(
            epochs=1,
            batch_size=2,
            learning_rate=2e-4,
            retrieval_loss_weight=retrieval_loss_weight,
            output_dir=train_output_str,
        ),
        retriever=RetrieverSection(backend="bm25", top_k=top_k),
        evaluation=EvaluationSection(top_k=top_k),
        output=OutputSection(runs_dir=runs_dir_str, analysis_dir=analysis_str),
    )


def write_jsonl(path: str | os.PathLike[str], rows: list[dict]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dest
