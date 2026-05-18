"""Config loading, validation, and run-path utilities.

The repository uses YAML configs with a small number of required sections.
Each script reads a config file, validates it, derives a deterministic run id,
and writes a copy of the resolved config into the run directory.

The dataclasses in this module are the canonical in-memory representation of
a configuration. Downstream modules must not read raw dictionaries, and should
instead consume ``ExperimentConfig`` objects.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


_REQUIRED_SECTIONS: tuple[str, ...] = (
    "experiment",
    "data",
    "model",
    "training",
    "retriever",
    "evaluation",
    "output",
)

_VALID_METHODS: frozenset[str] = frozenset(
    {"raw", "machine_translate", "translate", "supervised", "retrieval_aware"}
)


@dataclass
class ExperimentSection:
    name: str
    task_type: str = "evaluation"
    seed: int = 13


@dataclass
class DataSection:
    dataset_path: str
    corpus_path: str
    split: str = "train"


@dataclass
class ModelSection:
    base_model: str = "google/mt5-small"
    checkpoint_dir: str = "output/checkpoints/supervised"
    max_input_length: int = 256
    max_output_length: int = 64
    use_mock_model_for_smoke: bool = True
    src_lang: str | None = None
    tgt_lang: str | None = None
    lora_enabled: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)


@dataclass
class TrainingSection:
    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    retrieval_loss_weight: float = 0.5
    fp16: bool = False
    gradient_checkpointing: bool = False
    save_each_epoch: bool = False
    output_dir: str = "output/checkpoints"


@dataclass
class RetrieverSection:
    backend: str = "bm25"
    top_k: int = 5


@dataclass
class EvaluationSection:
    top_k: int = 5
    metrics: list[str] = field(
        default_factory=lambda: [
            "recall_at_k",
            "mrr",
            "ndcg",
            "source_diversity",
            "source_language_ratio",
            "faithfulness",
        ]
    )


@dataclass
class OutputSection:
    runs_dir: str = "output/runs"
    analysis_dir: str = "output/analysis"


@dataclass
class ExperimentConfig:
    """Canonical in-memory representation of a config file."""

    experiment: ExperimentSection
    data: DataSection
    model: ModelSection
    training: TrainingSection
    retriever: RetrieverSection
    evaluation: EvaluationSection
    output: OutputSection
    config_path: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation suitable for YAML serialization."""

        return {
            "experiment": asdict(self.experiment),
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "retriever": asdict(self.retriever),
            "evaluation": asdict(self.evaluation),
            "output": asdict(self.output),
        }


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML file from disk and return a dictionary."""

    file_path = Path(path)
    if not file_path.exists():
        raise ConfigError(f"Config file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ConfigError(f"Config file is empty: {file_path}")
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file must contain a mapping at the top level: {file_path}"
        )
    return data


def _section_from_dict(cls: type, payload: Mapping[str, Any]) -> Any:
    """Build a dataclass section from a mapping, tolerating extra keys.

    Unknown fields are ignored so that new optional keys don't crash older
    configs.
    """

    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(payload, Mapping):
        raise ConfigError(f"Expected mapping for {cls.__name__}, got {type(payload).__name__}")
    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in payload.items() if k in field_names}
    return cls(**filtered)


def _ensure_sections(raw: Mapping[str, Any]) -> None:
    missing = [s for s in _REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ConfigError(
            "Missing required config sections: " + ", ".join(sorted(missing))
        )


def parse_config(raw: Mapping[str, Any], *, config_path: str | None = None) -> ExperimentConfig:
    """Convert a raw dict into a validated :class:`ExperimentConfig`.

    Raises
    ------
    ConfigError
        If required sections or fields are missing.
    """

    _ensure_sections(raw)

    try:
        experiment = _section_from_dict(ExperimentSection, raw["experiment"])
        data = _section_from_dict(DataSection, raw["data"])
        model = _section_from_dict(ModelSection, raw["model"])
        training = _section_from_dict(TrainingSection, raw["training"])
        retriever = _section_from_dict(RetrieverSection, raw["retriever"])
        evaluation = _section_from_dict(EvaluationSection, raw["evaluation"])
        output = _section_from_dict(OutputSection, raw["output"])
    except TypeError as exc:
        raise ConfigError(f"Invalid config values: {exc}") from exc

    config = ExperimentConfig(
        experiment=experiment,
        data=data,
        model=model,
        training=training,
        retriever=retriever,
        evaluation=evaluation,
        output=output,
        config_path=str(config_path) if config_path is not None else None,
        raw=dict(raw),
    )
    return config


def load_config(path: str | os.PathLike[str]) -> ExperimentConfig:
    """Load and validate a config file from disk."""

    raw = load_yaml(path)
    return parse_config(raw, config_path=str(path))


def validate_config(cfg: ExperimentConfig, *, require_paths_exist: bool = True) -> None:
    """Validate that a config is usable for running experiments.

    When ``require_paths_exist`` is true (the default), dataset and corpus
    paths must resolve to files on disk. Set this to false for tests that only
    exercise parse behavior.
    """

    if not cfg.experiment.name:
        raise ConfigError("experiment.name must be a non-empty string")
    if cfg.retriever.top_k <= 0:
        raise ConfigError("retriever.top_k must be a positive integer")
    if cfg.evaluation.top_k <= 0:
        raise ConfigError("evaluation.top_k must be a positive integer")
    if cfg.training.epochs <= 0:
        raise ConfigError("training.epochs must be a positive integer")
    if cfg.training.batch_size <= 0:
        raise ConfigError("training.batch_size must be a positive integer")
    if cfg.training.gradient_accumulation_steps <= 0:
        raise ConfigError("training.gradient_accumulation_steps must be a positive integer")
    if cfg.model.lora_r <= 0:
        raise ConfigError("model.lora_r must be a positive integer")
    if cfg.model.lora_alpha <= 0:
        raise ConfigError("model.lora_alpha must be a positive integer")
    if not 0.0 <= cfg.model.lora_dropout < 1.0:
        raise ConfigError("model.lora_dropout must be in [0.0, 1.0)")

    if require_paths_exist:
        dataset_path = Path(cfg.data.dataset_path)
        if not dataset_path.exists():
            raise ConfigError(f"Dataset path does not exist: {dataset_path}")
        corpus_path = Path(cfg.data.corpus_path)
        if not corpus_path.exists():
            raise ConfigError(f"Corpus path does not exist: {corpus_path}")


_RUN_ID_SANITIZER = re.compile(r"[^A-Za-z0-9_\-]+")


def _sanitize_token(token: str) -> str:
    cleaned = _RUN_ID_SANITIZER.sub("-", token.strip())
    return cleaned.strip("-") or "run"


def generate_run_id(
    cfg: ExperimentConfig,
    method: str,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Produce a deterministic-but-unique run id for a method run.

    The run id embeds the experiment name, method, UTC timestamp, and a short
    hash of the resolved config so that repeated runs with different configs
    are distinguishable on disk.
    """

    if method not in _VALID_METHODS:
        raise ConfigError(
            f"Unknown method '{method}'. Expected one of: {sorted(_VALID_METHODS)}"
        )
    ts = timestamp or datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    digest_source = repr(sorted(cfg.to_dict().items()))
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:8]
    parts = [_sanitize_token(cfg.experiment.name), _sanitize_token(method), stamp, digest]
    return "_".join(parts)


def build_run_paths(
    cfg: ExperimentConfig,
    run_id: str,
    *,
    create: bool = True,
) -> dict[str, Path]:
    """Compute the standard file layout for a run directory."""

    runs_dir = Path(cfg.output.runs_dir)
    run_dir = runs_dir / run_id
    paths = {
        "runs_dir": runs_dir,
        "run_dir": run_dir,
        "config": run_dir / f"config_{run_id}.yaml",
        "summary": run_dir / f"summary_{run_id}.csv",
        "examples": run_dir / f"examples_{run_id}.jsonl",
        "errors": run_dir / f"errors_{run_id}.jsonl",
    }
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return paths


def write_resolved_config(cfg: ExperimentConfig, destination: str | os.PathLike[str]) -> Path:
    """Write the resolved config to ``destination`` and return the path."""

    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.to_dict()
    with dest.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True, allow_unicode=True)
    return dest


def valid_methods() -> Iterable[str]:
    """Return the set of method names accepted by scripts."""

    return tuple(sorted(_VALID_METHODS))


__all__ = [
    "ConfigError",
    "ExperimentSection",
    "DataSection",
    "ModelSection",
    "TrainingSection",
    "RetrieverSection",
    "EvaluationSection",
    "OutputSection",
    "ExperimentConfig",
    "load_yaml",
    "load_config",
    "parse_config",
    "validate_config",
    "generate_run_id",
    "build_run_paths",
    "write_resolved_config",
    "valid_methods",
]
