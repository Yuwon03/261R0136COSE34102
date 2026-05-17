"""Rewrite-model wrappers for real HuggingFace seq2seq models and mocks.

Two concrete models are provided so that the whole pipeline can run without
network access:

* :class:`HFRewriteModel` wraps a HuggingFace seq2seq model such as
  ``google/mt5-small`` and is used when ``use_mock_model_for_smoke`` is
  ``False``.
* :class:`MockRewriteModel` is a deterministic, offline rewriter that either
  returns the gold ``target_query`` supplied at training/inference time or
  produces a simple fallback derived from the Korean question. It supports
  the same ``save``/``load`` API as the real model so checkpoints can be
  written and re-loaded in local smoke tests.

Both classes implement the :class:`RewriteModel` interface.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import ExperimentConfig
from .data import RewriteExample


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded from its checkpoint directory."""


@dataclass
class GenerationRequest:
    """A lightweight bundle of per-call generation controls."""

    max_output_length: int = 64
    num_beams: int = 1


class RewriteModel(ABC):
    """Common interface for rewrite models used across the pipeline.

    All models accept a :class:`RewriteExample` (which carries both the
    Korean question and, when known, the target query) and return an English
    retrieval query.
    """

    model_type: str = "abstract"

    @abstractmethod
    def generate(
        self,
        examples: Sequence[RewriteExample],
        *,
        request: GenerationRequest | None = None,
    ) -> list[str]:
        """Return one generated query per example."""

    @abstractmethod
    def save(self, output_dir: str | Path, *, merge_lora: bool = False) -> None:
        """Persist the model to ``output_dir``."""

    @classmethod
    @abstractmethod
    def load(cls, checkpoint_dir: str | Path) -> "RewriteModel":
        """Load a previously saved model from ``checkpoint_dir``."""


_ENGLISH_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _deterministic_fallback_query(question_ko: str) -> str:
    """Derive a deterministic English-ish query from a Korean question.

    The implementation keeps any latinized tokens already present in the
    question (e.g. ``dataclass``, ``BM25``) and appends a short hash suffix to
    make the output distinguishable across different questions.
    """

    cleaned = question_ko.strip()
    latin_tokens = _ENGLISH_WORD_RE.findall(cleaned)
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    parts = [tok.lower() for tok in latin_tokens]
    if not parts:
        parts = ["query"]
    parts.append(f"kq{digest}")
    return " ".join(parts)


class MockRewriteModel(RewriteModel):
    """Deterministic offline rewriter used for local smoke tests.

    If a ``target_query`` is present on the example the mock returns it as-is
    (this simulates a perfectly trained rewriter). When no target is present
    the mock falls back to ``_deterministic_fallback_query`` so the pipeline
    still produces a valid non-empty query.
    """

    model_type = "mock"

    def __init__(
        self,
        *,
        variant: str = "mock",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.variant = variant
        self.metadata: dict[str, Any] = dict(metadata or {})

    def generate(
        self,
        examples: Sequence[RewriteExample],
        *,
        request: GenerationRequest | None = None,
    ) -> list[str]:
        _ = request  # unused; kept for interface compatibility
        outputs: list[str] = []
        for example in examples:
            if example.target_query and example.target_query.strip():
                outputs.append(example.target_query.strip())
            else:
                outputs.append(_deterministic_fallback_query(example.question_ko))
        return outputs

    def save(self, output_dir: str | Path, *, merge_lora: bool = False) -> None:
        _ = merge_lora
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "variant": self.variant,
            "metadata": self.metadata,
        }
        (dest / "checkpoint.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> "MockRewriteModel":
        ckpt_path = Path(checkpoint_dir) / "checkpoint.json"
        if not ckpt_path.exists():
            raise ModelLoadError(
                f"Missing mock checkpoint at {ckpt_path}. Run a training script first."
            )
        payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if payload.get("model_type") != cls.model_type:
            raise ModelLoadError(
                f"Checkpoint at {ckpt_path} is not a mock checkpoint ({payload.get('model_type')!r})."
            )
        return cls(
            variant=str(payload.get("variant", "mock")),
            metadata=payload.get("metadata") or {},
        )


class HFRewriteModel(RewriteModel):
    """Wrapper over a HuggingFace seq2seq model.

    Imports of :mod:`torch` and :mod:`transformers` are deferred so that the
    smoke pipeline and unit tests can run in environments where the packages
    exist as lightweight shims or are not yet installed.
    """

    model_type = "huggingface"

    def __init__(
        self,
        base_model: str,
        *,
        max_input_length: int = 256,
        max_output_length: int = 64,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        lora_enabled: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] | None = None,
        adapter_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        import torch  # local import so tests can skip it
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.base_model = base_model
        self.max_input_length = int(max_input_length)
        self.max_output_length = int(max_output_length)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.lora_enabled = bool(lora_enabled or adapter_dir is not None)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = list(lora_target_modules or [])

        tokenizer_kwargs: dict[str, Any] = {}
        if src_lang:
            tokenizer_kwargs["src_lang"] = src_lang
        if tgt_lang:
            tokenizer_kwargs["tgt_lang"] = tgt_lang
        self._tokenizer = AutoTokenizer.from_pretrained(base_model, **tokenizer_kwargs)
        self._set_tokenizer_languages()
        self._model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        if adapter_dir is not None:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, str(adapter_dir))
        elif lora_enabled:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_kwargs: dict[str, Any] = {
                "task_type": TaskType.SEQ_2_SEQ_LM,
                "r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
            }
            if self.lora_target_modules:
                lora_kwargs["target_modules"] = self.lora_target_modules
            self._model = get_peft_model(self._model, LoraConfig(**lora_kwargs))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model.to(self._device)

    @property
    def tokenizer(self):  # type: ignore[no-untyped-def]
        return self._tokenizer

    @property
    def model(self):  # type: ignore[no-untyped-def]
        return self._model

    @property
    def device(self) -> str:
        return self._device

    def _set_tokenizer_languages(self) -> None:
        if self.src_lang and hasattr(self._tokenizer, "src_lang"):
            self._tokenizer.src_lang = self.src_lang
        if self.tgt_lang and hasattr(self._tokenizer, "tgt_lang"):
            self._tokenizer.tgt_lang = self.tgt_lang

    def _target_bos_token_id(self) -> int | None:
        if not self.tgt_lang:
            return None
        lang_code_to_id = getattr(self._tokenizer, "lang_code_to_id", None)
        if isinstance(lang_code_to_id, Mapping):
            token_id = lang_code_to_id.get(self.tgt_lang)
            if token_id is not None:
                return int(token_id)
        token_id = self._tokenizer.convert_tokens_to_ids(self.tgt_lang)
        if token_id is None:
            return None
        unk = getattr(self._tokenizer, "unk_token_id", None)
        if unk is not None and int(token_id) == int(unk):
            return None
        return int(token_id)

    def encode(self, texts: Iterable[str]):  # type: ignore[no-untyped-def]
        """Tokenize a batch of texts for generation or training."""

        self._set_tokenizer_languages()
        return self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        ).to(self._device)

    def encode_targets(self, texts: Iterable[str]):  # type: ignore[no-untyped-def]
        """Tokenize target queries for teacher-forcing training."""

        self._set_tokenizer_languages()
        text_list = list(texts)
        kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": self.max_output_length,
            "return_tensors": "pt",
        }
        if self.tgt_lang:
            try:
                return self._tokenizer(text_target=text_list, **kwargs).to(self._device)
            except TypeError:
                pass
        return self._tokenizer(text_list, **kwargs).to(self._device)

    def generate(
        self,
        examples: Sequence[RewriteExample],
        *,
        request: GenerationRequest | None = None,
    ) -> list[str]:
        if not examples:
            return []
        req = request or GenerationRequest(max_output_length=self.max_output_length)
        inputs = self.encode(example.question_ko for example in examples)
        generation_kwargs: dict[str, Any] = {}
        target_bos = self._target_bos_token_id()
        if target_bos is not None:
            generation_kwargs["forced_bos_token_id"] = target_bos
        output_ids = self._model.generate(
            **inputs,
            max_length=req.max_output_length,
            num_beams=max(1, int(req.num_beams)),
            **generation_kwargs,
        )
        decoded: list[str] = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        return [text.strip() for text in decoded]

    def save(self, output_dir: str | Path, *, merge_lora: bool = False) -> None:
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        stale_mock_checkpoint = dest / "checkpoint.json"
        if stale_mock_checkpoint.exists():
            stale_mock_checkpoint.unlink()
        if self.lora_enabled and merge_lora and hasattr(self._model, "merge_and_unload"):
            for filename in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
                path = dest / filename
                if path.exists():
                    path.unlink()
            self._model = self._model.merge_and_unload()
            self.lora_enabled = False
        elif self.lora_enabled:
            for filename in ("model.safetensors", "pytorch_model.bin"):
                path = dest / filename
                if path.exists():
                    path.unlink()
        self._model.save_pretrained(dest)
        self._tokenizer.save_pretrained(dest)
        metadata = {
            "model_type": self.model_type,
            "base_model": self.base_model,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "lora_enabled": self.lora_enabled,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "lora_merged": bool(merge_lora and not self.lora_enabled),
        }
        (dest / "rewrite_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> "HFRewriteModel":
        meta_path = Path(checkpoint_dir) / "rewrite_metadata.json"
        if not meta_path.exists():
            raise ModelLoadError(
                f"Missing HuggingFace metadata at {meta_path}. Train the model first."
            )
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        checkpoint_path = Path(checkpoint_dir)
        adapter_path = checkpoint_path / "adapter_config.json"
        is_adapter_checkpoint = bool(metadata.get("lora_enabled")) and adapter_path.exists()
        base_model = str(metadata.get("base_model") or checkpoint_path)
        return cls(
            base_model=base_model if is_adapter_checkpoint else str(checkpoint_path),
            max_input_length=int(metadata.get("max_input_length", 256)),
            max_output_length=int(metadata.get("max_output_length", 64)),
            src_lang=metadata.get("src_lang"),
            tgt_lang=metadata.get("tgt_lang"),
            lora_enabled=False,
            lora_r=int(metadata.get("lora_r", 16)),
            lora_alpha=int(metadata.get("lora_alpha", 32)),
            lora_dropout=float(metadata.get("lora_dropout", 0.05)),
            lora_target_modules=metadata.get("lora_target_modules") or [],
            adapter_dir=checkpoint_path if is_adapter_checkpoint else None,
        )


def build_rewrite_model(
    cfg: ExperimentConfig,
    *,
    checkpoint_dir: str | Path | None = None,
    prefer_mock: bool | None = None,
) -> RewriteModel:
    """Factory that picks between the mock and HuggingFace implementations.

    Parameters
    ----------
    cfg:
        Parsed experiment config.
    checkpoint_dir:
        If provided, load a previously saved checkpoint. The factory inspects
        the directory to pick the right model class.
    prefer_mock:
        Optional override. When ``None`` the factory follows
        ``cfg.model.use_mock_model_for_smoke``.
    """

    use_mock = cfg.model.use_mock_model_for_smoke if prefer_mock is None else prefer_mock

    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        mock_file = ckpt_path / "checkpoint.json"
        hf_meta = ckpt_path / "rewrite_metadata.json"
        if hf_meta.exists():
            return HFRewriteModel.load(ckpt_path)
        if mock_file.exists():
            return MockRewriteModel.load(ckpt_path)
        if use_mock:
            return MockRewriteModel(variant="fresh", metadata={"reason": "missing_checkpoint"})
        raise ModelLoadError(
            f"No recognizable checkpoint files at {ckpt_path}. Expected "
            "'checkpoint.json' (mock) or 'rewrite_metadata.json' (HF)."
        )

    if use_mock:
        return MockRewriteModel(variant="fresh")
    return HFRewriteModel(
        base_model=cfg.model.base_model,
        max_input_length=cfg.model.max_input_length,
        max_output_length=cfg.model.max_output_length,
        src_lang=cfg.model.src_lang,
        tgt_lang=cfg.model.tgt_lang,
        lora_enabled=cfg.model.lora_enabled,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        lora_target_modules=cfg.model.lora_target_modules,
    )


__all__ = [
    "ModelLoadError",
    "GenerationRequest",
    "RewriteModel",
    "MockRewriteModel",
    "HFRewriteModel",
    "build_rewrite_model",
]
