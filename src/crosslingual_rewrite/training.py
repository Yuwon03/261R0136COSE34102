"""Supervised + retrieval-aware training loops.

Two objectives are supported:

* Supervised: standard sequence-to-sequence teacher forcing against
  ``target_query``.
* Retrieval-aware: supervised generation loss plus a retrieval-driven term
  that rewards generated queries whose BM25 score on the positive document
  exceeds the score on the negative document. The retriever remains frozen.

The module exposes mock training paths so local smoke runs can produce a
checkpoint and an appended JSONL log without requiring a HuggingFace model
download. The real code path exists for non-smoke configs and is kept
intentionally straightforward.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import ExperimentConfig
from .data import CorpusDocument, RewriteExample, DataValidationError
from .modeling import (
    HFRewriteModel,
    MockRewriteModel,
    RewriteModel,
    build_rewrite_model,
)
from .retriever import BM25Retriever, tokenize


@dataclass
class TrainingLogEvent:
    """A single structured log event written to ``training_log.jsonl``."""

    epoch: int
    step: int
    gen_loss: float
    retrieval_loss: float
    total_loss: float
    batch_size: int
    objective: str
    event: str = "batch"
    timestamp: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["extras"] = dict(self.extras)
        return payload


@dataclass
class TrainingResult:
    """Summary returned to scripts after a training run."""

    objective: str
    checkpoint_dir: str
    log_path: str
    epochs: int
    total_steps: int
    final_gen_loss: float
    final_retrieval_loss: float
    final_total_loss: float
    mode: str  # "mock" | "real"
    extras: Mapping[str, Any] = field(default_factory=dict)


def _iter_batches(examples: Sequence[RewriteExample], batch_size: int) -> list[list[RewriteExample]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [list(examples[i : i + batch_size]) for i in range(0, len(examples), batch_size)]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_target_queries(examples: Sequence[RewriteExample]) -> None:
    missing = [ex.example_id for ex in examples if not ex.target_query]
    if missing:
        raise DataValidationError(
            "Training requires target_query for every example. Missing: " + ", ".join(str(m) for m in missing)
        )


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _mock_gen_loss(example: RewriteExample) -> float:
    """Heuristic supervised loss used by the mock trainer.

    Higher overlap between the target tokens and the Korean question's
    latinized tokens means lower loss. The value is deterministic.
    """

    target = example.target_query or ""
    target_tokens = set(tokenize(target))
    source_tokens = set(tokenize(example.question_ko))
    overlap = _jaccard(target_tokens, source_tokens)
    return max(1e-3, -math.log(0.1 + 0.8 * overlap))


def _mock_retrieval_loss(
    example: RewriteExample,
    retriever: BM25Retriever,
    predicted_query: str,
) -> float:
    """Hinge-style retrieval loss using BM25 margin on the fixed retriever."""

    try:
        pos_score = retriever.score(predicted_query, example.positive_doc_id)
        neg_score = retriever.score(predicted_query, example.negative_doc_id)
    except KeyError:
        return 1.0
    margin = pos_score - neg_score
    return max(0.0, 1.0 - margin)


def _training_log_path(cfg: ExperimentConfig, objective: str) -> Path:
    base = Path(cfg.training.output_dir)
    dest = base / objective
    dest.mkdir(parents=True, exist_ok=True)
    return dest / "training_log.jsonl"


def _checkpoint_dir(cfg: ExperimentConfig, objective: str) -> Path:
    base = Path(cfg.training.output_dir)
    dest = base / objective
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _append_log(path: Path, event: TrainingLogEvent, on_log: Callable[[TrainingLogEvent], None] | None) -> None:
    payload = event.to_dict()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    if on_log is not None:
        on_log(event)


def _mock_train_supervised(
    cfg: ExperimentConfig,
    examples: Sequence[RewriteExample],
    *,
    on_log: Callable[[TrainingLogEvent], None] | None = None,
) -> TrainingResult:
    _ensure_target_queries(examples)
    log_path = _training_log_path(cfg, "supervised")
    log_path.write_text("", encoding="utf-8")

    total_steps = 0
    final_gen = 0.0
    for epoch in range(1, cfg.training.epochs + 1):
        for batch in _iter_batches(examples, cfg.training.batch_size):
            total_steps += 1
            gen_losses = [_mock_gen_loss(ex) for ex in batch]
            gen_loss = sum(gen_losses) / len(gen_losses)
            final_gen = gen_loss
            event = TrainingLogEvent(
                epoch=epoch,
                step=total_steps,
                gen_loss=gen_loss,
                retrieval_loss=0.0,
                total_loss=gen_loss,
                batch_size=len(batch),
                objective="supervised",
                event="batch",
                timestamp=_timestamp(),
                extras={"mode": "mock"},
            )
            _append_log(log_path, event, on_log)

    ckpt_dir = _checkpoint_dir(cfg, "supervised")
    model = MockRewriteModel(
        variant="supervised_mock",
        metadata={
            "objective": "supervised",
            "epochs": cfg.training.epochs,
            "steps": total_steps,
            "final_gen_loss": final_gen,
        },
    )
    model.save(ckpt_dir)
    return TrainingResult(
        objective="supervised",
        checkpoint_dir=str(ckpt_dir),
        log_path=str(log_path),
        epochs=cfg.training.epochs,
        total_steps=total_steps,
        final_gen_loss=float(final_gen),
        final_retrieval_loss=0.0,
        final_total_loss=float(final_gen),
        mode="mock",
    )


def _mock_train_retrieval_aware(
    cfg: ExperimentConfig,
    examples: Sequence[RewriteExample],
    corpus: Sequence[CorpusDocument],
    *,
    on_log: Callable[[TrainingLogEvent], None] | None = None,
) -> TrainingResult:
    _ensure_target_queries(examples)
    retriever = BM25Retriever(corpus)
    log_path = _training_log_path(cfg, "retrieval_aware")
    log_path.write_text("", encoding="utf-8")

    weight = float(cfg.training.retrieval_loss_weight)
    total_steps = 0
    final_gen = 0.0
    final_retr = 0.0
    final_total = 0.0
    for epoch in range(1, cfg.training.epochs + 1):
        for batch in _iter_batches(examples, cfg.training.batch_size):
            total_steps += 1
            gen_values = [_mock_gen_loss(ex) for ex in batch]
            retr_values = [
                _mock_retrieval_loss(ex, retriever, ex.target_query or ex.question_ko)
                for ex in batch
            ]
            gen_loss = sum(gen_values) / len(gen_values)
            retr_loss = sum(retr_values) / len(retr_values)
            total = gen_loss + weight * retr_loss
            final_gen, final_retr, final_total = gen_loss, retr_loss, total
            event = TrainingLogEvent(
                epoch=epoch,
                step=total_steps,
                gen_loss=gen_loss,
                retrieval_loss=retr_loss,
                total_loss=total,
                batch_size=len(batch),
                objective="retrieval_aware",
                event="batch",
                timestamp=_timestamp(),
                extras={"mode": "mock", "retrieval_loss_weight": weight},
            )
            _append_log(log_path, event, on_log)

    ckpt_dir = _checkpoint_dir(cfg, "retrieval_aware")
    model = MockRewriteModel(
        variant="retrieval_aware_mock",
        metadata={
            "objective": "retrieval_aware",
            "epochs": cfg.training.epochs,
            "steps": total_steps,
            "retrieval_loss_weight": weight,
            "final_gen_loss": final_gen,
            "final_retrieval_loss": final_retr,
            "final_total_loss": final_total,
        },
    )
    model.save(ckpt_dir)
    return TrainingResult(
        objective="retrieval_aware",
        checkpoint_dir=str(ckpt_dir),
        log_path=str(log_path),
        epochs=cfg.training.epochs,
        total_steps=total_steps,
        final_gen_loss=float(final_gen),
        final_retrieval_loss=float(final_retr),
        final_total_loss=float(final_total),
        mode="mock",
    )


def _real_train_supervised(
    cfg: ExperimentConfig,
    examples: Sequence[RewriteExample],
    *,
    on_log: Callable[[TrainingLogEvent], None] | None = None,
) -> TrainingResult:
    import torch  # local imports so smoke tests never load torch if not needed
    from torch.optim import AdamW

    _ensure_target_queries(examples)
    model = build_rewrite_model(cfg, prefer_mock=False)
    assert isinstance(model, HFRewriteModel)

    if cfg.training.gradient_checkpointing and hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()
        if hasattr(model.model, "config") and hasattr(model.model.config, "use_cache"):
            model.model.config.use_cache = False

    optimizer = AdamW(model.model.parameters(), lr=cfg.training.learning_rate)
    gradient_accumulation_steps = max(1, int(cfg.training.gradient_accumulation_steps))
    use_fp16 = bool(cfg.training.fp16) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    log_path = _training_log_path(cfg, "supervised")
    log_path.write_text("", encoding="utf-8")

    model.model.train()
    total_steps = 0
    final_gen = 0.0
    optimizer.zero_grad()
    batches_per_epoch = math.ceil(len(examples) / cfg.training.batch_size)
    total_expected_steps = cfg.training.epochs * batches_per_epoch
    for epoch in range(1, cfg.training.epochs + 1):
        for batch in _iter_batches(examples, cfg.training.batch_size):
            inputs = model.encode(ex.question_ko for ex in batch)
            labels = model.encode_targets(ex.target_query or "" for ex in batch)
            label_ids = labels["input_ids"].clone()
            label_ids[label_ids == model.tokenizer.pad_token_id] = -100
            with torch.cuda.amp.autocast(enabled=use_fp16):
                output = model.model(**inputs, labels=label_ids)
                raw_loss = output.loss
                loss = raw_loss / gradient_accumulation_steps
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_steps += 1

            should_step = (
                total_steps % gradient_accumulation_steps == 0
                or total_steps == total_expected_steps
            )
            if should_step:
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            gen_value = float(raw_loss.detach().cpu().item())
            final_gen = gen_value
            event = TrainingLogEvent(
                epoch=epoch,
                step=total_steps,
                gen_loss=gen_value,
                retrieval_loss=0.0,
                total_loss=gen_value,
                batch_size=len(batch),
                objective="supervised",
                event="batch",
                timestamp=_timestamp(),
                extras={
                    "mode": "real",
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "optimizer_step": should_step,
                    "fp16": use_fp16,
                    "gradient_checkpointing": bool(cfg.training.gradient_checkpointing),
                },
            )
            _append_log(log_path, event, on_log)

    ckpt_dir = _checkpoint_dir(cfg, "supervised")
    model.save(ckpt_dir)
    return TrainingResult(
        objective="supervised",
        checkpoint_dir=str(ckpt_dir),
        log_path=str(log_path),
        epochs=cfg.training.epochs,
        total_steps=total_steps,
        final_gen_loss=float(final_gen),
        final_retrieval_loss=0.0,
        final_total_loss=float(final_gen),
        mode="real",
    )


def _real_train_retrieval_aware(
    cfg: ExperimentConfig,
    examples: Sequence[RewriteExample],
    corpus: Sequence[CorpusDocument],
    *,
    on_log: Callable[[TrainingLogEvent], None] | None = None,
) -> TrainingResult:
    import torch
    from torch.optim import AdamW

    _ensure_target_queries(examples)
    retriever = BM25Retriever(corpus)
    model = build_rewrite_model(cfg, prefer_mock=False)
    assert isinstance(model, HFRewriteModel)
    optimizer = AdamW(model.model.parameters(), lr=cfg.training.learning_rate)
    weight = float(cfg.training.retrieval_loss_weight)
    log_path = _training_log_path(cfg, "retrieval_aware")
    log_path.write_text("", encoding="utf-8")

    total_steps = 0
    final_gen = final_retr = final_total = 0.0
    model.model.train()
    for epoch in range(1, cfg.training.epochs + 1):
        for batch in _iter_batches(examples, cfg.training.batch_size):
            inputs = model.encode(ex.question_ko for ex in batch)
            labels = model.encode_targets(ex.target_query or "" for ex in batch)
            label_ids = labels["input_ids"].clone()
            label_ids[label_ids == model.tokenizer.pad_token_id] = -100
            out = model.model(**inputs, labels=label_ids)
            gen_loss = out.loss

            with torch.no_grad():
                predictions = model.generate(batch)
            margins = []
            for ex, pred in zip(batch, predictions):
                try:
                    pos = retriever.score(pred, ex.positive_doc_id)
                    neg = retriever.score(pred, ex.negative_doc_id)
                except KeyError:
                    margins.append(-1.0)
                    continue
                margins.append(pos - neg)
            mean_margin = sum(margins) / max(1, len(margins))
            retrieval_hinge = max(0.0, 1.0 - mean_margin)

            retrieval_scalar = torch.tensor(
                retrieval_hinge, dtype=gen_loss.dtype, device=gen_loss.device
            )
            total = gen_loss + weight * retrieval_scalar
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            total_steps += 1
            final_gen = float(gen_loss.detach().cpu().item())
            final_retr = float(retrieval_scalar.detach().cpu().item())
            final_total = float(total.detach().cpu().item())
            event = TrainingLogEvent(
                epoch=epoch,
                step=total_steps,
                gen_loss=final_gen,
                retrieval_loss=final_retr,
                total_loss=final_total,
                batch_size=len(batch),
                objective="retrieval_aware",
                event="batch",
                timestamp=_timestamp(),
                extras={
                    "mode": "real",
                    "retrieval_loss_weight": weight,
                    "mean_margin": mean_margin,
                },
            )
            _append_log(log_path, event, on_log)

    ckpt_dir = _checkpoint_dir(cfg, "retrieval_aware")
    model.save(ckpt_dir)
    return TrainingResult(
        objective="retrieval_aware",
        checkpoint_dir=str(ckpt_dir),
        log_path=str(log_path),
        epochs=cfg.training.epochs,
        total_steps=total_steps,
        final_gen_loss=float(final_gen),
        final_retrieval_loss=float(final_retr),
        final_total_loss=float(final_total),
        mode="real",
    )


def train_supervised(
    cfg: ExperimentConfig,
    examples: Sequence[RewriteExample],
    *,
    on_log: Callable[[TrainingLogEvent], None] | None = None,
) -> TrainingResult:
    """Run the supervised training objective.

    In smoke mode this function produces a deterministic log and saves a
    :class:`MockRewriteModel` checkpoint. In real mode it fine-tunes the
    underlying HuggingFace seq2seq model.
    """

    if cfg.model.use_mock_model_for_smoke:
        return _mock_train_supervised(cfg, examples, on_log=on_log)
    return _real_train_supervised(cfg, examples, on_log=on_log)


def train_retrieval_aware(
    cfg: ExperimentConfig,
    examples: Sequence[RewriteExample],
    corpus: Sequence[CorpusDocument],
    *,
    on_log: Callable[[TrainingLogEvent], None] | None = None,
) -> TrainingResult:
    """Run the retrieval-aware training objective."""

    if cfg.model.use_mock_model_for_smoke:
        return _mock_train_retrieval_aware(cfg, examples, corpus, on_log=on_log)
    return _real_train_retrieval_aware(cfg, examples, corpus, on_log=on_log)


__all__ = [
    "TrainingLogEvent",
    "TrainingResult",
    "train_supervised",
    "train_retrieval_aware",
]
