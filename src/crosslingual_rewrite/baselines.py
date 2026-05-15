"""Per-method query generation and one shared runner.

Each of the four methods (``raw``, ``translate``, ``supervised``,
``retrieval_aware``) ultimately boils down to picking or generating a single
retrieval query per example. The runner :func:`run_method` wires together
config parsing, model loading, retrieval, and evaluation so every method
produces artifacts in the same shape.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .config import ExperimentConfig
from .data import CorpusDocument, RetrievedDocument, RewriteExample
from .evaluation import MetricBundle, aggregate_metrics, compute_metrics
from .modeling import RewriteModel, build_rewrite_model
from .retriever import BM25Retriever


VALID_METHODS: tuple[str, ...] = ("raw", "translate", "supervised", "retrieval_aware")


class MethodError(RuntimeError):
    """Raised when a requested method cannot be executed."""


@dataclass
class ExampleOutcome:
    """Per-example result produced by :func:`run_method`."""

    example: RewriteExample
    query: str
    retrieved: list[RetrievedDocument]
    metrics: MetricBundle | None
    error: str | None = None


@dataclass
class MethodRunResult:
    """Aggregate outcome of running a single method over a dataset."""

    method: str
    outcomes: list[ExampleOutcome]
    aggregated: dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    example_count: int = 0


def _require(method: str) -> None:
    if method not in VALID_METHODS:
        raise MethodError(
            f"Unknown method {method!r}. Expected one of: {VALID_METHODS}"
        )


def generate_queries(
    method: str,
    examples: Sequence[RewriteExample],
    *,
    model: RewriteModel | None = None,
) -> list[str]:
    """Return one retrieval query per example for the requested method.

    * ``raw``: returns ``example.question_ko``.
    * ``translate``: returns ``example.target_query``. Missing target queries
      raise a clear error.
    * ``supervised`` and ``retrieval_aware``: delegate to ``model.generate``.
    """

    _require(method)
    if method == "raw":
        return [ex.question_ko for ex in examples]
    if method == "translate":
        queries: list[str] = []
        for ex in examples:
            queries.append(ex.require_target_query(context="translate baseline"))
        return queries
    if model is None:
        raise MethodError(f"Method {method!r} requires a loaded model")
    generated = model.generate(examples)
    if len(generated) != len(examples):
        raise MethodError(
            f"Model returned {len(generated)} queries for {len(examples)} examples"
        )
    return [text if text else ex.question_ko for text, ex in zip(generated, examples)]


def _resolve_checkpoint(
    cfg: ExperimentConfig, method: str, override: str | Path | None
) -> Path | None:
    if override is not None:
        return Path(override)
    if method == "supervised":
        return Path(cfg.model.checkpoint_dir)
    if method == "retrieval_aware":
        return Path(cfg.training.output_dir) / "retrieval_aware"
    return None


def _load_model_for(
    cfg: ExperimentConfig, method: str, override: str | Path | None
) -> RewriteModel | None:
    if method in {"raw", "translate"}:
        return None
    checkpoint = _resolve_checkpoint(cfg, method, override)
    return build_rewrite_model(cfg, checkpoint_dir=checkpoint)


def run_method(
    cfg: ExperimentConfig,
    method: str,
    *,
    examples: Sequence[RewriteExample],
    corpus: Sequence[CorpusDocument],
    checkpoint_dir: str | Path | None = None,
    top_k_override: int | None = None,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    generation_batch_size: int | None = None,
) -> MethodRunResult:
    """Run one method end-to-end and return a :class:`MethodRunResult`.

    Errors on individual examples (for example a missing ``target_query``)
    are captured as per-example errors so the run can continue and downstream
    tooling still sees a full list of outcomes.
    """

    _require(method)
    if limit is not None and limit >= 0:
        examples = list(examples)[:limit]

    top_k = int(top_k_override) if top_k_override is not None else int(cfg.retriever.top_k)
    if top_k <= 0:
        raise MethodError("top_k must be positive")

    retriever = BM25Retriever(corpus)
    model = _load_model_for(cfg, method, checkpoint_dir)

    outcomes: list[ExampleOutcome] = []
    metrics_accumulator: list[dict[str, float]] = []
    error_count = 0

    queries: list[str | None] = [None] * len(examples)
    generation_errors: dict[int, str] = {}

    if method in {"raw", "translate"}:
        try:
            generated = generate_queries(method, examples, model=model)
        except Exception as exc:  # noqa: BLE001 - surface to per-example error
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            for ex in examples:
                outcomes.append(
                    ExampleOutcome(
                        example=ex,
                        query="",
                        retrieved=[],
                        metrics=None,
                        error=f"generate_queries_failed: {detail}",
                    )
                )
                error_count += 1
            return MethodRunResult(
                method=method,
                outcomes=outcomes,
                aggregated=aggregate_metrics([]),
                error_count=error_count,
                example_count=len(examples),
            )
        queries = list(generated)
    else:
        batch_size = (
            int(generation_batch_size)
            if generation_batch_size is not None
            else int(cfg.training.batch_size)
        )
        if batch_size <= 0:
            raise MethodError("generation_batch_size must be positive")
        for start in range(0, len(examples), batch_size):
            batch = list(examples[start : start + batch_size])
            try:
                generated = generate_queries(method, batch, model=model)
            except Exception as exc:  # noqa: BLE001 - keep remaining batches running
                detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                for offset in range(len(batch)):
                    generation_errors[start + offset] = f"generate_queries_failed: {detail}"
                continue
            if len(generated) != len(batch):
                detail = (
                    f"MethodError: Model returned {len(generated)} queries "
                    f"for {len(batch)} examples"
                )
                for offset in range(len(batch)):
                    generation_errors[start + offset] = f"generate_queries_failed: {detail}"
                continue
            for offset, query in enumerate(generated):
                queries[start + offset] = query

    total_examples = len(examples)
    for index, (ex, query) in enumerate(zip(examples, queries), start=1):
        generation_error = generation_errors.get(index - 1)
        if generation_error is not None:
            outcomes.append(
                ExampleOutcome(
                    example=ex,
                    query="",
                    retrieved=[],
                    metrics=None,
                    error=generation_error,
                )
            )
            error_count += 1
            if on_progress is not None:
                on_progress(index, total_examples)
            continue
        if not query or not query.strip():
            outcomes.append(
                ExampleOutcome(
                    example=ex,
                    query="",
                    retrieved=[],
                    metrics=None,
                    error="empty_query",
                )
            )
            error_count += 1
            if on_progress is not None:
                on_progress(index, total_examples)
            continue
        try:
            retrieved = retriever.retrieve(
                query, top_k=top_k, positive_doc_id=ex.positive_doc_id
            )
            metrics = compute_metrics(
                retrieved,
                positive_doc_id=ex.positive_doc_id,
                top_k=top_k,
                generated_query=query,
                target_query=ex.target_query,
            )
            outcomes.append(
                ExampleOutcome(
                    example=ex,
                    query=query,
                    retrieved=retrieved,
                    metrics=metrics,
                )
            )
            metrics_accumulator.append(metrics.to_dict())
        except Exception as exc:  # noqa: BLE001 - surface to per-example error
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            outcomes.append(
                ExampleOutcome(
                    example=ex,
                    query=query,
                    retrieved=[],
                    metrics=None,
                    error=f"runtime_error: {detail}",
                )
            )
            error_count += 1
        if on_progress is not None:
            on_progress(index, total_examples)

    aggregated = aggregate_metrics(metrics_accumulator)
    return MethodRunResult(
        method=method,
        outcomes=outcomes,
        aggregated=aggregated,
        error_count=error_count,
        example_count=len(examples),
    )


__all__ = [
    "VALID_METHODS",
    "MethodError",
    "ExampleOutcome",
    "MethodRunResult",
    "generate_queries",
    "run_method",
]
