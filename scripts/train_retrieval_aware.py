"""Train (or mock-train) the retrieval-aware rewriter."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from crosslingual_rewrite.config import load_config, validate_config  # noqa: E402
from crosslingual_rewrite.data import load_corpus, load_dataset  # noqa: E402
from crosslingual_rewrite.training import train_retrieval_aware  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the retrieval-aware rewriter.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N training steps. Use 0 to disable progress output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    validate_config(cfg)
    examples = load_dataset(cfg.data.dataset_path)
    corpus = load_corpus(cfg.data.corpus_path)
    steps_per_epoch = math.ceil(len(examples) / cfg.training.batch_size)
    total_expected_steps = steps_per_epoch * cfg.training.epochs
    print(
        f"[train_retrieval_aware] loaded examples={len(examples)} corpus={len(corpus)} "
        f"epochs={cfg.training.epochs} batch_size={cfg.training.batch_size} "
        f"expected_steps={total_expected_steps}",
        flush=True,
    )

    def report_progress(event) -> None:
        if args.progress_every <= 0:
            return
        if event.step == 1 or event.step == total_expected_steps or event.step % args.progress_every == 0:
            pct = 100.0 * event.step / max(1, total_expected_steps)
            print(
                f"[train_retrieval_aware] progress step={event.step}/{total_expected_steps} "
                f"epoch={event.epoch}/{cfg.training.epochs} ({pct:.1f}%) "
                f"gen_loss={event.gen_loss:.4f} retrieval_loss={event.retrieval_loss:.4f} "
                f"total_loss={event.total_loss:.4f}",
                flush=True,
            )

    result = train_retrieval_aware(cfg, examples, corpus, on_log=report_progress)

    print(
        f"[train_retrieval_aware] mode={result.mode} epochs={result.epochs} "
        f"steps={result.total_steps} gen_loss={result.final_gen_loss:.4f} "
        f"retrieval_loss={result.final_retrieval_loss:.4f} "
        f"total_loss={result.final_total_loss:.4f}"
    )
    print(f"checkpoint_dir: {result.checkpoint_dir}")
    print(f"log_path: {result.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
