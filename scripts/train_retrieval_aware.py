"""Train (or mock-train) the retrieval-aware rewriter."""

from __future__ import annotations

import argparse
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    validate_config(cfg)
    examples = load_dataset(cfg.data.dataset_path)
    corpus = load_corpus(cfg.data.corpus_path)
    result = train_retrieval_aware(cfg, examples, corpus)

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
