"""Run a single retrieval method and write run artifacts.

Usage
-----
    python scripts/run_method.py --config configs/local_smoke.yaml --method raw

The script resolves the config, loads the dataset and corpus, runs the
requested method against the fixed BM25 retriever, and writes the four
standard artifacts to ``output/runs/<run_id>/``.
"""

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

from crosslingual_rewrite.baselines import VALID_METHODS, run_method  # noqa: E402
from crosslingual_rewrite.config import (  # noqa: E402
    generate_run_id,
    load_config,
    validate_config,
)
from crosslingual_rewrite.data import load_corpus, load_dataset  # noqa: E402
from crosslingual_rewrite.results import write_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one retrieval method end-to-end.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(VALID_METHODS),
        help="Retrieval method to execute.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint directory for supervised / retrieval_aware runs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override retriever.top_k for this run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples processed (useful for quick checks).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    validate_config(cfg)

    examples = load_dataset(cfg.data.dataset_path)
    corpus = load_corpus(cfg.data.corpus_path)

    run_id = generate_run_id(cfg, args.method)
    run_result = run_method(
        cfg,
        args.method,
        examples=examples,
        corpus=corpus,
        checkpoint_dir=args.checkpoint,
        top_k_override=args.top_k,
        limit=args.limit,
    )
    artifacts = write_run(cfg, run_id=run_id, run_result=run_result)

    summary_line = (
        f"[{args.method}] run_id={run_id} examples={run_result.example_count} "
        f"errors={run_result.error_count} "
        f"recall@k={run_result.aggregated.get('recall_at_k', 0):.4f} "
        f"mrr={run_result.aggregated.get('mrr', 0):.4f} "
        f"ndcg={run_result.aggregated.get('ndcg', 0):.4f}"
    )
    print(summary_line)
    print(f"run_dir: {artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
