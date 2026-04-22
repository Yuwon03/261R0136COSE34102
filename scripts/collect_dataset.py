"""Collect project-format dataset rows from MKQA/XOR-style sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from crosslingual_rewrite.data import load_corpus  # noqa: E402
from crosslingual_rewrite.dataset_builder import (  # noqa: E402
    build_dataset_from_hf,
    build_dataset_from_records,
    iter_jsonl_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Korean query-rewrite JSONL data from MKQA/XOR-TyDi records."
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["mkqa", "xor_tydi"],
        help="Source dataset format.",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Project-format corpus JSONL used to retrieve d+ and d- candidates.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination dataset JSONL path.",
    )
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="Optional local source JSONL. If omitted, HuggingFace datasets is used.",
    )
    parser.add_argument("--split", default="train", help="Source split to load or label.")
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Override HuggingFace dataset name, e.g. apple/mkqa.",
    )
    parser.add_argument(
        "--hf-config",
        default=None,
        help="Optional HuggingFace dataset config name.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow HuggingFace datasets that require custom loading code.",
    )
    parser.add_argument(
        "--top-k-positive",
        type=int,
        default=100,
        help="Retriever depth for finding answer-containing positive docs.",
    )
    parser.add_argument(
        "--top-k-negative",
        type=int,
        default=20,
        help="Retriever depth for selecting hard negative docs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of source records to inspect.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.input_jsonl:
        corpus = load_corpus(args.corpus)
        records = iter_jsonl_records(args.input_jsonl)
        stats = build_dataset_from_records(
            records,
            source=args.source,
            corpus=corpus,
            output_path=args.output,
            split_name=args.split,
            top_k_positive=args.top_k_positive,
            top_k_negative=args.top_k_negative,
            limit=args.limit,
        )
    else:
        stats = build_dataset_from_hf(
            source=args.source,
            corpus_path=args.corpus,
            output_path=args.output,
            split=args.split,
            dataset_name=args.hf_dataset,
            config_name=args.hf_config,
            trust_remote_code=args.trust_remote_code,
            top_k_positive=args.top_k_positive,
            top_k_negative=args.top_k_negative,
            limit=args.limit,
        )

    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
