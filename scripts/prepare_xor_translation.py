"""Prepare XOR-TyDi Korean questions for manual/external translation."""

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

from crosslingual_rewrite.dataset_builder import iter_jsonl_records  # noqa: E402
from crosslingual_rewrite.xor_translation_prep import (  # noqa: E402
    merge_translated_queries,
    prepare_xor_from_hf,
    prepare_xor_translation_files,
    split_query_list_file,
    validate_question_mark_query_counts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export XOR-TyDi Korean questions for translation or merge English query output."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Export Korean questions and attributes.")
    export.add_argument("--jsonl-output", required=True, help="Prepared Korean attribute JSONL.")
    export.add_argument(
        "--query-list-output",
        required=True,
        help="Comma-separated Korean question list for external translation.",
    )
    export.add_argument(
        "--query-chunks-dir",
        default=None,
        help="Optional directory for smaller comma-separated Korean query chunk files.",
    )
    export.add_argument("--chunk-size", type=int, default=100, help="Queries per chunk when chunking.")
    export.add_argument(
        "--input-jsonl",
        default=None,
        help="Optional local XOR-style JSONL. If omitted, HuggingFace datasets is used.",
    )
    export.add_argument("--split", default="train", help="Source split.")
    export.add_argument("--hf-dataset", default=None, help="Override HuggingFace dataset name.")
    export.add_argument("--hf-config", default=None, help="Optional HuggingFace dataset config.")
    export.add_argument("--trust-remote-code", action="store_true")
    export.add_argument("--limit", type=int, default=None)

    merge = subparsers.add_parser("merge", help="Merge comma-separated English queries back into JSONL.")
    merge.add_argument("--jsonl-input", required=True, help="Prepared Korean attribute JSONL.")
    merge.add_argument(
        "--english-query-list",
        required=True,
        help="Comma-separated English query list or directory of chunk .txt files in export order.",
    )
    merge.add_argument("--output-jsonl", required=True, help="XOR JSONL with target_query added.")
    merge.add_argument("--field-name", default="target_query")
    merge.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Allow missing/extra English queries instead of failing fast.",
    )

    split = subparsers.add_parser("split", help="Split an existing query list into smaller chunk files.")
    split.add_argument("--query-list-input", required=True, help="Comma-separated query list to split.")
    split.add_argument("--output-dir", required=True, help="Directory where chunk files will be written.")
    split.add_argument("--chunk-size", type=int, default=100, help="Queries per chunk.")
    split.add_argument("--prefix", default=None, help="Optional output file prefix. Defaults to input stem.")

    validate = subparsers.add_parser(
        "validate",
        help="Validate Korean and English query counts by splitting on question marks.",
    )
    validate.add_argument(
        "--korean-query-list",
        required=True,
        help="Korean query file or directory of chunk .txt files.",
    )
    validate.add_argument(
        "--english-query-list",
        required=True,
        help="English query file or directory of chunk .txt files.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "export":
        if args.input_jsonl:
            records = iter_jsonl_records(args.input_jsonl)
            stats = prepare_xor_translation_files(
                records,
                jsonl_output=args.jsonl_output,
                query_list_output=args.query_list_output,
                query_chunks_dir=args.query_chunks_dir,
                chunk_size=args.chunk_size,
                split_name=args.split,
                limit=args.limit,
            )
        else:
            stats = prepare_xor_from_hf(
                jsonl_output=args.jsonl_output,
                query_list_output=args.query_list_output,
                query_chunks_dir=args.query_chunks_dir,
                chunk_size=args.chunk_size,
                split=args.split,
                dataset_name=args.hf_dataset,
                config_name=args.hf_config,
                trust_remote_code=args.trust_remote_code,
                limit=args.limit,
            )
        print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "merge":
        stats = merge_translated_queries(
            jsonl_input=args.jsonl_input,
            english_query_list=args.english_query_list,
            output_jsonl=args.output_jsonl,
            field_name=args.field_name,
            require_exact_count=not args.allow_count_mismatch,
        )
        print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "split":
        chunks_written = split_query_list_file(
            args.query_list_input,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            prefix=args.prefix,
        )
        print(json.dumps({"chunks_written": chunks_written}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    validation = validate_question_mark_query_counts(
        korean_query_list=args.korean_query_list,
        english_query_list=args.english_query_list,
    )
    print(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if validation.matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
