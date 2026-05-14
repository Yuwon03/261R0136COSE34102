"""Translate XOR-TyDi Korean query chunks with a local HuggingFace model.

This script does not call any external translation API. It uses a local
Transformers seq2seq model, downloading model weights through HuggingFace if
they are not already cached.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_QUESTION_SPLIT_RE = re.compile(r"\?")
_COMMA_RE = re.compile(r"[,，、]+")
_QUESTION_RE = re.compile(r"[?？]+")
_SPACE_RE = re.compile(r"\s+")


def _read_queries(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [part.strip(" \t\r\n,") for part in _QUESTION_SPLIT_RE.split(text) if part.strip(" \t\r\n,")]


def _clean_english_query(text: str) -> str:
    cleaned = _COMMA_RE.sub(" ", text)
    cleaned = _QUESTION_RE.sub(" ", cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned).strip(" .")
    return f"{cleaned}?" if cleaned else ""


def _count_queries(path: Path) -> int:
    return len(_read_queries(path))


def _translate_batch(model, tokenizer, queries: list[str], *, max_length: int) -> list[str]:  # type: ignore[no-untyped-def]
    encoded = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    output_ids = model.generate(
        **encoded,
        max_length=max_length,
        num_beams=4,
    )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def _translate_batch_nllb(
    model,
    tokenizer,
    queries: list[str],
    *,
    max_length: int,
    source_lang: str,
    target_lang: str,
) -> list[str]:  # type: ignore[no-untyped-def]
    tokenizer.src_lang = source_lang
    encoded = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
    output_ids = model.generate(
        **encoded,
        forced_bos_token_id=forced_bos_token_id,
        max_length=max_length,
        num_beams=4,
    )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate Korean query chunk files locally.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing Korean chunk .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where English chunk .txt files will be written.",
    )
    parser.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="HuggingFace seq2seq translation model.",
    )
    parser.add_argument(
        "--model-kind",
        choices=["auto", "nllb"],
        default="nllb",
        help="Translation model family. Use nllb for facebook/nllb-* models.",
    )
    parser.add_argument("--source-lang", default="kor_Hang", help="NLLB source language code.")
    parser.add_argument("--target-lang", default="eng_Latn", help="NLLB target language code.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-output-length", type=int, default=64)
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Translate only the first N chunks, useful for a smoke test.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip output chunks that already exist and have matching ? counts.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="1-based first chunk index to translate, based on sorted chunk filenames.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="1-based final chunk index to translate, based on sorted chunk filenames.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.txt"))
    if args.start_index is not None:
        if args.start_index < 1:
            raise SystemExit("--start-index must be >= 1")
        input_files = input_files[args.start_index - 1 :]
    if args.end_index is not None:
        if args.end_index < 1:
            raise SystemExit("--end-index must be >= 1")
        if args.start_index is None:
            input_files = input_files[: args.end_index]
        else:
            input_files = input_files[: args.end_index - args.start_index + 1]
    if args.limit_chunks is not None:
        input_files = input_files[: args.limit_chunks]
    if not input_files:
        raise SystemExit(f"No .txt chunk files found in {input_dir}")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()

    summaries: list[dict[str, int | str]] = []
    for input_path in input_files:
        output_path = output_dir / input_path.name
        korean_count = _count_queries(input_path)
        if args.resume and output_path.exists() and _count_queries(output_path) == korean_count:
            summaries.append(
                {
                    "chunk": input_path.name,
                    "korean_count": korean_count,
                    "english_count": korean_count,
                    "status": "skipped_existing",
                }
            )
            print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
            continue

        queries = _read_queries(input_path)
        translated: list[str] = []
        for start in range(0, len(queries), args.batch_size):
            batch = queries[start : start + args.batch_size]
            if args.model_kind == "nllb":
                batch_translations = _translate_batch_nllb(
                    model,
                    tokenizer,
                    batch,
                    max_length=args.max_output_length,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                )
            else:
                batch_translations = _translate_batch(
                    model,
                    tokenizer,
                    batch,
                    max_length=args.max_output_length,
                )
            translated.extend(_clean_english_query(text) for text in batch_translations)

        output_path.write_text(",".join(translated), encoding="utf-8")
        english_count = _count_queries(output_path)
        summary = {
            "chunk": input_path.name,
            "korean_count": korean_count,
            "english_count": english_count,
            "status": "ok" if korean_count == english_count else "count_mismatch",
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if korean_count != english_count:
            raise SystemExit(
                f"Count mismatch at {input_path.name}: Korean={korean_count} English={english_count}"
            )

    print(json.dumps({"chunks": len(summaries), "status": "done"}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
