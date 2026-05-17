"""Build a supervised dataset with BM25-friendly expanded target queries."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from crosslingual_rewrite.data import load_corpus  # noqa: E402
from crosslingual_rewrite.retriever import tokenize  # noqa: E402


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _words(text: str | None) -> list[str]:
    if not text:
        return []
    return [match.group(0).lower().strip("'") for match in _WORD_RE.finditer(text)]


def _append_unique(target: list[str], tokens: Iterable[str], *, max_tokens: int) -> None:
    seen = set(target)
    for token in tokens:
        cleaned = token.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        if len(cleaned) <= 1 and not cleaned.isdigit():
            continue
        target.append(cleaned)
        seen.add(cleaned)
        if len(target) >= max_tokens:
            return


def _answer_tokens(metadata: Mapping[str, Any]) -> list[str]:
    answers = metadata.get("answers")
    if not isinstance(answers, list):
        return []
    tokens: list[str] = []
    for answer in answers:
        tokens.extend(_words(str(answer)))
    return tokens


def build_expanded_query(
    record: Mapping[str, Any],
    *,
    positive_doc_title: str | None,
    max_tokens: int,
) -> str:
    target_query = str(record.get("target_query") or "")
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}

    tokens: list[str] = []
    _append_unique(tokens, _words(target_query), max_tokens=max_tokens)
    _append_unique(tokens, _words(positive_doc_title), max_tokens=max_tokens)
    _append_unique(tokens, _answer_tokens(metadata), max_tokens=max_tokens)

    keyword_tokens = [tok for tok in tokens if tok not in _STOPWORDS]
    expanded: list[str] = []
    _append_unique(expanded, keyword_tokens, max_tokens=max_tokens)
    _append_unique(expanded, tokens, max_tokens=max_tokens)
    return " ".join(expanded[:max_tokens])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create expanded-query training data.")
    parser.add_argument("--input", required=True, help="Input dataset JSONL.")
    parser.add_argument("--corpus", required=True, help="Corpus JSONL used to resolve positive doc titles.")
    parser.add_argument("--output", required=True, help="Output expanded dataset JSONL.")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum target query tokens.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    corpus = load_corpus(args.corpus)
    title_by_doc_id = {doc.doc_id: doc.title for doc in corpus}

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    unchanged = 0
    with output_path.open("w", encoding="utf-8") as out:
        for record in _iter_jsonl(input_path):
            positive_title = title_by_doc_id.get(str(record.get("positive_doc_id")))
            expanded = build_expanded_query(
                record,
                positive_doc_title=positive_title,
                max_tokens=args.max_tokens,
            )
            if not expanded:
                expanded = str(record.get("target_query") or "")
            new_record = dict(record)
            metadata = dict(new_record.get("metadata") or {})
            metadata["original_target_query"] = new_record.get("target_query")
            metadata["expanded_positive_title"] = positive_title
            metadata["expanded_target_tokens"] = tokenize(expanded)
            new_record["metadata"] = metadata
            if expanded == new_record.get("target_query"):
                unchanged += 1
            new_record["target_query"] = expanded
            out.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "written": written,
                "unchanged": unchanged,
                "max_tokens": args.max_tokens,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
