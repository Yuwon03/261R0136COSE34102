"""Human-in-the-loop terminal labeling for citation support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from textwrap import shorten


LABEL_KEYS = {
    "g": "supported",
    "p": "partial",
    "b": "unsupported",
    "c": "contradicted",
    "s": "skip",
    "q": "quit",
}


def _existing_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((str(row.get("candidate_id")), str(row.get("chunk_id") or row.get("doc_id"))))
    return keys


def _iter_citations(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            for citation in row.get("citations") or []:
                yield row, citation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label citation candidates as good/bad/supporting evidence.")
    parser.add_argument("--input", required=True, help="Retrieved citation JSONL.")
    parser.add_argument("--labels-output", required=True, help="Append-only human label JSONL.")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--show-chars", type=int, default=1400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.labels_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = _existing_keys(output_path)
    labeled = 0

    print("Labels: [g] supported/good, [p] partial, [b] unsupported/bad, [c] contradicted, [s] skip, [q] quit")
    with output_path.open("a", encoding="utf-8") as out:
        for record, citation in _iter_citations(input_path):
            key = (str(record.get("candidate_id")), str(citation.get("chunk_id") or citation.get("doc_id")))
            if key in seen:
                continue
            print("\n" + "=" * 100)
            print(f"question_id: {record.get('question_id')}")
            print(f"method: {((record.get('search_plan') or {}).get('method'))}")
            print(f"question: {record.get('question')}")
            print(f"queries: {((record.get('search_plan') or {}).get('queries'))}")
            print(f"doc: {citation.get('doc_id')} rank={citation.get('rank')} lang={citation.get('language')} title={citation.get('title')}")
            print("-" * 100)
            print(shorten(str(citation.get("text") or ""), width=args.show_chars, placeholder=" ..."))
            choice = input("label> ").strip().lower()
            label = LABEL_KEYS.get(choice)
            if label == "quit":
                break
            if label in (None, "skip"):
                continue
            payload = {
                "question_id": record.get("question_id"),
                "candidate_id": record.get("candidate_id"),
                "method": (record.get("search_plan") or {}).get("method"),
                "doc_id": citation.get("doc_id"),
                "chunk_id": citation.get("chunk_id") or citation.get("doc_id"),
                "label": label,
                "question": record.get("question"),
                "query": ((record.get("search_plan") or {}).get("queries") or [""])[0],
                "title": citation.get("title"),
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            out.flush()
            seen.add(key)
            labeled += 1
            if args.max_items is not None and labeled >= args.max_items:
                break
    print(json.dumps({"labels_output": str(output_path), "new_labels": labeled}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
