"""Human-in-the-loop terminal labeling for citation support."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def _label_payload(record: dict, citation: dict, label: str) -> dict:
    return {
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


def _existing_method_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[str(row.get("method") or "unknown")] += 1
    return counts


def _iter_citations(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            for citation in row.get("citations") or []:
                yield row, citation


def _format_doc_line(citation: dict) -> str:
    return (
        f"doc={citation.get('doc_id')} rank={citation.get('rank')} "
        f"lang={citation.get('language')} title={citation.get('title')}"
    )


def _parse_batch_labels(raw: str, count: int) -> list[str] | None:
    cleaned = raw.strip().lower()
    if cleaned == "q":
        return ["quit"]
    if not cleaned:
        return None
    tokens = cleaned.split()
    if len(tokens) == 1 and len(tokens[0]) > 1:
        tokens = list(tokens[0])
    if len(tokens) != count:
        print(f"Expected {count} labels, got {len(tokens)}. Use labels like: b b p s g")
        return None
    labels = []
    for token in tokens:
        label = LABEL_KEYS.get(token)
        if label is None:
            print(f"Unknown label: {token}")
            return None
        labels.append(label)
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label citation candidates as good/bad/supporting evidence.")
    parser.add_argument("--input", required=True, help="Retrieved citation JSONL.")
    parser.add_argument("--labels-output", required=True, help="Append-only human label JSONL.")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--show-chars", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of citations to label per prompt.")
    parser.add_argument("--balanced", action="store_true", help="Stop labeling methods that reached --per-method.")
    parser.add_argument("--per-method", type=int, default=None, help="Target total labels per method.")
    return parser


def _method(record: dict) -> str:
    return str((record.get("search_plan") or {}).get("method") or "unknown")


def _method_complete(args: argparse.Namespace, method_counts: Counter[str], method: str) -> bool:
    return bool(args.balanced and args.per_method is not None and method_counts[method] >= args.per_method)


def _all_methods_complete(args: argparse.Namespace, method_counts: Counter[str]) -> bool:
    if not args.balanced or args.per_method is None:
        return False
    methods = ("raw", "machine_translate", "gold_target", "entity_expand", "hyde", "query2doc", "multilingual_plan")
    return all(method_counts[method] >= args.per_method for method in methods)


def _run_single_labeling(args: argparse.Namespace, seen: set[tuple[str, str]], method_counts: Counter[str], out) -> int:
    labeled = 0
    for record, citation in _iter_citations(Path(args.input)):
        if _all_methods_complete(args, method_counts):
            break
        method = _method(record)
        if _method_complete(args, method_counts, method):
            continue
        key = (str(record.get("candidate_id")), str(citation.get("chunk_id") or citation.get("doc_id")))
        if key in seen:
            continue
        print("\n" + "=" * 100)
        print(f"question_id: {record.get('question_id')}")
        print(f"method: {((record.get('search_plan') or {}).get('method'))}")
        print(f"question: {record.get('question')}")
        print(f"queries: {((record.get('search_plan') or {}).get('queries'))}")
        print(_format_doc_line(citation))
        print("-" * 100)
        print(shorten(str(citation.get("text") or ""), width=args.show_chars, placeholder=" ..."))
        choice = input("label> ").strip().lower()
        label = LABEL_KEYS.get(choice)
        if label == "quit":
            break
        if label in (None, "skip"):
            continue
        out.write(json.dumps(_label_payload(record, citation, label), ensure_ascii=False) + "\n")
        out.flush()
        seen.add(key)
        method_counts[method] += 1
        labeled += 1
        if args.max_items is not None and labeled >= args.max_items:
            break
    return labeled


def _run_batch_labeling(args: argparse.Namespace, seen: set[tuple[str, str]], method_counts: Counter[str], out) -> int:
    labeled = 0
    batch_size = max(1, int(args.batch_size))
    pending: list[tuple[dict, dict, tuple[str, str]]] = []

    def flush_batch() -> bool:
        nonlocal labeled, pending
        if not pending:
            return True
        first_record = pending[0][0]
        print("\n" + "=" * 100)
        print(f"question_id: {first_record.get('question_id')}")
        print(f"method: {((first_record.get('search_plan') or {}).get('method'))}")
        print(f"question: {first_record.get('question')}")
        print(f"queries: {((first_record.get('search_plan') or {}).get('queries'))}")
        print("Labels: g=supported, p=partial, b=bad, c=contradicted, s=skip, q=quit")
        for index, (_, citation, _) in enumerate(pending, start=1):
            print("\n" + "-" * 100)
            print(f"[{index}] {_format_doc_line(citation)}")
            print(shorten(str(citation.get("text") or ""), width=args.show_chars, placeholder=" ..."))
        while True:
            raw = input(f"labels for 1-{len(pending)}> ")
            labels = _parse_batch_labels(raw, len(pending))
            if labels is None:
                continue
            if labels == ["quit"]:
                return False
            break
        for (record, citation, key), label in zip(pending, labels):
            if label == "skip":
                continue
            out.write(json.dumps(_label_payload(record, citation, label), ensure_ascii=False) + "\n")
            seen.add(key)
            method_counts[_method(record)] += 1
            labeled += 1
        out.flush()
        pending = []
        return (args.max_items is None or labeled < args.max_items) and not _all_methods_complete(args, method_counts)

    current_candidate_id = None
    for record, citation in _iter_citations(Path(args.input)):
        if _all_methods_complete(args, method_counts):
            break
        method = _method(record)
        if _method_complete(args, method_counts, method):
            continue
        key = (str(record.get("candidate_id")), str(citation.get("chunk_id") or citation.get("doc_id")))
        if key in seen:
            continue
        candidate_id = str(record.get("candidate_id"))
        if pending and candidate_id != current_candidate_id:
            if not flush_batch():
                break
            current_candidate_id = None
        pending.append((record, citation, key))
        current_candidate_id = candidate_id
        if len(pending) >= batch_size:
            if not flush_batch():
                break
    if pending and (args.max_items is None or labeled < args.max_items):
        flush_batch()
    return labeled


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.labels_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = _existing_keys(output_path)
    method_counts = _existing_method_counts(output_path)

    print("Labels: [g] supported/good, [p] partial, [b] unsupported/bad, [c] contradicted, [s] skip, [q] quit")
    if args.balanced and args.per_method is not None:
        print("Current labels by method:")
        for method in ("raw", "machine_translate", "gold_target", "entity_expand", "hyde", "query2doc", "multilingual_plan"):
            print(f"  {method}: {method_counts[method]}/{args.per_method}")
    with output_path.open("a", encoding="utf-8") as out:
        if args.batch_size <= 1:
            labeled = _run_single_labeling(args, seen, method_counts, out)
        else:
            labeled = _run_batch_labeling(args, seen, method_counts, out)
    print(
        json.dumps(
            {
                "labels_output": str(output_path),
                "new_labels": labeled,
                "method_counts": dict(sorted(method_counts.items())),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
