"""Automatically label citation support with an LLM judge.

This writes the same append-only label schema as label_citations.py, with
extra audit fields such as label_source, confidence, and reason.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


METHODS = ("raw", "machine_translate", "gold_target", "entity_expand", "hyde", "query2doc", "multilingual_plan")
VALID_LABELS = {"supported", "partial", "unsupported", "contradicted"}


SYSTEM_PROMPT = """You are a strict citation support judge.

Task:
Given a Korean or mixed-language question, the search query used, and citation snippets, label whether each citation helps answer the question.

Labels:
- supported: The citation directly supports an answer to the question.
- partial: The citation is relevant and contains useful evidence, but does not fully answer/support the question.
- unsupported: The citation is only topically related, irrelevant, too broad, or does not help answer the question.
- contradicted: The citation conflicts with the expected answer or makes the proposed answer false.

Important:
- Prefer unsupported over partial when the snippet only shares entities/keywords.
- A source must help answer the actual question, not just mention a related term.
- Judge the citation snippet shown here. Do not use outside knowledge.
- Return JSON only, following the schema.
"""


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _label_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("candidate_id")), str(row.get("chunk_id") or row.get("doc_id")))


def _read_label_file(path: Path | None) -> tuple[set[tuple[str, str]], Counter[str]]:
    keys: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    if path is None or not path.exists():
        return keys, counts
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add(_label_key(row))
            counts[str(row.get("method") or "unknown")] += 1
    return keys, counts


def _citation_key(record: dict[str, Any], citation: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("candidate_id")), str(citation.get("chunk_id") or citation.get("doc_id")))


def _method(record: dict[str, Any]) -> str:
    return str((record.get("search_plan") or {}).get("method") or "unknown")


def _primary_query(record: dict[str, Any]) -> str:
    queries = (record.get("search_plan") or {}).get("queries") or []
    return str(queries[0]) if queries else ""


def _truncate(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    return text[:max_chars]


def _iter_unlabeled_items(
    input_path: Path,
    *,
    seen: set[tuple[str, str]],
    method_counts: Counter[str],
    per_method: int,
    max_snippet_chars: int,
):
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            method = _method(record)
            if method not in METHODS or method_counts[method] >= per_method:
                continue
            for citation in record.get("citations") or []:
                key = _citation_key(record, citation)
                if key in seen or method_counts[method] >= per_method:
                    continue
                yield {
                    "id": f"{record.get('candidate_id')}::{citation.get('chunk_id') or citation.get('doc_id')}",
                    "question_id": record.get("question_id"),
                    "candidate_id": record.get("candidate_id"),
                    "method": method,
                    "question": record.get("question"),
                    "query": _primary_query(record),
                    "doc_id": citation.get("doc_id"),
                    "chunk_id": citation.get("chunk_id") or citation.get("doc_id"),
                    "rank": citation.get("rank"),
                    "language": citation.get("language"),
                    "title": citation.get("title"),
                    "snippet": _truncate(str(citation.get("text") or ""), max_snippet_chars),
                }


def _build_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"items": items}


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string", "enum": sorted(VALID_LABELS)},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "label", "confidence", "reason"],
                },
            }
        },
        "required": ["labels"],
    }


def _gemini_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string", "enum": sorted(VALID_LABELS)},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "label", "confidence", "reason"],
                },
            }
        },
        "required": ["labels"],
    }


def _extract_text(payload: Any) -> str | None:
    if isinstance(payload, dict):
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        for key in ("text", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        for value in payload.values():
            found = _extract_text(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _extract_text(value)
            if found:
                return found
    return None


def _parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned)


def _call_openai_judge(
    items: list[dict[str, Any]],
    *,
    model: str,
    api_key: str,
    timeout: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    request_payload = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(_build_payload(items), ensure_ascii=False)},
        ],
        "temperature": 0,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "citation_support_labels",
                "schema": _schema(),
                "strict": True,
            }
        },
    }
    body = json.dumps(request_payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request("https://api.openai.com/v1/responses", data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            text = _extract_text(response_payload)
            if not text:
                raise RuntimeError("OpenAI response did not contain output text")
            parsed = _parse_json_text(text)
            labels = parsed.get("labels")
            if not isinstance(labels, list):
                raise RuntimeError("OpenAI response JSON did not contain labels[]")
            return labels
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"judge request failed after retries: {last_error}") from last_error


def _call_gemini_judge(
    items: list[dict[str, Any]],
    *,
    model: str,
    api_key: str,
    timeout: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    prompt = SYSTEM_PROMPT + "\n\n" + json.dumps(_build_payload(items), ensure_ascii=False)
    request_payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": _gemini_schema(),
        },
    }
    body = json.dumps(request_payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    model_path = model if model.startswith("models/") else f"models/{model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={api_key}"
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            text = _extract_text(response_payload)
            if not text:
                raise RuntimeError("Gemini response did not contain text")
            parsed = _parse_json_text(text)
            labels = parsed.get("labels")
            if not isinstance(labels, list):
                raise RuntimeError("Gemini response JSON did not contain labels[]")
            return labels
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Gemini judge request failed after retries: {last_error}") from last_error


def _dry_run_labels(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = []
    for item in items:
        snippet = str(item.get("snippet") or "").lower()
        title = str(item.get("title") or "").lower()
        query_terms = [token for token in str(item.get("query") or "").lower().split() if len(token) > 3]
        overlap = sum(1 for token in query_terms if token.strip("?,.()") in snippet or token.strip("?,.()") in title)
        if overlap >= 4:
            label = "partial"
            confidence = 0.55
        elif overlap >= 2:
            label = "unsupported"
            confidence = 0.45
        else:
            label = "unsupported"
            confidence = 0.65
        labels.append(
            {
                "id": item["id"],
                "label": label,
                "confidence": confidence,
                "reason": "dry-run lexical heuristic; use provider=openai for final labels",
            }
        )
    return labels


def _to_label_row(item: dict[str, Any], judged: dict[str, Any], *, source: str, model: str) -> dict[str, Any]:
    label = str(judged.get("label") or "unsupported")
    if label not in VALID_LABELS:
        label = "unsupported"
    return {
        "question_id": item.get("question_id"),
        "candidate_id": item.get("candidate_id"),
        "method": item.get("method"),
        "doc_id": item.get("doc_id"),
        "chunk_id": item.get("chunk_id") or item.get("doc_id"),
        "label": label,
        "question": item.get("question"),
        "query": item.get("query"),
        "title": item.get("title"),
        "label_source": source,
        "judge_model": model,
        "confidence": float(judged.get("confidence") or 0.0),
        "reason": str(judged.get("reason") or ""),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-label citation support with an LLM judge.")
    parser.add_argument("--input", required=True, help="Retrieved citation JSONL.")
    parser.add_argument("--human-labels", default=None, help="Existing human label JSONL to skip and count.")
    parser.add_argument("--ai-labels-output", required=True, help="Append-only AI label JSONL.")
    parser.add_argument("--target-total", type=int, default=30000, help="Target total labels including human and AI.")
    parser.add_argument("--per-method", type=int, default=None, help="Override target labels per method.")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--max-snippet-chars", type=int, default=650)
    parser.add_argument("--provider", choices=["openai", "gemini", "dry-run"], default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--env-file", default=".env", help="Optional dotenv file for API keys.")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--limit-batches", type=int, default=None, help="Debug limit for number of judged batches.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_dotenv(Path(args.env_file))
    if args.model is None:
        args.model = "gemini-flash-latest" if args.provider == "gemini" else os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    if args.api_key_env is None:
        args.api_key_env = "OPENAI_API_KEY" if args.provider == "openai" else "GEMINI_API_KEY"
    input_path = Path(args.input)
    output_path = Path(args.ai_labels_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    human_keys, human_counts = _read_label_file(Path(args.human_labels) if args.human_labels else None)
    ai_keys, ai_counts = _read_label_file(output_path)
    seen = set(human_keys) | set(ai_keys)
    method_counts = human_counts + ai_counts
    per_method = int(args.per_method or ((args.target_total + len(METHODS) - 1) // len(METHODS)))

    print(
        json.dumps(
            {
                "provider": args.provider,
                "model": args.model,
                "target_total": args.target_total,
                "per_method": per_method,
                "existing_human": sum(human_counts.values()),
                "existing_ai": sum(ai_counts.values()),
                "method_counts": {method: method_counts[method] for method in METHODS},
            },
            indent=2,
        )
    )

    api_key = os.environ.get(args.api_key_env)
    if args.provider == "gemini" and not api_key and args.api_key_env == "GEMINI_API_KEY":
        api_key = os.environ.get("GOOGLE_API_KEY")
    if args.provider in {"openai", "gemini"} and not api_key:
        raise SystemExit(
            f"{args.api_key_env} is not set. "
            "For Gemini, set GEMINI_API_KEY or GOOGLE_API_KEY. Use --provider dry-run for local plumbing tests."
        )

    batch: list[dict[str, Any]] = []
    batches = 0
    written = 0
    with output_path.open("a", encoding="utf-8") as out:
        for item in _iter_unlabeled_items(
            input_path,
            seen=seen,
            method_counts=method_counts,
            per_method=per_method,
            max_snippet_chars=args.max_snippet_chars,
        ):
            batch.append(item)
            if len(batch) < args.batch_size:
                continue
            if args.provider == "dry-run":
                labels = _dry_run_labels(batch)
            elif args.provider == "gemini":
                labels = _call_gemini_judge(
                    batch,
                    model=args.model,
                    api_key=str(api_key),
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
            else:
                labels = _call_openai_judge(
                    batch,
                    model=args.model,
                    api_key=str(api_key),
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
            by_id = {str(row.get("id")): row for row in labels}
            for judged_item in batch:
                judged = by_id.get(str(judged_item["id"]))
                if judged is None:
                    raise RuntimeError(f"missing label for item id={judged_item['id']}")
                if float(judged.get("confidence") or 0.0) < args.confidence_threshold:
                    continue
                row = _to_label_row(judged_item, judged, source=args.provider, model=args.model)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen.add((str(row["candidate_id"]), str(row["chunk_id"])))
                method_counts[str(row["method"])] += 1
                written += 1
            out.flush()
            batches += 1
            if args.progress_every > 0 and (batches == 1 or batches % args.progress_every == 0):
                print(
                    json.dumps(
                        {
                            "batches": batches,
                            "written": written,
                            "method_counts": {method: method_counts[method] for method in METHODS},
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            batch = []
            if args.limit_batches is not None and batches >= args.limit_batches:
                break
            if all(method_counts[method] >= per_method for method in METHODS):
                break

        if batch and (args.limit_batches is None or batches < args.limit_batches):
            if args.provider == "dry-run":
                labels = _dry_run_labels(batch)
            elif args.provider == "gemini":
                labels = _call_gemini_judge(
                    batch,
                    model=args.model,
                    api_key=str(api_key),
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
            else:
                labels = _call_openai_judge(
                    batch,
                    model=args.model,
                    api_key=str(api_key),
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
            by_id = {str(row.get("id")): row for row in labels}
            for judged_item in batch:
                judged = by_id.get(str(judged_item["id"]))
                if judged is None or float(judged.get("confidence") or 0.0) < args.confidence_threshold:
                    continue
                row = _to_label_row(judged_item, judged, source=args.provider, model=args.model)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                method_counts[str(row["method"])] += 1
                written += 1
            out.flush()

    print(
        json.dumps(
            {
                "ai_labels_output": str(output_path),
                "new_labels": written,
                "method_counts": {method: method_counts[method] for method in METHODS},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
