"""Build SFT and preference datasets from scored citation candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from crosslingual_rewrite.citation import CitationCandidateRecord  # noqa: E402


SYSTEM_PROMPT = (
    "You are a citation-seeking search planner. Given a Korean or mixed-language "
    "question, output compact JSON with queries, entities, aliases, answer_type, "
    "preferred_source_languages, and source_priority. Optimize the plan for finding "
    "citation-worthy evidence, not for producing a fluent translation."
)


def _plan_json(record: CitationCandidateRecord) -> str:
    return json.dumps(record.search_plan.to_dict(), ensure_ascii=False, sort_keys=True)


def _prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}\n\nSearch plan JSON:"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build citation planner training datasets.")
    parser.add_argument("--input", required=True, help="Scored candidate JSONL.")
    parser.add_argument("--sft-output", required=True)
    parser.add_argument("--preference-output", required=True)
    parser.add_argument("--min-score-margin", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grouped: dict[str, list[CitationCandidateRecord]] = defaultdict(list)
    with Path(args.input).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = CitationCandidateRecord.from_dict(json.loads(line))
            grouped[record.question_id].append(record)

    sft_rows = []
    preference_rows = []
    for question_id, records in grouped.items():
        ordered = sorted(records, key=lambda row: (-row.candidate_score, row.candidate_id))
        if not ordered:
            continue
        chosen = ordered[0]
        sft_rows.append(
            {
                "question_id": question_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chosen.question},
                    {"role": "assistant", "content": _plan_json(chosen)},
                ],
                "candidate_score": chosen.candidate_score,
                "method": chosen.search_plan.method,
            }
        )
        rejected = None
        for candidate in reversed(ordered):
            if chosen.candidate_score - candidate.candidate_score >= args.min_score_margin:
                rejected = candidate
                break
        if rejected is not None:
            preference_rows.append(
                {
                    "question_id": question_id,
                    "prompt": _prompt(chosen.question),
                    "chosen": _plan_json(chosen),
                    "rejected": _plan_json(rejected),
                    "chosen_score": chosen.candidate_score,
                    "rejected_score": rejected.candidate_score,
                    "chosen_method": chosen.search_plan.method,
                    "rejected_method": rejected.search_plan.method,
                }
            )

    for path, rows in ((Path(args.sft_output), sft_rows), (Path(args.preference_output), preference_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"sft_rows": len(sft_rows), "preference_rows": len(preference_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
