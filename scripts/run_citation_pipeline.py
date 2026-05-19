"""Run the full citation-aware candidate/retrieval/scoring/training-data pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full citation-aware pipeline. This is not a smoke runner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", default="output/citation_full")
    parser.add_argument("--limit", type=int, default=None, help="Optional explicit full-run cap.")
    parser.add_argument("--labels", default=None, help="Optional human label JSONL.")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--skip-candidate-generation", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    candidates = work / "citation_candidates.jsonl"
    retrieved = work / "citation_retrieved.jsonl"
    scored = work / "citation_scored.jsonl"
    summary = work / "citation_summary.csv"
    sft = work / "citation_sft_train.jsonl"
    pref = work / "citation_preferences.jsonl"

    limit_args = [] if args.limit is None else ["--limit", str(args.limit)]
    if not args.skip_candidate_generation:
        _run(
            [
                args.python,
                "scripts/build_citation_candidates.py",
                "--config",
                args.config,
                "--output",
                str(candidates),
                *limit_args,
            ]
        )
    retrieval_cmd = [
        args.python,
        "scripts/run_citation_retrieval.py",
        "--config",
        args.config,
        "--candidates",
        str(candidates),
        "--output",
        str(retrieved),
        *limit_args,
    ]
    if args.no_dense:
        retrieval_cmd.append("--no-dense")
    if args.no_reranker:
        retrieval_cmd.append("--no-reranker")
    if not args.skip_retrieval:
        _run(retrieval_cmd)
    score_cmd = [
        args.python,
        "scripts/score_citation_candidates.py",
        "--input",
        str(retrieved),
        "--output",
        str(scored),
        "--summary-output",
        str(summary),
    ]
    if args.labels:
        score_cmd.extend(["--labels", args.labels])
    _run(score_cmd)
    _run(
        [
            args.python,
            "scripts/build_citation_sft_dataset.py",
            "--input",
            str(scored),
            "--sft-output",
            str(sft),
            "--preference-output",
            str(pref),
        ]
    )
    print(f"candidate_file: {candidates}")
    print(f"retrieved_file: {retrieved}")
    print(f"scored_file: {scored}")
    print(f"summary_file: {summary}")
    print(f"sft_train_file: {sft}")
    print(f"preference_file: {pref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
