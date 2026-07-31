"""Refresh citation planner scoring CSV and comparison figure."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-score citation planner labels and regenerate paper figures.")
    parser.add_argument(
        "--retrieved",
        type=Path,
        default=Path("output/citation_planner_v2_local/citation_planner_retrieved.jsonl"),
        help="Retrieved planner citations JSONL.",
    )
    parser.add_argument(
        "--ai-labels",
        type=Path,
        default=Path("output/citation_planner_v2_local/citation_planner_labels_ai.jsonl"),
        help="AI label JSONL for planner citations.",
    )
    parser.add_argument(
        "--scored-output",
        type=Path,
        default=Path("output/citation_planner_v2_local/citation_planner_scored_ai_current.jsonl"),
        help="Output scored planner JSONL.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("output/citation_planner_v2_local/citation_planner_summary_ai_current.csv"),
        help="Output planner summary CSV.",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("output/citation_local/citation_summary.csv"),
        help="Baseline summary CSV.",
    )
    parser.add_argument(
        "--figure-pdf",
        type=Path,
        default=Path("docs/figures/citation_planner_comparison.pdf"),
        help="Output figure PDF.",
    )
    parser.add_argument(
        "--figure-png",
        type=Path,
        default=Path("docs/figures/citation_planner_comparison.png"),
        help="Output figure PNG.",
    )
    parser.add_argument("--skip-figure", action="store_true", help="Only refresh scored JSONL and summary CSV.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    _run(
        [
            sys.executable,
            "scripts/score_citation_candidates.py",
            "--input",
            str(args.retrieved),
            "--output",
            str(args.scored_output),
            "--summary-output",
            str(args.summary_output),
            "--ai-labels",
            str(args.ai_labels),
        ]
    )

    if not args.skip_figure:
        _run(
            [
                sys.executable,
                "scripts/plot_citation_comparison.py",
                "--baseline-summary",
                str(args.baseline_summary),
                "--planner-summary",
                str(args.summary_output),
                "--output-pdf",
                str(args.figure_pdf),
                "--output-png",
                str(args.figure_png),
            ]
        )

    print(
        {
            "summary_output": str(args.summary_output),
            "scored_output": str(args.scored_output),
            "figure_pdf": None if args.skip_figure else str(args.figure_pdf),
            "figure_png": None if args.skip_figure else str(args.figure_png),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
