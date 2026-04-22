"""Compare existing run directories and emit analysis artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from crosslingual_rewrite.analysis import analyze_runs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge run summaries into a comparison report."
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Directory that contains per-run subdirectories (output/runs).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for analysis artifacts (output/analysis).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    artifacts = analyze_runs(args.runs_dir, args.output_dir)
    print(f"comparison.csv     -> {artifacts.comparison_path}")
    print(f"slice_summary.csv  -> {artifacts.slice_summary_path}")
    print(f"failure_cases.jsonl-> {artifacts.failure_cases_path}")
    print(f"report.md          -> {artifacts.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
