"""Create paper-ready citation planner comparison figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHODS = [
    ("raw", "Raw"),
    ("machine_translate", "MT"),
    ("query2doc", "Query2Doc"),
    ("hyde", "HyDE"),
    ("citation_planner", "Planner"),
]

RETRIEVAL_METRICS = [
    ("recall_at_10", "Recall@10"),
    ("mrr", "MRR"),
    ("ndcg_at_10", "NDCG@10"),
]

SUPPORT_METRICS = [
    ("candidate_score", "Candidate\nScore"),
    ("citation_f1", "Citation\nF1"),
    ("citation_precision", "Citation\nPrecision"),
]

UNSUPPORTED_METRIC = ("unsupported_claim_rate", "Unsupported\nRate")


def _load_summary(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            method = str(row["method"])
            rows[method] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"method", "candidate_count"} and value not in {"", None}
            }
    return rows


def _collect_values(
    baseline_rows: dict[str, dict[str, float]],
    planner_rows: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for method, _label in METHODS:
        source = planner_rows if method == "citation_planner" else baseline_rows
        if method not in source:
            raise KeyError(f"Missing method {method!r} in summary input.")
        values[method] = source[method]
    return values


def _plot_metric_group(ax, values: dict[str, dict[str, float]], metrics, *, title: str) -> None:
    import numpy as np

    colors = {
        "raw": "#8c8c8c",
        "machine_translate": "#4c78a8",
        "query2doc": "#54a24b",
        "hyde": "#f58518",
        "citation_planner": "#b279a2",
    }
    hatches = {
        "raw": "",
        "machine_translate": "",
        "query2doc": "",
        "hyde": "",
        "citation_planner": "///",
    }

    x = np.arange(len(metrics))
    width = 0.15
    offsets = np.linspace(-2, 2, len(METHODS)) * width
    for offset, (method, label) in zip(offsets, METHODS):
        ys = [values[method][metric] for metric, _metric_label in metrics]
        ax.bar(
            x + offset,
            ys,
            width=width,
            label=label,
            color=colors[method],
            hatch=hatches[method],
            edgecolor="black",
            linewidth=0.35,
        )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _metric, label in metrics], fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def build_figure(
    *,
    baseline_summary: Path,
    planner_summary: Path,
    output_pdf: Path,
    output_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    baseline_rows = _load_summary(baseline_summary)
    planner_rows = _load_summary(planner_summary)
    values = _collect_values(baseline_rows, planner_rows)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(8.0, 5.9), constrained_layout=False)
    grid = fig.add_gridspec(2, 2)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]
    _plot_metric_group(axes[0], values, RETRIEVAL_METRICS, title="Gold-Document Retrieval")
    _plot_metric_group(axes[1], values, SUPPORT_METRICS, title="Citation Support Quality")
    _plot_metric_group(axes[2], values, [UNSUPPORTED_METRIC], title="Unsupported Evidence")
    axes[2].text(
        0.5,
        0.94,
        "lower is better",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(METHODS),
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.text(
        0.5,
        0.01,
        "Planner support metrics checked 47,360 labeled citations. Retrieval metrics are label-independent.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#444444",
    )
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.08, right=0.99, wspace=0.28, hspace=0.55)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot citation planner comparison figure.")
    parser.add_argument("--baseline-summary", type=Path, default=Path("output/citation_local/citation_summary.csv"))
    parser.add_argument(
        "--planner-summary",
        type=Path,
        default=Path("output/citation_planner_v2_local/citation_planner_summary_ai_partial_47360.csv"),
    )
    parser.add_argument("--output-pdf", type=Path, default=Path("docs/figures/citation_planner_comparison.pdf"))
    parser.add_argument("--output-png", type=Path, default=Path("docs/figures/citation_planner_comparison.png"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_figure(
        baseline_summary=args.baseline_summary,
        planner_summary=args.planner_summary,
        output_pdf=args.output_pdf,
        output_png=args.output_png,
    )
    print(
        {
            "output_pdf": str(args.output_pdf),
            "output_png": str(args.output_png),
            "baseline_summary": str(args.baseline_summary),
            "planner_summary": str(args.planner_summary),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
