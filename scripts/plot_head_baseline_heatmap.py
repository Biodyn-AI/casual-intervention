from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = ["aupr", "f1"]
DEFAULT_LABELS = ["kidney", "lung", "immune", "external_krasnow_lung"]


def _parse_list(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or list(default)


def _ordered_unique(values: pd.Series) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _build_matrix(
    df: pd.DataFrame, tissues: list[str], metrics: list[str], value_col: str
) -> np.ndarray:
    matrix = np.full((len(tissues), len(metrics)), np.nan, dtype=float)
    for i, tissue in enumerate(tissues):
        for j, metric in enumerate(metrics):
            rows = df[(df["tissue"] == tissue) & (df["metric"] == metric)]
            if rows.empty:
                continue
            matrix[i, j] = float(rows.iloc[0][value_col])
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot baseline delta heatmap from head_baseline_delta_summary.tsv."
    )
    parser.add_argument(
        "--input",
        default="outputs/atlas/head_baseline_delta_summary.tsv",
        help="Path to head baseline delta summary TSV.",
    )
    parser.add_argument(
        "--output",
        default="outputs/atlas/head_baseline_delta_heatmap.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--metrics",
        default="aupr,f1",
        help="Comma-separated metric list for columns (default: aupr,f1).",
    )
    parser.add_argument(
        "--value",
        default="delta_top_vs_aggregate",
        help="Value column to visualize (e.g., delta_top_vs_aggregate).",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated tissue labels to control ordering.",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    matplotlib.use("Agg")
    plt.rcParams.update({"figure.dpi": 120})

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep="\t")
    if args.value not in df.columns:
        raise ValueError(f"Value column '{args.value}' not found in {input_path}")

    metrics = _parse_list(args.metrics, DEFAULT_METRICS)
    labels = _parse_list(args.labels, DEFAULT_LABELS) if args.labels else _ordered_unique(df["tissue"])

    matrix = _build_matrix(df, labels, metrics, args.value)
    if np.isfinite(matrix).any():
        max_abs = float(np.nanmax(np.abs(matrix)))
        if max_abs <= 0:
            max_abs = 1.0
    else:
        max_abs = 1.0

    fig, ax = plt.subplots(
        figsize=(max(4, len(metrics) * 1.4), max(4, len(labels) * 1.0))
    )
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    title = args.title or f"{args.value} by tissue/metric"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=args.value)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
