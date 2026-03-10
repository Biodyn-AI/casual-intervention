from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_labels(value: str | None) -> list[str] | None:
    if not value:
        return None
    labels = [label.strip() for label in value.split(",") if label.strip()]
    return labels or None


def _build_matrix(df: pd.DataFrame, metric: str, labels: list[str] | None) -> tuple[np.ndarray, list[str]]:
    if labels is None:
        labels = sorted(set(df["tissue_a"]) | set(df["tissue_b"]))

    n = len(labels)
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((n, n), dtype=float)
    np.fill_diagonal(matrix, 1.0)

    for row in df.itertuples(index=False):
        a = getattr(row, "tissue_a")
        b = getattr(row, "tissue_b")
        value = getattr(row, metric)
        if a not in index or b not in index:
            continue
        i = index[a]
        j = index[b]
        matrix[i, j] = float(value)
        matrix[j, i] = float(value)

    return matrix, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot head overlap heatmap from pairwise TSV.")
    parser.add_argument(
        "--input",
        default="outputs/atlas/head_overlap_pairwise.tsv",
        help="Path to pairwise overlap TSV.",
    )
    parser.add_argument(
        "--output",
        default="outputs/atlas/head_overlap_heatmap.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--metric",
        default="jaccard",
        help="Metric column to plot (e.g., jaccard).",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated tissue labels to control ordering.",
    )
    args = parser.parse_args()

    matplotlib.use("Agg")
    plt.rcParams.update({"figure.dpi": 120})

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep="\t")
    if args.metric not in df.columns:
        raise ValueError(f"Metric '{args.metric}' not found in {input_path}")

    labels = _parse_labels(args.labels)
    matrix, labels = _build_matrix(df, args.metric, labels)

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), max(4, len(labels) * 1.2)))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(f"Head Overlap ({args.metric})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=args.metric)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
