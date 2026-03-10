from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SWEEPS = [
    "outputs/atlas/sweeps/kidney/head_ranking_shift.tsv",
    "outputs/atlas/sweeps/lung/head_ranking_shift.tsv",
    "outputs/atlas/sweeps/immune/head_ranking_shift.tsv",
    "outputs/atlas/sweeps/external_krasnow_lung/head_ranking_shift.tsv",
]
DEFAULT_ABLATIONS = [
    "outputs/atlas/kidney/head_ablation_summary.tsv",
    "outputs/atlas/lung/head_ablation_summary.tsv",
    "outputs/atlas/immune/head_ablation_summary.tsv",
    "outputs/atlas/external_krasnow_lung/head_ablation_summary.tsv",
]
DEFAULT_LABELS = ["kidney", "lung", "immune", "external_krasnow_lung"]


def _mean_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def _load_sweep_summary(paths: list[Path], labels: list[str]) -> pd.DataFrame:
    rows = []
    for label, path in zip(labels, paths):
        df = pd.read_csv(path, sep="\t")
        rows.append(
            {
                "tissue": label,
                "mean_spearman": _mean_or_nan(df["spearman"]),
                "mean_jaccard": _mean_or_nan(df["jaccard"]),
            }
        )
    return pd.DataFrame(rows)


def _load_ablation_summary(paths: list[Path], labels: list[str]) -> pd.DataFrame:
    rows = []
    for label, path in zip(labels, paths):
        df = pd.read_csv(path, sep="\t")
        for ablation_type in ["top_head", "random_head"]:
            subset = df[df["ablation_type"] == ablation_type]
            if subset.empty:
                value = float("nan")
            else:
                value = float(subset.iloc[0]["mean_aupr_drop"])
            rows.append(
                {
                    "tissue": label,
                    "ablation_type": ablation_type,
                    "mean_aupr_drop": value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot atlas sweep + ablation summary panel.")
    parser.add_argument("--sweeps", nargs="*", default=[])
    parser.add_argument("--ablations", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument(
        "--output", default="outputs/atlas/atlas_summary_panel.png"
    )
    args = parser.parse_args()

    matplotlib.use("Agg")
    plt.rcParams.update({"figure.dpi": 120})

    sweep_paths = [Path(p) for p in (args.sweeps or DEFAULT_SWEEPS)]
    ablation_paths = [Path(p) for p in (args.ablations or DEFAULT_ABLATIONS)]
    labels = args.labels or DEFAULT_LABELS
    if not (len(sweep_paths) == len(ablation_paths) == len(labels)):
        raise ValueError("Number of sweep/ablation paths and labels must match.")

    sweep_df = _load_sweep_summary(sweep_paths, labels)
    ablation_df = _load_ablation_summary(ablation_paths, labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(labels))
    width = 0.35

    axes[0].bar(
        x - width / 2,
        sweep_df["mean_spearman"],
        width,
        label="Spearman",
        color="#4C72B0",
    )
    axes[0].bar(
        x + width / 2,
        sweep_df["mean_jaccard"],
        width,
        label="Jaccard",
        color="#DD8452",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("Mean similarity")
    axes[0].set_title("Sweep ranking stability")
    axes[0].legend()

    ablation_top = ablation_df[ablation_df["ablation_type"] == "top_head"].set_index("tissue")
    ablation_rand = ablation_df[ablation_df["ablation_type"] == "random_head"].set_index("tissue")
    top_vals = [ablation_top.loc[label, "mean_aupr_drop"] for label in labels]
    rand_vals = [ablation_rand.loc[label, "mean_aupr_drop"] for label in labels]

    axes[1].bar(
        x - width / 2,
        top_vals,
        width,
        label="Top heads",
        color="#55A868",
    )
    axes[1].bar(
        x + width / 2,
        rand_vals,
        width,
        label="Random heads",
        color="#C44E52",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("Mean AUPR drop")
    axes[1].set_title("Head ablation impact")
    axes[1].legend()

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
