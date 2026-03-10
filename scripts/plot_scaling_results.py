from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_scaling_f1(confidence_df: pd.DataFrame, out_dir: Path) -> None:
    subset = confidence_df[
        (confidence_df["metric"] == "f1") & (confidence_df["tissue"] == "kidney")
    ].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for tier in sorted(subset["model_tier"].unique()):
        group = subset[subset["model_tier"] == tier].sort_values("max_cells")
        ax.errorbar(
            group["max_cells"],
            group["mean"],
            yerr=[group["mean"] - group["ci_lower"], group["ci_upper"] - group["mean"]],
            marker="o",
            capsize=3,
            label=tier,
        )
    ax.set_title("Kidney F1 vs Cells (95% CI)")
    ax.set_xlabel("Max cells")
    ax.set_ylabel("F1")
    ax.legend(title="Model tier", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_scaling_f1_kidney.png", dpi=200)
    plt.close(fig)


def _plot_robustness(robust_df: pd.DataFrame, out_dir: Path) -> None:
    subset = robust_df[robust_df["tissue"] == "kidney"].copy()
    if subset.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for tier in sorted(subset["model_tier"].unique()):
        group = subset[subset["model_tier"] == tier].sort_values("max_cells")
        axes[0].plot(group["max_cells"], group["jaccard_mean"], marker="o", label=tier)
        axes[1].plot(group["max_cells"], group["spearman_mean"], marker="o", label=tier)
    axes[0].set_title("Jaccard vs Cells")
    axes[0].set_xlabel("Max cells")
    axes[0].set_ylabel("Jaccard mean")
    axes[1].set_title("Spearman vs Cells")
    axes[1].set_xlabel("Max cells")
    axes[1].set_ylabel("Spearman mean")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(title="Model tier", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_robustness_kidney.png", dpi=200)
    plt.close(fig)


def _plot_cross_tissue(confidence_df: pd.DataFrame, out_dir: Path) -> None:
    subset = confidence_df[
        (confidence_df["metric"] == "f1")
        & (confidence_df["model_tier"] == "large")
        & (confidence_df["max_cells"] == 1000)
    ].copy()
    if subset.empty:
        return
    subset = subset.sort_values("tissue")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(subset["tissue"], subset["mean"], yerr=subset["ci_upper"] - subset["mean"], capsize=4)
    ax.set_title("Large Tier F1 by Tissue (1000 cells)")
    ax.set_xlabel("Tissue")
    ax.set_ylabel("F1")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cross_tissue_large_f1.png", dpi=200)
    plt.close(fig)


def _plot_baselines(
    metrics_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    control_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if metrics_df.empty or baseline_df.empty:
        return
    focus = metrics_df[
        (metrics_df["tissue"] == "kidney")
        & (metrics_df["max_cells"] == 1000)
        & (metrics_df["model_tier"].isin(["small", "medium", "large"]))
    ].copy()
    if focus.empty:
        return
    focus = focus.groupby(["model_tier"])["f1"].mean().reset_index().sort_values("model_tier")
    method_order = ["coexpression", "grnboost2", "genie3"]
    baseline_methods = [m for m in method_order if m in baseline_df["method"].unique()]
    if not baseline_methods:
        return

    def method_series(method: str) -> pd.Series:
        subset = baseline_df[
            (baseline_df["method"] == method)
            & (baseline_df["experiment_id"].str.contains("_kidney_cells1000_"))
        ].copy()
        subset["model_tier"] = subset["experiment_id"].str.split("_").str[0]
        subset = subset.groupby("model_tier")["f1"].mean()
        return subset.reindex(focus["model_tier"])

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(focus))
    total_bars = 1 + len(baseline_methods)
    width = 0.8 / total_bars
    offsets = [(-0.4 + width / 2) + idx * width for idx in range(total_bars)]
    ax.bar([i + offsets[0] for i in x], focus["f1"], width=width, label="attention")
    for idx, method in enumerate(baseline_methods, start=1):
        values = method_series(method)
        ax.bar([i + offsets[idx] for i in x], values, width=width, label=method)

    ax.set_xticks(list(x))
    ax.set_xticklabels(focus["model_tier"])
    ax.set_title("F1 Baseline Comparison (Kidney, 1000 cells)")
    ax.set_xlabel("Model tier")
    ax.set_ylabel("F1")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_baseline_comparison.png", dpi=200)
    plt.close(fig)

    if not control_df.empty:
        control = control_df.copy()
        control["model_tier"] = control["experiment_id"].str.split("_").str[0]
        control = control.groupby("model_tier")["f1"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(control["model_tier"], control["f1"], color="gray")
        ax.set_title("Permuted-Label Control F1 (mean)")
        ax.set_xlabel("Model tier")
        ax.set_ylabel("F1")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_control_permute_labels.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scaling and robustness results.")
    parser.add_argument("--metrics", default="outputs/scaling/run_metrics.csv")
    parser.add_argument("--confidence", default="outputs/scaling/scaling_confidence.csv")
    parser.add_argument("--robustness", default="outputs/scaling/robustness_variance.tsv")
    parser.add_argument("--baselines", default="outputs/scaling/baseline_metrics.csv")
    parser.add_argument("--controls", default="outputs/scaling/control_metrics.csv")
    parser.add_argument("--out-dir", default="outputs/scaling/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    metrics_df = pd.read_csv(args.metrics) if Path(args.metrics).exists() else pd.DataFrame()
    confidence_df = (
        pd.read_csv(args.confidence) if Path(args.confidence).exists() else pd.DataFrame()
    )
    robust_df = (
        pd.read_csv(args.robustness, sep="\t")
        if Path(args.robustness).exists()
        else pd.DataFrame()
    )
    baseline_df = (
        pd.read_csv(args.baselines) if Path(args.baselines).exists() else pd.DataFrame()
    )
    control_df = (
        pd.read_csv(args.controls) if Path(args.controls).exists() else pd.DataFrame()
    )

    _plot_scaling_f1(confidence_df, out_dir)
    _plot_robustness(robust_df, out_dir)
    _plot_cross_tissue(confidence_df, out_dir)
    _plot_baselines(metrics_df, baseline_df, control_df, out_dir)


if __name__ == "__main__":
    main()
