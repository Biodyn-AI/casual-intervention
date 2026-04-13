"""Generate the GRN baseline comparison figure for the PLOS revision.

Reads outputs/grn_baseline_comparison/{kidney,lung,immune}_metrics.tsv and
the paper's published scGPT causal ablation numbers, and writes a single
grouped bar chart to figures/fig_grn_baseline_comparison.png.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "outputs/grn_baseline_comparison"
FIG_PATH = REPO_ROOT / "figures/fig_grn_baseline_comparison.png"

# scGPT causal ablation AUPR from the paper's main kidney/lung/immune runs.
PAPER_SCGPT = {"kidney": 0.6018, "lung": 0.7272, "immune": 0.5955}

METHOD_ORDER = [
    ("scgpt_causal", "scGPT causal (ours)"),
    ("pearson_coexpr", "Pearson"),
    ("spearman_coexpr", "Spearman"),
    ("distance_correlation", "Distance corr"),
    ("partial_correlation", "Partial corr (GGM)"),
    ("mutual_information", "MI"),
    ("clr", "CLR"),
    ("aracne", "ARACNE"),
    ("lasso", "LASSO"),
    ("elasticnet", "Elastic Net"),
    ("grnboost2", "GRNBoost2"),
    ("genie3", "GENIE3"),
]
COLORS = {
    "scgpt_causal":        "#1f77b4",
    "pearson_coexpr":      "#ff7f0e",
    "spearman_coexpr":     "#ffa55e",
    "distance_correlation":"#c27018",
    "partial_correlation": "#2ca02c",
    "mutual_information":  "#9467bd",
    "clr":                 "#b594d5",
    "aracne":              "#7e5099",
    "lasso":               "#8c564b",
    "elasticnet":          "#c08579",
    "grnboost2":           "#d62728",
    "genie3":              "#e66b6c",
}

TISSUES = ["kidney", "lung", "immune"]


def main() -> None:
    rows = []
    for tissue in TISSUES:
        m = pd.read_csv(METRICS_DIR / f"{tissue}_metrics.tsv", sep="\t")
        rows.append({"tissue": tissue, "method": "scgpt_causal", "aupr": PAPER_SCGPT[tissue]})
        for _, r in m.iterrows():
            rows.append({"tissue": tissue, "method": r["method"], "aupr": float(r["aupr"])})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    n_tissues = len(TISSUES)
    n_methods = len(METHOD_ORDER)
    width = 0.85 / n_methods
    xbase = np.arange(n_tissues)

    for i, (key, label) in enumerate(METHOD_ORDER):
        values = []
        for t in TISSUES:
            row = df[(df["tissue"] == t) & (df["method"] == key)]
            values.append(float(row["aupr"].values[0]) if len(row) else np.nan)
        x = xbase - 0.425 + i * width + width / 2
        ax.bar(x, values, width, label=label, color=COLORS[key], edgecolor="black", linewidth=0.4)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="chance (AUPR=0.5)")
    ax.set_xticks(xbase)
    ax.set_xticklabels([t.capitalize() for t in TISSUES])
    ax.set_ylabel("AUPR")
    ax.set_ylim(0.4, 0.85)
    ax.set_title(
        "scGPT causal ablation vs. eleven classical GRN inference baselines\n"
        "(matched cell sample, preprocessing, pair sampling and evidence filter)"
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.92, ncol=2)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    print(f"wrote {FIG_PATH}")


if __name__ == "__main__":
    main()
