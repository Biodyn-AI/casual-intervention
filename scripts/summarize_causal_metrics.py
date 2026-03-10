from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_metrics(
    manifest_path: str,
    output_path: str,
    n_boot: int,
    seed: int,
    score_sources: List[str] | None,
    interventions: List[str] | None,
) -> None:
    manifest = pd.read_csv(manifest_path)
    if "metrics_path" not in manifest.columns:
        raise ValueError("Manifest must include metrics_path column")
    if "group" not in manifest.columns:
        raise ValueError("Manifest must include group column")

    rows = []
    for _, row in manifest.iterrows():
        metrics_path = Path(row["metrics_path"])
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path, sep="\t")
        df["group"] = row["group"]
        df["run"] = row.get("run", metrics_path.parent.name)
        rows.append(df)

    if not rows:
        raise FileNotFoundError("No metrics files found from manifest")

    metrics = pd.concat(rows, ignore_index=True)
    if score_sources:
        metrics = metrics[metrics["score_source"].isin(score_sources)]
    if interventions:
        metrics = metrics[metrics["intervention"].isin(interventions)]

    rng = np.random.default_rng(seed)
    summary_rows = []
    for (group, reference, score_source, intervention), subset in metrics.groupby(
        ["group", "reference", "score_source", "intervention"]
    ):
        aupr_vals = subset["aupr"].dropna().to_numpy(dtype=float)
        auroc_vals = subset["auroc"].dropna().to_numpy(dtype=float)
        perm_vals = subset["perm_p_value"].dropna().to_numpy(dtype=float)
        n_runs = int(len(subset))
        aupr_mean = float(np.nanmean(aupr_vals)) if aupr_vals.size else float("nan")
        auroc_mean = float(np.nanmean(auroc_vals)) if auroc_vals.size else float("nan")
        aupr_low, aupr_high = _bootstrap_ci(aupr_vals, n_boot, rng) if aupr_vals.size else (float("nan"), float("nan"))
        auroc_low, auroc_high = _bootstrap_ci(auroc_vals, n_boot, rng) if auroc_vals.size else (float("nan"), float("nan"))
        perm_median = float(np.nanmedian(perm_vals)) if perm_vals.size else float("nan")

        summary_rows.append(
            {
                "group": group,
                "reference": reference,
                "score_source": score_source,
                "intervention": intervention,
                "n_runs": n_runs,
                "aupr_mean": aupr_mean,
                "aupr_ci_low": aupr_low,
                "aupr_ci_high": aupr_high,
                "auroc_mean": auroc_mean,
                "auroc_ci_low": auroc_low,
                "auroc_ci_high": auroc_high,
                "perm_p_median": perm_median,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["group", "reference", "score_source", "intervention"]
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize causal metrics across runs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-source", action="append", default=None)
    parser.add_argument("--intervention", action="append", default=None)
    args = parser.parse_args()

    summarize_metrics(
        manifest_path=args.manifest,
        output_path=args.output,
        n_boot=args.n_boot,
        seed=args.seed,
        score_sources=args.score_source,
        interventions=args.intervention,
    )


if __name__ == "__main__":
    main()
