from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from src.eval.gene_symbols import canonical_symbol, load_hgnc_alias_map, normalize_gene_names
from src.eval.metrics import aupr


def _discover_seed_entries(
    base_dir: Path,
    mode: str,
    include_seed_labels: Set[str] | None = None,
) -> List[Tuple[str, Path]]:
    entries: List[Tuple[str, Path]] = []
    if mode == "strict":
        base_path = base_dir / "results" / "edge_scores.tsv"
    else:
        base_path = base_dir / "results_min_cells1" / "edge_scores.tsv"
    if base_path.exists():
        entries.append(("seed42", base_path))

    multiseed_dir = base_dir / "multiseed"
    if multiseed_dir.exists():
        for child in sorted(multiseed_dir.glob("seed_*")):
            if not child.is_dir():
                continue
            suffix = child.name.split("seed_", 1)[-1].strip()
            if not suffix.isdigit():
                continue
            seed_label = f"seed{int(suffix)}"
            if mode == "strict":
                path = child / "results" / "edge_scores.tsv"
            else:
                path = child / "results_min_cells1" / "edge_scores.tsv"
            if path.exists():
                entries.append((seed_label, path))

    entries = sorted(
        list({(seed, path) for seed, path in entries}),
        key=lambda item: int(item[0].replace("seed", "")),
    )
    if include_seed_labels is not None:
        entries = [item for item in entries if item[0] in include_seed_labels]
    if not entries:
        raise FileNotFoundError(f"No edge score files found for mode={mode} under {base_dir}")
    return entries


def _load_mode_edge_scores(
    base_dir: Path,
    mode: str,
    include_seed_labels: Set[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for seed, path in _discover_seed_entries(
        base_dir, mode, include_seed_labels=include_seed_labels
    ):
        df = pd.read_csv(path, sep="\t")
        df["mode"] = mode
        df["seed"] = seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _parse_include_seed_labels(values: Iterable[int] | None) -> Set[str] | None:
    if not values:
        return None
    return {f"seed{int(value)}" for value in values}


def _bootstrap_mean_ci(values: np.ndarray, reps: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, float("nan"), float("nan")
    samples = rng.choice(values, size=(reps, values.size), replace=True)
    means = samples.mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return mean, float(lo), float(hi)


def _ranking_delta_bootstrap(base_dir: Path, output_dir: Path, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for mode in ("strict", "relaxed"):
        ranking_path = base_dir / "multiseed" / f"aggregate_{mode}" / "ranking_seed_values.tsv"
        if not ranking_path.exists():
            raise FileNotFoundError(f"Missing ranking seed values: {ranking_path}")
        ranking = pd.read_csv(ranking_path, sep="\t")
        for reference, ref_df in ranking.groupby("reference"):
            pivot = ref_df.pivot(index="seed", columns="method", values="aupr")
            required = {"invariance_score", "attention_mean_score", "causal_mean_abs_score"}
            if not required.issubset(set(pivot.columns)):
                continue
            delta_inv_attn = (
                pivot["invariance_score"].to_numpy() - pivot["attention_mean_score"].to_numpy()
            )
            delta_inv_causal = (
                pivot["invariance_score"].to_numpy() - pivot["causal_mean_abs_score"].to_numpy()
            )
            for label, values in (
                ("inv_minus_attention", delta_inv_attn),
                ("inv_minus_causal", delta_inv_causal),
            ):
                values = values[~np.isnan(values)]
                mean, ci_lo, ci_hi = _bootstrap_mean_ci(values, reps=reps, rng=rng)
                p_gt0 = float((np.sum(values > 0) + 1) / (len(values) + 2)) if len(values) else float("nan")
                rows.append(
                    {
                        "mode": mode,
                        "reference": reference,
                        "delta": label,
                        "n_seeds": int(len(values)),
                        "mean": mean,
                        "bootstrap_ci95_low": ci_lo,
                        "bootstrap_ci95_high": ci_hi,
                        "positive_fraction": float(np.mean(values > 0)) if len(values) else float("nan"),
                        "bayes_p_gt0": p_gt0,
                    }
                )
    out = pd.DataFrame(rows).sort_values(["mode", "reference", "delta"])
    out.to_csv(output_dir / "blocker_delta_bootstrap.tsv", sep="\t", index=False)
    return out


def _safe_aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(aupr(scores, labels))


def _aupr_perm_p(labels: np.ndarray, scores: np.ndarray, permutations: int, rng: np.random.Generator) -> float:
    if permutations <= 0:
        return float("nan")
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    observed = float(aupr(scores, labels))
    null = 0
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        if float(aupr(scores, shuffled)) >= observed:
            null += 1
    return float((null + 1) / (permutations + 1))


def _pooled_ranking_and_overlap(
    base_dir: Path,
    output_dir: Path,
    permutations: int,
    bootstrap_reps: int,
    seed: int,
    include_seed_labels: Set[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pooled_rows = []
    overlap_rows = []
    for mode in ("strict", "relaxed"):
        edge_df = _load_mode_edge_scores(
            base_dir, mode, include_seed_labels=include_seed_labels
        )
        edge_df["pair"] = edge_df["source"].astype(str) + "->" + edge_df["target"].astype(str)
        by_seed_counts = edge_df.groupby("seed")["pair"].nunique()
        overlap_rows.append(
            {
                "mode": mode,
                "n_seed_edge_rows": int(len(edge_df)),
                "n_unique_pairs": int(edge_df["pair"].nunique()),
                "n_seeds": int(edge_df["seed"].nunique()),
                "median_pairs_per_seed": float(by_seed_counts.median()),
                "min_pairs_per_seed": int(by_seed_counts.min()),
                "max_pairs_per_seed": int(by_seed_counts.max()),
            }
        )

        for reference in ("dorothea", "trrust"):
            label_col = f"label_{reference}"
            if label_col not in edge_df.columns:
                continue
            labels = edge_df[label_col].to_numpy(dtype=int)
            for method in (
                "invariance_score",
                "causal_mean_abs_score",
                "attention_mean_score",
            ):
                scores = edge_df[method].to_numpy(dtype=float)
                valid = ~np.isnan(scores)
                labels_valid = labels[valid]
                scores_valid = scores[valid]
                value = _safe_aupr(labels_valid, scores_valid)
                p_val = _aupr_perm_p(labels_valid, scores_valid, permutations=permutations, rng=rng)

                boot_values = []
                if len(scores_valid) and labels_valid.sum() not in (0, len(labels_valid)):
                    n = len(scores_valid)
                    for _ in range(bootstrap_reps):
                        idx = rng.integers(0, n, size=n)
                        boot_values.append(_safe_aupr(labels_valid[idx], scores_valid[idx]))
                boot_arr = np.array([x for x in boot_values if not np.isnan(x)], dtype=float)
                if boot_arr.size:
                    ci_lo, ci_hi = np.quantile(boot_arr, [0.025, 0.975])
                else:
                    ci_lo, ci_hi = float("nan"), float("nan")

                pooled_rows.append(
                    {
                        "mode": mode,
                        "reference": reference,
                        "method": method,
                        "n_pairs": int(len(scores_valid)),
                        "n_pos": int(labels_valid.sum()),
                        "aupr": value,
                        "bootstrap_ci95_low": float(ci_lo),
                        "bootstrap_ci95_high": float(ci_hi),
                        "perm_p": p_val,
                    }
                )

    pooled_df = pd.DataFrame(pooled_rows).sort_values(["mode", "reference", "aupr"], ascending=[True, True, False])
    overlap_df = pd.DataFrame(overlap_rows).sort_values("mode")
    pooled_df.to_csv(output_dir / "blocker_pooled_ranking.tsv", sep="\t", index=False)
    overlap_df.to_csv(output_dir / "blocker_overlap_summary.tsv", sep="\t", index=False)
    return pooled_df, overlap_df


def _component_ablation(
    base_dir: Path,
    output_dir: Path,
    bootstrap_reps: int,
    seed: int,
    include_seed_labels: Set[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    seed_rows = []
    summary_rows = []

    for mode in ("strict", "relaxed"):
        for seed_label, path in _discover_seed_entries(
            base_dir, mode, include_seed_labels=include_seed_labels
        ):
            df = pd.read_csv(path, sep="\t")
            mean_abs = df["causal_mean_abs_score"].to_numpy(dtype=float)
            sign = df["sign_fraction"].to_numpy(dtype=float)
            cv = df["effect_abs_cv"].to_numpy(dtype=float)
            unc = df["uncertainty_penalty"].to_numpy(dtype=float)

            score_map = {
                "invariance_full": mean_abs * sign * np.exp(-cv) * unc,
                "invariance_no_sign": mean_abs * np.exp(-cv) * unc,
                "invariance_no_cv": mean_abs * sign * unc,
                "invariance_no_uncertainty": mean_abs * sign * np.exp(-cv),
                "invariance_sign_only": mean_abs * sign,
                "invariance_cv_only": mean_abs * np.exp(-cv),
                "invariance_uncertainty_only": mean_abs * unc,
                "causal_mean_abs_score": mean_abs,
                "attention_mean_score": df["attention_mean_score"].to_numpy(dtype=float),
            }

            for reference in ("dorothea", "trrust"):
                label_col = f"label_{reference}"
                if label_col not in df.columns:
                    continue
                labels = df[label_col].to_numpy(dtype=int)
                full_aupr = _safe_aupr(labels, score_map["invariance_full"])

                for method, scores in score_map.items():
                    value = _safe_aupr(labels, scores)
                    seed_rows.append(
                        {
                            "mode": mode,
                            "seed": seed_label,
                            "reference": reference,
                            "method": method,
                            "aupr": value,
                            "delta_vs_full": float(value - full_aupr) if not np.isnan(value) and not np.isnan(full_aupr) else float("nan"),
                        }
                    )

    seed_df = pd.DataFrame(seed_rows).sort_values(["mode", "reference", "method", "seed"])

    for (mode, reference, method), group in seed_df.groupby(["mode", "reference", "method"]):
        values = group["aupr"].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        mean, ci_lo, ci_hi = _bootstrap_mean_ci(values, reps=bootstrap_reps, rng=rng)

        deltas = group["delta_vs_full"].to_numpy(dtype=float)
        deltas = deltas[~np.isnan(deltas)]
        if deltas.size:
            d_mean, d_lo, d_hi = _bootstrap_mean_ci(deltas, reps=bootstrap_reps, rng=rng)
        else:
            d_mean, d_lo, d_hi = float("nan"), float("nan"), float("nan")

        summary_rows.append(
            {
                "mode": mode,
                "reference": reference,
                "method": method,
                "n_seeds": int(values.size),
                "aupr_mean": mean,
                "aupr_ci95_low": ci_lo,
                "aupr_ci95_high": ci_hi,
                "delta_vs_full_mean": d_mean,
                "delta_vs_full_ci95_low": d_lo,
                "delta_vs_full_ci95_high": d_hi,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["mode", "reference", "aupr_mean"], ascending=[True, True, False])
    seed_df.to_csv(output_dir / "blocker_component_ablation_seed.tsv", sep="\t", index=False)
    summary_df.to_csv(output_dir / "blocker_component_ablation_summary.tsv", sep="\t", index=False)
    return seed_df, summary_df


def _mean_expression(adata: sc.AnnData) -> np.ndarray:
    mean = adata.X.mean(axis=0)
    if hasattr(mean, "A1"):
        return mean.A1
    return np.asarray(mean).ravel()


def _top_targets(values: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    k = min(k, values.size)
    idx = np.argpartition(-values, k - 1)[:k]
    return idx[np.argsort(-values[idx])]


def _parse_source(
    raw_label: str,
    alias_map: Dict[str, str],
    delimiter: str | None,
    control_labels: Sequence[str],
    allow_multi: bool,
) -> str:
    raw = str(raw_label).strip()
    if not raw:
        return ""
    parts = [raw]
    if delimiter and delimiter in raw:
        parts = [p.strip() for p in raw.split(delimiter) if p.strip()]
    control = {item.lower() for item in control_labels}
    parts = [p for p in parts if p.lower() not in control]
    if not parts:
        return ""
    if len(parts) > 1 and not allow_multi:
        return ""
    return canonical_symbol(parts[0], alias_map)


def _build_perturbation_edge_set(
    dataset_path: Path,
    obs_key: str,
    control_labels: Sequence[str],
    alias_map: Dict[str, str],
    top_k_targets: int,
    min_cells: int,
    delimiter: str | None,
    allow_multi: bool,
) -> Tuple[set[Tuple[str, str]], set[str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing perturbation dataset: {dataset_path}")
    adata = sc.read_h5ad(dataset_path)
    if obs_key not in adata.obs:
        raise ValueError(f"Missing obs key '{obs_key}' in {dataset_path}")

    norm_genes = normalize_gene_names(adata.var_names.values, alias_map)
    gene_to_idx = {}
    for i, gene in enumerate(norm_genes):
        if gene and gene not in gene_to_idx:
            gene_to_idx[gene] = i

    obs_values = adata.obs[obs_key].astype(str)
    control_set = {item.lower() for item in control_labels}
    control_mask = obs_values.str.lower().isin(control_set)
    if int(control_mask.sum()) == 0:
        raise ValueError(f"No control cells for {dataset_path}")
    control_mean = _mean_expression(adata[control_mask])

    edges: set[Tuple[str, str]] = set()
    sources: set[str] = set()
    for raw_label in sorted(obs_values.unique()):
        if str(raw_label).lower() in control_set:
            continue
        source = _parse_source(
            raw_label=raw_label,
            alias_map=alias_map,
            delimiter=delimiter,
            control_labels=control_labels,
            allow_multi=allow_multi,
        )
        if not source:
            continue
        group_mask = obs_values == raw_label
        if int(group_mask.sum()) < min_cells:
            continue
        group_mean = _mean_expression(adata[group_mask])
        delta = group_mean - control_mean
        score = np.abs(delta)
        if source in gene_to_idx:
            score[gene_to_idx[source]] = -np.inf
        top_idx = _top_targets(score, top_k_targets)
        if top_idx.size == 0:
            continue
        sources.add(source)
        for idx in top_idx:
            target = norm_genes[idx]
            if not target:
                continue
            edges.add((source, target))

    return edges, sources


def _source_matched_overlap_permutation(
    top_pairs: List[Tuple[str, str]],
    candidate_pairs: List[Tuple[str, str]],
    candidate_sources: List[str],
    perturb_set: set[Tuple[str, str]],
    reps: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    if not top_pairs or not candidate_pairs:
        return float("nan"), float("nan"), float("nan")

    by_source: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for pair, src in zip(candidate_pairs, candidate_sources):
        by_source[src].append(pair)

    top_set = set(top_pairs)
    observed = float(sum(1 for pair in top_pairs if pair in perturb_set))

    null = np.zeros(reps, dtype=float)
    for idx in range(reps):
        sampled: List[Tuple[str, str]] = []
        for source, _ in top_pairs:
            options = [pair for pair in by_source.get(source, []) if pair not in top_set]
            if not options:
                options = [pair for pair in candidate_pairs if pair not in top_set]
            if not options:
                continue
            sampled.append(options[int(rng.integers(0, len(options)))])
        null[idx] = float(sum(1 for pair in sampled if pair in perturb_set))

    p_val = float((int(np.sum(null >= observed)) + 1) / (reps + 1))
    null_mean = float(null.mean()) if reps > 0 else float("nan")
    # Smoothed fold avoids unstable 1e9 artifacts when null_mean is exactly zero.
    fold = (
        float((observed + 0.5) / (null_mean + 0.5))
        if not np.isnan(null_mean)
        else float("nan")
    )
    return observed, p_val, fold


def _perturbation_validation(
    base_dir: Path,
    output_dir: Path,
    alias_map: Dict[str, str],
    top_ks: Sequence[int],
    top_k_targets: int,
    min_cells: int,
    permutations: int,
    seed: int,
    include_seed_labels: Set[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    datasets = [
        {
            "name": "adamson",
            "path": Path("data/perturb/adamson/perturb_processed_symbols.h5ad"),
            "obs_key": "condition",
            "control_labels": ["ctrl"],
            "delimiter": "+",
            "allow_multi": False,
        },
        {
            "name": "dixit13",
            "path": Path("data/perturb/dixit/perturb_processed_symbols.h5ad"),
            "obs_key": "condition",
            "control_labels": ["control"],
            "delimiter": None,
            "allow_multi": False,
        },
        {
            "name": "shifrut",
            "path": Path("data/perturb/shifrut/perturb_processed_symbols.h5ad"),
            "obs_key": "condition",
            "control_labels": ["control"],
            "delimiter": None,
            "allow_multi": False,
        },
        {
            "name": "dixit7",
            "path": Path("data/perturb/dixit_7_days/perturb_processed_symbols.h5ad"),
            "obs_key": "condition",
            "control_labels": ["control"],
            "delimiter": None,
            "allow_multi": False,
        },
    ]

    perturb_sets: Dict[str, set[Tuple[str, str]]] = {}
    perturb_sources: Dict[str, set[str]] = {}
    perturb_summary_rows = []
    for item in datasets:
        edge_set, source_set = _build_perturbation_edge_set(
            dataset_path=item["path"],
            obs_key=item["obs_key"],
            control_labels=item["control_labels"],
            alias_map=alias_map,
            top_k_targets=top_k_targets,
            min_cells=min_cells,
            delimiter=item["delimiter"],
            allow_multi=item["allow_multi"],
        )
        perturb_sets[item["name"]] = edge_set
        perturb_sources[item["name"]] = source_set
        perturb_summary_rows.append(
            {
                "dataset": item["name"],
                "n_edges": int(len(edge_set)),
                "n_sources": int(len(source_set)),
                "top_k_targets": int(top_k_targets),
                "min_cells": int(min_cells),
            }
        )

    seed_metric_rows = []
    seed_enrichment_rows = []

    for mode in ("strict", "relaxed"):
        for seed_label, path in _discover_seed_entries(
            base_dir, mode, include_seed_labels=include_seed_labels
        ):
            edge_df = pd.read_csv(path, sep="\t")
            edge_pairs = list(zip(edge_df["source"].astype(str), edge_df["target"].astype(str)))

            for dataset_name, edge_set in perturb_sets.items():
                source_set = perturb_sources[dataset_name]
                src_mask = edge_df["source"].astype(str).isin(source_set).to_numpy()
                if int(src_mask.sum()) == 0:
                    continue

                subset = edge_df.loc[src_mask].copy().reset_index(drop=True)
                subset_pairs = list(zip(subset["source"].astype(str), subset["target"].astype(str)))
                labels = np.array([1 if pair in edge_set else 0 for pair in subset_pairs], dtype=int)

                for method in (
                    "invariance_score",
                    "causal_mean_abs_score",
                    "attention_mean_score",
                ):
                    scores = subset[method].to_numpy(dtype=float)
                    valid = ~np.isnan(scores)
                    if int(valid.sum()) == 0:
                        continue
                    labels_valid = labels[valid]
                    scores_valid = scores[valid]
                    value = _safe_aupr(labels_valid, scores_valid)
                    p_val = _aupr_perm_p(labels_valid, scores_valid, permutations=permutations, rng=rng)
                    seed_metric_rows.append(
                        {
                            "mode": mode,
                            "seed": seed_label,
                            "dataset": dataset_name,
                            "method": method,
                            "n_pairs": int(len(scores_valid)),
                            "n_pos": int(labels_valid.sum()),
                            "aupr": value,
                            "perm_p": p_val,
                        }
                    )

                ranked = subset.sort_values("invariance_score", ascending=False).reset_index(drop=True)
                candidate_pairs = list(zip(ranked["source"].astype(str), ranked["target"].astype(str)))
                candidate_sources = list(ranked["source"].astype(str))
                for top_k in top_ks:
                    k = min(int(top_k), len(candidate_pairs))
                    if k <= 0:
                        continue
                    top_pairs = candidate_pairs[:k]
                    observed, p_val, fold = _source_matched_overlap_permutation(
                        top_pairs=top_pairs,
                        candidate_pairs=candidate_pairs,
                        candidate_sources=candidate_sources,
                        perturb_set=edge_set,
                        reps=permutations,
                        rng=rng,
                    )
                    seed_enrichment_rows.append(
                        {
                            "mode": mode,
                            "seed": seed_label,
                            "dataset": dataset_name,
                            "top_k": int(k),
                            "observed_overlap": observed,
                            "perm_p": p_val,
                            "fold_over_source_matched_null": fold,
                            "n_candidates": int(len(candidate_pairs)),
                        }
                    )

    metrics_seed_df = pd.DataFrame(seed_metric_rows).sort_values(
        ["mode", "dataset", "method", "seed"]
    )
    enrich_seed_df = pd.DataFrame(seed_enrichment_rows).sort_values(
        ["mode", "dataset", "top_k", "seed"]
    )

    metric_summary_rows = []
    for (mode, dataset, method), group in metrics_seed_df.groupby(["mode", "dataset", "method"]):
        vals = group["aupr"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        mean, ci_lo, ci_hi = _bootstrap_mean_ci(vals, reps=2000, rng=rng)
        metric_summary_rows.append(
            {
                "mode": mode,
                "dataset": dataset,
                "method": method,
                "n_seeds": int(group["seed"].nunique()),
                "aupr_mean": mean,
                "aupr_ci95_low": ci_lo,
                "aupr_ci95_high": ci_hi,
                "mean_n_pairs": float(group["n_pairs"].mean()),
                "mean_n_pos": float(group["n_pos"].mean()),
                "perm_p_mean": float(group["perm_p"].mean()),
            }
        )

    enrichment_summary_rows = []
    for (mode, dataset, top_k), group in enrich_seed_df.groupby(["mode", "dataset", "top_k"]):
        obs_vals = group["observed_overlap"].to_numpy(dtype=float)
        fold_vals = group["fold_over_source_matched_null"].to_numpy(dtype=float)
        obs_mean, obs_lo, obs_hi = _bootstrap_mean_ci(obs_vals, reps=2000, rng=rng)
        fold_mean, fold_lo, fold_hi = _bootstrap_mean_ci(fold_vals, reps=2000, rng=rng)
        enrichment_summary_rows.append(
            {
                "mode": mode,
                "dataset": dataset,
                "top_k": int(top_k),
                "n_seeds": int(group["seed"].nunique()),
                "observed_overlap_mean": obs_mean,
                "observed_overlap_ci95_low": obs_lo,
                "observed_overlap_ci95_high": obs_hi,
                "fold_mean": fold_mean,
                "fold_ci95_low": fold_lo,
                "fold_ci95_high": fold_hi,
                "perm_p_mean": float(group["perm_p"].mean()),
                "mean_n_candidates": float(group["n_candidates"].mean()),
            }
        )

    perturb_summary_df = pd.DataFrame(perturb_summary_rows).sort_values("dataset")
    metric_summary_df = pd.DataFrame(metric_summary_rows).sort_values(
        ["mode", "dataset", "aupr_mean"], ascending=[True, True, False]
    )
    enrichment_summary_df = pd.DataFrame(enrichment_summary_rows).sort_values(
        ["mode", "dataset", "top_k"]
    )

    perturb_summary_df.to_csv(output_dir / "blocker_perturbation_edge_sets.tsv", sep="\t", index=False)
    metrics_seed_df.to_csv(output_dir / "blocker_perturbation_metrics_seed.tsv", sep="\t", index=False)
    metric_summary_df.to_csv(output_dir / "blocker_perturbation_metrics_summary.tsv", sep="\t", index=False)
    enrich_seed_df.to_csv(output_dir / "blocker_perturbation_enrichment_seed.tsv", sep="\t", index=False)
    enrichment_summary_df.to_csv(output_dir / "blocker_perturbation_enrichment_summary.tsv", sep="\t", index=False)

    return perturb_summary_df, metric_summary_df, enrichment_summary_df


def _write_markdown_summary(
    output_dir: Path,
    delta_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    ablation_summary_df: pd.DataFrame,
    perturb_metric_df: pd.DataFrame,
    perturb_enrichment_df: pd.DataFrame,
) -> None:
    lines = [
        "# Invariant Blocker Resolution Summary",
        "",
        "## Seed Delta Bootstrap (invariance minus baselines)",
        "",
    ]
    for _, row in delta_df.iterrows():
        lines.append(
            f"- {row['mode']} / {row['reference']} / {row['delta']}: "
            f"mean={row['mean']:.4f}, CI95=[{row['bootstrap_ci95_low']:.4f}, {row['bootstrap_ci95_high']:.4f}], "
            f"positive_fraction={row['positive_fraction']:.3f}, n_seeds={int(row['n_seeds'])}"
        )

    lines.extend(["", "## Overlap Scale", ""])
    for _, row in overlap_df.iterrows():
        lines.append(
            f"- {row['mode']}: unique_pairs={int(row['n_unique_pairs'])}, "
            f"seed_edge_rows={int(row['n_seed_edge_rows'])}, n_seeds={int(row['n_seeds'])}, "
            f"median_pairs_per_seed={row['median_pairs_per_seed']:.1f}"
        )

    lines.extend(["", "## Pooled Ranking", ""])
    for mode in ("strict", "relaxed"):
        mode_df = pooled_df[pooled_df["mode"] == mode]
        if mode_df.empty:
            continue
        lines.append(f"### {mode}")
        for reference in sorted(mode_df["reference"].unique()):
            ref_df = mode_df[mode_df["reference"] == reference].sort_values("aupr", ascending=False)
            lines.append(f"- {reference}")
            for _, row in ref_df.iterrows():
                lines.append(
                    f"  - {row['method']}: AUPR={row['aupr']:.4f}, "
                    f"CI95=[{row['bootstrap_ci95_low']:.4f}, {row['bootstrap_ci95_high']:.4f}], "
                    f"perm_p={row['perm_p']:.4f}, n_pairs={int(row['n_pairs'])}"
                )

    lines.extend(["", "## Component Ablation (AUPR mean)", ""])
    for mode in ("strict", "relaxed"):
        mode_df = ablation_summary_df[ablation_summary_df["mode"] == mode]
        if mode_df.empty:
            continue
        lines.append(f"### {mode}")
        for reference in sorted(mode_df["reference"].unique()):
            ref_df = mode_df[mode_df["reference"] == reference].sort_values("aupr_mean", ascending=False)
            lines.append(f"- {reference}")
            for _, row in ref_df.iterrows():
                lines.append(
                    f"  - {row['method']}: mean={row['aupr_mean']:.4f}, "
                    f"delta_vs_full={row['delta_vs_full_mean']:.4f}, n_seeds={int(row['n_seeds'])}"
                )

    lines.extend(["", "## Perturbation External Validation", ""])
    for mode in ("strict", "relaxed"):
        mode_df = perturb_metric_df[perturb_metric_df["mode"] == mode]
        if mode_df.empty:
            continue
        lines.append(f"### {mode} ranking")
        for dataset in sorted(mode_df["dataset"].unique()):
            ref_df = mode_df[mode_df["dataset"] == dataset].sort_values("aupr_mean", ascending=False)
            lines.append(f"- {dataset}")
            for _, row in ref_df.iterrows():
                lines.append(
                    f"  - {row['method']}: AUPR mean={row['aupr_mean']:.4f}, "
                    f"perm_p_mean={row['perm_p_mean']:.4f}, mean_n_pairs={row['mean_n_pairs']:.1f}"
                )

    lines.extend(["", "## Perturbation Top-k Enrichment", ""])
    for _, row in perturb_enrichment_df.iterrows():
        lines.append(
            f"- {row['mode']} / {row['dataset']} @k={int(row['top_k'])}: "
            f"fold={row['fold_mean']:.3f}, perm_p_mean={row['perm_p_mean']:.4f}, "
            f"observed_overlap_mean={row['observed_overlap_mean']:.3f}"
        )

    (output_dir / "blocker_resolution_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run blocker-resolution analyses for invariant causal-edge project."
    )
    parser.add_argument("--base-dir", default="outputs/invariant_causal_edges")
    parser.add_argument("--output-dir", default="../reports/invariant_causal_edges/artifacts")
    parser.add_argument("--hgnc-alias", default="external/hgnc_complete_set.txt")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=3000)
    parser.add_argument("--top-k", action="append", type=int, default=None)
    parser.add_argument("--perturb-top-k-targets", type=int, default=50)
    parser.add_argument("--perturb-min-cells", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-seed",
        action="append",
        type=int,
        default=None,
        help="Optional seed numbers to include (e.g., --include-seed 42 --include-seed 43).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    alias_map = load_hgnc_alias_map(args.hgnc_alias)
    top_ks = sorted(set(int(k) for k in (args.top_k or [25, 50]) if int(k) > 0))
    include_seed_labels = _parse_include_seed_labels(args.include_seed)

    delta_df = _ranking_delta_bootstrap(
        base_dir=base_dir,
        output_dir=output_dir,
        reps=int(args.bootstrap_reps),
        seed=int(args.seed),
    )
    pooled_df, overlap_df = _pooled_ranking_and_overlap(
        base_dir=base_dir,
        output_dir=output_dir,
        permutations=int(args.permutations),
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 1,
        include_seed_labels=include_seed_labels,
    )
    _, ablation_summary_df = _component_ablation(
        base_dir=base_dir,
        output_dir=output_dir,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 2,
        include_seed_labels=include_seed_labels,
    )
    _, perturb_metric_df, perturb_enrichment_df = _perturbation_validation(
        base_dir=base_dir,
        output_dir=output_dir,
        alias_map=alias_map,
        top_ks=top_ks,
        top_k_targets=int(args.perturb_top_k_targets),
        min_cells=int(args.perturb_min_cells),
        permutations=int(args.permutations),
        seed=int(args.seed) + 3,
        include_seed_labels=include_seed_labels,
    )

    _write_markdown_summary(
        output_dir=output_dir,
        delta_df=delta_df,
        overlap_df=overlap_df,
        pooled_df=pooled_df,
        ablation_summary_df=ablation_summary_df,
        perturb_metric_df=perturb_metric_df,
        perturb_enrichment_df=perturb_enrichment_df,
    )


if __name__ == "__main__":
    main()
