from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from src.eval.gene_symbols import canonical_symbol, load_hgnc_alias_map, normalize_edges, normalize_gene_names
from src.eval.metrics import aupr
from src.utils.config import load_config


def _mean_expression(adata: sc.AnnData) -> np.ndarray:
    X = adata.X
    mean = X.mean(axis=0)
    if hasattr(mean, "A1"):
        return mean.A1
    return np.asarray(mean).ravel()


def _parse_source(
    label: str,
    delimiter: str | None,
    allow_multi: bool,
    alias_map: Dict[str, str],
    control_labels: Iterable[str],
) -> str:
    raw = str(label).strip()
    parts = [raw]
    if delimiter and delimiter in raw:
        parts = [part.strip() for part in raw.split(delimiter) if part.strip()]
    control_set = {str(item).strip().lower() for item in control_labels}
    filtered = [part for part in parts if part.lower() not in control_set]
    if not filtered:
        return ""
    if len(filtered) > 1 and not allow_multi:
        return ""
    return canonical_symbol(filtered[0], alias_map)


def _top_targets(values: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or values.size == 0:
        return np.array([], dtype=int)
    k = min(k, values.size)
    idx = np.argpartition(-values, k - 1)[:k]
    return idx[np.argsort(-values[idx])]


def _labels_for_pairs(pairs: List[Tuple[str, str]], edge_set: set[Tuple[str, str]]) -> np.ndarray:
    return np.array([1 if pair in edge_set else 0 for pair in pairs], dtype=int)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.ravel()
    labels = labels.ravel().astype(bool)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    sum_pos = ranks[labels].sum()
    return float((sum_pos - pos * (pos + 1) / 2) / (pos * neg))


def _permutation_p_value(
    labels: np.ndarray,
    scores: np.ndarray,
    num_permutations: int,
    rng: np.random.Generator,
) -> float:
    if num_permutations <= 0 or len(np.unique(labels)) < 2:
        return float("nan")
    observed = aupr(scores, labels)
    count = 0
    for _ in range(num_permutations):
        perm_labels = rng.permutation(labels)
        if aupr(scores, perm_labels) >= observed:
            count += 1
    return (count + 1) / (num_permutations + 1)


def evaluate_perturbation(
    config_path: str,
    output_dir: str | None,
    permutations: int,
    score_mode: str | None,
) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    perturb_cfg = cfg.get("perturbation_validation", {})

    causal_scores_path = Path(paths.get("causal_scores", ""))
    if not causal_scores_path.exists():
        raise FileNotFoundError(f"Missing causal scores: {causal_scores_path}")

    perturb_path = Path(paths.get("perturbation_h5ad", ""))
    if not perturb_path.exists():
        raise FileNotFoundError(f"Missing perturbation dataset: {perturb_path}")

    output_base = Path(output_dir or paths.get("output_dir") or "outputs/perturb_validation")
    output_base.mkdir(parents=True, exist_ok=True)

    alias_map = load_hgnc_alias_map(paths.get("hgnc_alias_tsv"))
    scores_df = pd.read_csv(causal_scores_path, sep="\t")
    scores_df = normalize_edges(scores_df, alias_map)

    adata = sc.read_h5ad(perturb_path)
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    gene_to_idx: Dict[str, int] = {}
    for idx, name in enumerate(gene_names_norm):
        if name and name not in gene_to_idx:
            gene_to_idx[name] = idx

    obs_key = perturb_cfg.get("obs_key", "perturbation")
    control_labels = perturb_cfg.get("control_labels", ["control", "ctrl", "unperturbed"])
    delimiter = perturb_cfg.get("delimiter", "+")
    allow_multi = bool(perturb_cfg.get("allow_multi", False))
    min_cells = int(perturb_cfg.get("min_cells", 10))
    top_k_targets = int(perturb_cfg.get("top_k_targets", 50))
    min_abs_delta = float(perturb_cfg.get("min_abs_delta", 0.0))
    use_abs = bool(perturb_cfg.get("use_abs", True))
    exclude_self = bool(perturb_cfg.get("exclude_self", True))

    if obs_key not in adata.obs:
        raise ValueError(f"Missing obs_key in perturbation dataset: {obs_key}")

    control_mask = adata.obs[obs_key].isin(control_labels)
    if control_mask.sum() == 0:
        raise ValueError("No control cells found for perturbation validation")

    control_mean = _mean_expression(adata[control_mask])
    control_count = int(control_mask.sum())

    edges: List[Dict[str, object]] = []
    source_stats: List[Dict[str, object]] = []
    sources_seen: set[str] = set()

    for label in sorted(adata.obs[obs_key].unique()):
        if label in control_labels:
            continue
        source = _parse_source(label, delimiter, allow_multi, alias_map, control_labels)
        if not source:
            continue
        group_mask = adata.obs[obs_key] == label
        group_count = int(group_mask.sum())
        if group_count < min_cells:
            continue
        group_mean = _mean_expression(adata[group_mask])
        delta = group_mean - control_mean
        if use_abs:
            scores = np.abs(delta)
        else:
            scores = delta.copy()
        if exclude_self and source in gene_to_idx:
            scores[gene_to_idx[source]] = -np.inf
        if min_abs_delta > 0:
            scores[np.abs(delta) < min_abs_delta] = -np.inf
        top_idx = _top_targets(scores, top_k_targets)
        if top_idx.size == 0:
            continue
        sources_seen.add(source)
        source_stats.append(
            {
                "source": source,
                "group_label": label,
                "n_cells": group_count,
                "n_control": control_count,
            }
        )
        for idx in top_idx:
            target = gene_names_norm[idx]
            if not target:
                continue
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "delta": float(delta[idx]),
                    "abs_delta": float(abs(delta[idx])),
                    "n_cells": group_count,
                    "n_control": control_count,
                }
            )

    if not edges:
        raise ValueError("No perturbation edges derived from dataset")

    edges_df = pd.DataFrame(edges).drop_duplicates(subset=["source", "target"])
    edges_df.to_csv(output_base / "perturbation_edges.tsv", sep="\t", index=False)
    pd.DataFrame(source_stats).to_csv(output_base / "perturbation_sources.tsv", sep="\t", index=False)

    edge_set = set(zip(edges_df["source"], edges_df["target"]))
    score_mode = score_mode or perturb_cfg.get("score_mode", "abs")
    rng = np.random.default_rng(int(cfg.get("project", {}).get("seed", 0)))

    results = []
    for intervention in sorted(scores_df["intervention"].unique()):
        subset = scores_df[scores_df["intervention"] == intervention].copy()
        subset = subset[subset["source"].isin(sources_seen)]
        pairs = list(zip(subset["source"], subset["target"]))
        labels = _labels_for_pairs(pairs, edge_set)
        if score_mode == "abs":
            scores = subset["effect_mean"].abs().to_numpy()
        else:
            scores = subset["effect_mean"].to_numpy()

        labels_sum = int(labels.sum())
        n_pairs = int(len(labels))
        if n_pairs == 0 or labels_sum == 0 or labels_sum == n_pairs:
            aupr_val = float("nan")
            auroc_val = float("nan")
            p_value = float("nan")
        else:
            aupr_val = float(aupr(scores, labels))
            auroc_val = _auroc(scores, labels)
            p_value = _permutation_p_value(labels, scores, permutations, rng)

        results.append(
            {
                "reference": "perturbation",
                "score_source": "causal",
                "intervention": intervention,
                "score_mode": score_mode,
                "n_pairs": n_pairs,
                "n_pos": labels_sum,
                "aupr": aupr_val,
                "auroc": auroc_val,
                "perm_p_value": p_value,
                "n_sources": len(sources_seen),
                "top_k_targets": top_k_targets,
                "min_abs_delta": min_abs_delta,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_base / "perturbation_metrics.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate causal scores on perturbation datasets.")
    parser.add_argument("--config", default="configs/perturbation_validation_adamson.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--score-mode", default=None)
    args = parser.parse_args()

    evaluate_perturbation(
        config_path=args.config,
        output_dir=args.output_dir,
        permutations=args.permutations,
        score_mode=args.score_mode,
    )


if __name__ == "__main__":
    main()
