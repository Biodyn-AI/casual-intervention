from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import load_hgnc_alias_map, normalize_edges, normalize_gene_names
from src.eval.metrics import aupr

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover - optional dependency
    roc_auc_score = None


def _parse_named_paths(values: Iterable[str], label: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} entries must use name=path format: '{value}'")
        name, path_str = value.split("=", 1)
        name = name.strip()
        path = Path(path_str.strip())
        if not name:
            raise ValueError(f"{label} entry has an empty name: '{value}'")
        if name in mapping:
            raise ValueError(f"Duplicate {label} name: '{name}'")
        mapping[name] = path
    return mapping


def _load_gene_to_idx(processed_path: Path, alias_map: Dict[str, str]) -> Dict[str, int]:
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed h5ad not found: {processed_path}")
    adata = sc.read_h5ad(processed_path, backed="r")
    normalized = normalize_gene_names(adata.var_names.values, alias_map)
    gene_to_idx: Dict[str, int] = {}
    for idx, gene in enumerate(normalized):
        if gene and gene not in gene_to_idx:
            gene_to_idx[gene] = idx
    if adata.file is not None:
        adata.file.close()
    return gene_to_idx


def _load_env_causal_scores(
    env: str,
    score_path: Path,
    intervention: str,
    alias_map: Dict[str, str],
) -> pd.DataFrame:
    if not score_path.exists():
        raise FileNotFoundError(f"Causal score file not found for '{env}': {score_path}")
    scores_df = pd.read_csv(score_path, sep="\t")
    scores_df = normalize_edges(scores_df, alias_map)
    if intervention:
        scores_df = scores_df[scores_df["intervention"] == intervention].copy()
    if scores_df.empty:
        raise ValueError(f"No rows for intervention '{intervention}' in {score_path}")

    required = ["source", "target", "effect_mean", "effect_std", "n_cells"]
    missing = [col for col in required if col not in scores_df.columns]
    if missing:
        raise ValueError(f"Missing columns in {score_path}: {missing}")

    subset = scores_df[required].drop_duplicates(subset=["source", "target"])
    subset = subset.rename(
        columns={
            "effect_mean": f"{env}_effect_mean",
            "effect_std": f"{env}_effect_std",
            "n_cells": f"{env}_n_cells",
        }
    )
    return subset


def _add_attention_scores(
    edges_df: pd.DataFrame,
    env: str,
    attention_path: Path,
    gene_to_idx: Dict[str, int],
) -> pd.Series:
    if not attention_path.exists():
        raise FileNotFoundError(f"Attention score matrix not found for '{env}': {attention_path}")
    matrix = np.load(attention_path, mmap_mode="r")
    scores = np.full(len(edges_df), np.nan, dtype=float)
    for row_idx, (source, target) in enumerate(zip(edges_df["source"], edges_df["target"])):
        source_idx = gene_to_idx.get(str(source))
        target_idx = gene_to_idx.get(str(target))
        if source_idx is None or target_idx is None:
            continue
        scores[row_idx] = float(matrix[source_idx, target_idx])
    return pd.Series(scores, name=f"{env}_attention")


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if roc_auc_score is None:
        return float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _permutation_aupr_p(
    labels: np.ndarray,
    scores: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    if permutations <= 0 or len(np.unique(labels)) < 2:
        return float("nan")
    observed = float(aupr(scores, labels))
    count = 0
    for _ in range(permutations):
        if float(aupr(scores, rng.permutation(labels))) >= observed:
            count += 1
    return (count + 1) / (permutations + 1)


def _topk_counts(scores: np.ndarray, labels: np.ndarray, top_k: int) -> Tuple[int, int]:
    k = min(int(top_k), int(len(scores)))
    if k <= 0:
        return 0, 0
    ranked = np.argsort(scores)[::-1][:k]
    positives = int(labels[ranked].sum())
    return positives, k


def _topk_permutation_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    permutations: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    observed_pos, k = _topk_counts(scores, labels, top_k)
    if k == 0:
        return float("nan"), float("nan"), float("nan")
    observed_precision = observed_pos / k
    if permutations <= 0:
        return observed_precision, float("nan"), float("nan")

    null_precisions = np.zeros(permutations, dtype=float)
    for idx in range(permutations):
        permuted = rng.permutation(labels)
        perm_pos, _ = _topk_counts(scores, permuted, k)
        null_precisions[idx] = perm_pos / k
    p_value = (int((null_precisions >= observed_precision).sum()) + 1) / (permutations + 1)
    expected_null_precision = float(null_precisions.mean())
    expected_null_pos = expected_null_precision * k
    fdr_estimate = min(1.0, expected_null_pos / max(observed_pos, 1))
    return observed_precision, p_value, fdr_estimate


def _invariance_metrics(
    merged: pd.DataFrame,
    envs: List[str],
) -> pd.DataFrame:
    effect_cols = [f"{env}_effect_mean" for env in envs]
    std_cols = [f"{env}_effect_std" for env in envs]
    n_cols = [f"{env}_n_cells" for env in envs]

    effects = merged[effect_cols].to_numpy(dtype=float)
    effect_abs = np.abs(effects)
    mean_abs = effect_abs.mean(axis=1)
    std_abs = effect_abs.std(axis=1, ddof=0)
    cv_abs = std_abs / np.maximum(mean_abs, 1e-9)

    eps = 1e-9
    positive = (effects > eps).sum(axis=1)
    negative = (effects < -eps).sum(axis=1)
    sign_fraction = np.maximum(positive, negative) / max(len(envs), 1)
    strict_sign_consistency = (np.maximum(positive, negative) == len(envs)).astype(int)

    effect_std = merged[std_cols].to_numpy(dtype=float)
    effect_n = np.maximum(merged[n_cols].to_numpy(dtype=float), 1.0)
    stderr = effect_std / np.sqrt(effect_n)
    rel_stderr = stderr / np.maximum(effect_abs, 1e-6)
    uncertainty_penalty = 1.0 / (1.0 + np.nanmean(rel_stderr, axis=1))

    invariance_score = mean_abs * sign_fraction * np.exp(-cv_abs) * uncertainty_penalty

    result = merged.copy()
    result["causal_mean_abs_score"] = mean_abs
    result["causal_min_abs_score"] = effect_abs.min(axis=1)
    result["effect_abs_cv"] = cv_abs
    result["sign_fraction"] = sign_fraction
    result["strict_sign_consistency"] = strict_sign_consistency
    result["uncertainty_penalty"] = uncertainty_penalty
    result["invariance_score"] = invariance_score
    return result


def _reference_labels(
    edges_df: pd.DataFrame,
    reference_paths: Dict[str, Path],
    alias_map: Dict[str, str],
    confidence_levels: List[str] | None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    labels: Dict[str, np.ndarray] = {}
    result = edges_df.copy()
    pair_tuples = list(zip(result["source"], result["target"]))
    for name, path in reference_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Reference file not found for '{name}': {path}")
        edges = load_dorothea(str(path), confidence_levels=confidence_levels)
        edges = normalize_edges(edges, alias_map)
        edge_set = {(row["source"], row["target"]) for _, row in edges.iterrows()}
        vec = np.array([1 if pair in edge_set else 0 for pair in pair_tuples], dtype=int)
        labels[name] = vec
        result[f"label_{name}"] = vec
    return result, labels


def _write_summary(
    output_path: Path,
    envs: List[str],
    edge_scores: pd.DataFrame,
    ranking_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    top_n: int,
) -> None:
    lines = [
        "# Invariant Causal Edges: Analysis Summary",
        "",
        f"- Environments: {', '.join(envs)}",
        f"- Candidate edges scored in all environments: {len(edge_scores)}",
        f"- Strict sign-consistent edges: {int(edge_scores['strict_sign_consistency'].sum())}",
        f"- Median abs-effect CV: {edge_scores['effect_abs_cv'].median():.4f}",
        "",
        "## AUPR by Method",
        "",
    ]

    if ranking_df.empty:
        lines.append("No ranking metrics were produced.")
    else:
        ranked = ranking_df.sort_values(["reference", "aupr"], ascending=[True, False])
        for _, row in ranked.iterrows():
            aupr_text = "nan" if pd.isna(row["aupr"]) else f"{row['aupr']:.4f}"
            p_text = "nan" if pd.isna(row["aupr_perm_p"]) else f"{row['aupr_perm_p']:.4f}"
            lines.append(
                f"- {row['reference']} | {row['method']}: AUPR={aupr_text}, perm_p={p_text}, "
                f"n_edges={int(row['n_edges'])}, n_pos={int(row['n_pos'])}"
            )

    lines.extend(["", "## Top-K Precision (Invariant Method)", ""])
    invariant_topk = topk_df[topk_df["method"] == "invariance_score"].copy()
    if invariant_topk.empty:
        lines.append("No top-k metrics available for invariant ranking.")
    else:
        invariant_topk = invariant_topk.sort_values(["reference", "top_k"])
        for _, row in invariant_topk.iterrows():
            lines.append(
                f"- {row['reference']} @k={int(row['top_k'])}: precision={row['precision']:.4f}, "
                f"recall={row['recall']:.4f}, perm_p={row['precision_perm_p']:.4f}, "
                f"fdr_est={row['fdr_estimate']:.4f}"
            )

    lines.extend(["", f"## Top {top_n} Invariant Edges", ""])
    top_edges = edge_scores.sort_values("invariance_score", ascending=False).head(top_n)
    for _, row in top_edges.iterrows():
        lines.append(
            f"- {row['source']} -> {row['target']}: invariance={row['invariance_score']:.4f}, "
            f"mean_abs={row['causal_mean_abs_score']:.4f}, cv={row['effect_abs_cv']:.4f}, "
            f"sign_fraction={row['sign_fraction']:.2f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def analyze(
    score_paths: Dict[str, Path],
    processed_paths: Dict[str, Path],
    attention_paths: Dict[str, Path],
    reference_paths: Dict[str, Path],
    output_dir: Path,
    intervention: str,
    min_cells_per_env: int,
    permutations: int,
    top_ks: List[int],
    confidence_levels: List[str] | None,
    alias_path: Path | None,
    seed: int,
) -> None:
    envs = sorted(score_paths.keys())
    for env in envs:
        if env not in processed_paths:
            raise ValueError(f"Missing processed path for environment '{env}'")
        if env not in attention_paths:
            raise ValueError(f"Missing attention path for environment '{env}'")

    alias_map = load_hgnc_alias_map(str(alias_path) if alias_path else None)

    merged: pd.DataFrame | None = None
    for env in envs:
        env_scores = _load_env_causal_scores(env, score_paths[env], intervention, alias_map)
        if merged is None:
            merged = env_scores
        else:
            merged = merged.merge(env_scores, on=["source", "target"], how="inner")
    if merged is None or merged.empty:
        raise ValueError("No overlap across environments after merging causal scores")

    cell_mask = np.ones(len(merged), dtype=bool)
    for env in envs:
        cell_mask &= merged[f"{env}_n_cells"].to_numpy(dtype=float) >= float(min_cells_per_env)
    merged = merged[cell_mask].reset_index(drop=True)
    if merged.empty:
        raise ValueError(
            f"No overlapping edges with at least {min_cells_per_env} cells per environment"
        )

    scored = _invariance_metrics(merged, envs)

    for env in envs:
        gene_to_idx = _load_gene_to_idx(processed_paths[env], alias_map)
        scored[f"{env}_attention"] = _add_attention_scores(
            scored, env, attention_paths[env], gene_to_idx
        )
    attention_cols = [f"{env}_attention" for env in envs]
    scored["attention_mean_score"] = scored[attention_cols].mean(axis=1, skipna=True)

    scored, labels_by_reference = _reference_labels(
        scored, reference_paths, alias_map, confidence_levels
    )

    methods = ["invariance_score", "causal_mean_abs_score", "attention_mean_score"]
    ranking_rows = []
    topk_rows = []
    rng = np.random.default_rng(seed)

    for reference, labels in labels_by_reference.items():
        for method in methods:
            scores = scored[method].to_numpy(dtype=float)
            valid = ~np.isnan(scores)
            if not np.any(valid):
                continue
            scores_valid = scores[valid]
            labels_valid = labels[valid]
            n_edges = int(len(labels_valid))
            n_pos = int(labels_valid.sum())
            if n_edges == 0:
                continue
            if n_pos == 0 or n_pos == n_edges:
                aupr_value = float("nan")
                auroc_value = float("nan")
                aupr_perm_p = float("nan")
            else:
                aupr_value = float(aupr(scores_valid, labels_valid))
                auroc_value = _safe_auroc(labels_valid, scores_valid)
                aupr_perm_p = _permutation_aupr_p(
                    labels_valid, scores_valid, permutations, rng
                )

            ranking_rows.append(
                {
                    "reference": reference,
                    "method": method,
                    "n_edges": n_edges,
                    "n_pos": n_pos,
                    "aupr": aupr_value,
                    "auroc": auroc_value,
                    "aupr_perm_p": aupr_perm_p,
                    "permutations": permutations,
                }
            )

            if n_pos == 0:
                continue
            seen_effective_k: set[int] = set()
            for top_k in top_ks:
                observed_pos, effective_k = _topk_counts(scores_valid, labels_valid, top_k)
                if effective_k == 0:
                    continue
                if effective_k in seen_effective_k:
                    continue
                seen_effective_k.add(effective_k)
                precision = observed_pos / effective_k
                recall = observed_pos / n_pos if n_pos else float("nan")
                precision_perm_p = float("nan")
                fdr_estimate = float("nan")
                if n_pos < n_edges and permutations > 0:
                    precision, precision_perm_p, fdr_estimate = _topk_permutation_metrics(
                        labels_valid, scores_valid, effective_k, permutations, rng
                    )
                    recall = observed_pos / n_pos if n_pos else float("nan")

                topk_rows.append(
                    {
                        "reference": reference,
                        "method": method,
                        "top_k": int(effective_k),
                        "observed_pos": int(observed_pos),
                        "precision": float(precision),
                        "recall": float(recall),
                        "precision_perm_p": float(precision_perm_p),
                        "fdr_estimate": float(fdr_estimate),
                        "permutations": permutations,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    edge_scores_path = output_dir / "edge_scores.tsv"
    ranking_path = output_dir / "ranking_metrics.tsv"
    topk_path = output_dir / "topk_metrics.tsv"
    top_invariant_path = output_dir / "top_invariant_edges.tsv"
    summary_path = output_dir / "analysis_summary.md"

    scored.sort_values("invariance_score", ascending=False).to_csv(
        edge_scores_path, sep="\t", index=False
    )
    pd.DataFrame(ranking_rows).sort_values(
        ["reference", "aupr"], ascending=[True, False]
    ).to_csv(ranking_path, sep="\t", index=False)
    pd.DataFrame(topk_rows).sort_values(
        ["reference", "method", "top_k"]
    ).to_csv(topk_path, sep="\t", index=False)
    scored.sort_values("invariance_score", ascending=False).head(100).to_csv(
        top_invariant_path, sep="\t", index=False
    )

    _write_summary(
        output_path=summary_path,
        envs=envs,
        edge_scores=scored,
        ranking_df=pd.DataFrame(ranking_rows),
        topk_df=pd.DataFrame(topk_rows),
        top_n=20,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute cross-tissue invariant causal edges and benchmark rankings."
    )
    parser.add_argument(
        "--scores",
        action="append",
        required=True,
        help="Per-environment causal score TSV (name=path).",
    )
    parser.add_argument(
        "--processed",
        action="append",
        required=True,
        help="Per-environment processed h5ad used to map genes to attention indices (name=path).",
    )
    parser.add_argument(
        "--attention",
        action="append",
        required=True,
        help="Per-environment attention matrix path (name=path).",
    )
    parser.add_argument(
        "--reference",
        action="append",
        required=True,
        help="Reference edge file (name=path).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/invariant_causal_edges/results",
    )
    parser.add_argument("--intervention", default="ablation")
    parser.add_argument("--min-cells-per-env", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument(
        "--top-k",
        action="append",
        type=int,
        default=None,
        help="Top-k thresholds for precision/recall. Repeat flag for multiple values.",
    )
    parser.add_argument(
        "--confidence",
        action="append",
        default=None,
        help="Confidence levels for DoRothEA-like references. Repeat flag for multiple values.",
    )
    parser.add_argument("--hgnc-alias", default="external/hgnc_complete_set.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analyze(
        score_paths=_parse_named_paths(args.scores, "scores"),
        processed_paths=_parse_named_paths(args.processed, "processed"),
        attention_paths=_parse_named_paths(args.attention, "attention"),
        reference_paths=_parse_named_paths(args.reference, "reference"),
        output_dir=Path(args.output_dir),
        intervention=str(args.intervention),
        min_cells_per_env=int(args.min_cells_per_env),
        permutations=int(args.permutations),
        top_ks=sorted(set(int(k) for k in (args.top_k or [25, 50, 100]) if int(k) > 0)),
        confidence_levels=list(args.confidence) if args.confidence else ["A", "B", "C", "D"],
        alias_path=Path(args.hgnc_alias) if args.hgnc_alias else None,
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
