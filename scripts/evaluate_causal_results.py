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
from src.utils.config import load_config

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover - optional dependency
    roc_auc_score = None


def _parse_reference_args(values: Iterable[str]) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []
    for item in values:
        if ":" in item:
            name, path = item.split(":", 1)
            refs.append((name.strip(), path.strip()))
        else:
            path = item.strip()
            name = Path(path).stem
            refs.append((name, path))
    return refs


def _build_edge_set(edges: pd.DataFrame) -> set[Tuple[str, str]]:
    return {(row["source"], row["target"]) for _, row in edges.iterrows()}


def _labels_for_pairs(pairs: List[Tuple[str, str]], edge_set: set[Tuple[str, str]]) -> np.ndarray:
    return np.array([1 if pair in edge_set else 0 for pair in pairs], dtype=int)


def _attention_scores_for_pairs(
    attention_scores: np.ndarray,
    gene_to_idx: Dict[str, int],
    pairs: List[Tuple[str, str]],
) -> Tuple[np.ndarray, int]:
    scores = np.full(len(pairs), np.nan, dtype=float)
    missing = 0
    for idx, (source, target) in enumerate(pairs):
        source_idx = gene_to_idx.get(source)
        target_idx = gene_to_idx.get(target)
        if source_idx is None or target_idx is None:
            missing += 1
            continue
        scores[idx] = float(attention_scores[source_idx, target_idx])
    return scores, missing


def _maybe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if roc_auc_score is None:
        return float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _permutation_p_value(
    labels: np.ndarray,
    scores: np.ndarray,
    num_permutations: int,
    rng: np.random.Generator,
) -> float:
    if num_permutations <= 0:
        return float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan")
    observed = aupr(scores, labels)
    count = 0
    for _ in range(num_permutations):
        perm_labels = rng.permutation(labels)
        if aupr(scores, perm_labels) >= observed:
            count += 1
    return (count + 1) / (num_permutations + 1)


def _load_processed_adata(
    processed_h5ad: str | None,
    alias_map: Dict[str, str],
) -> Tuple[sc.AnnData | None, Dict[str, int] | None]:
    if not processed_h5ad:
        return None, None
    processed_h5ad = str(processed_h5ad)
    if not Path(processed_h5ad).exists():
        return None, None
    adata = sc.read_h5ad(processed_h5ad)
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    gene_to_idx: Dict[str, int] = {}
    for idx, name in enumerate(gene_names_norm):
        if name and name not in gene_to_idx:
            gene_to_idx[name] = idx
    return adata, gene_to_idx


def _load_attention_baseline(
    attention_path: str | None,
    adata: sc.AnnData | None,
    gene_to_idx: Dict[str, int] | None,
) -> Tuple[np.ndarray | None, Dict[str, int] | None]:
    if not attention_path or adata is None or gene_to_idx is None:
        return None, None
    attention_path = str(attention_path)
    if not Path(attention_path).exists():
        return None, None
    attention_scores = np.load(attention_path)
    return attention_scores, gene_to_idx


def _coexpression_scores_for_pairs(
    adata: sc.AnnData,
    gene_to_idx: Dict[str, int],
    pairs: List[Tuple[str, str]],
) -> Tuple[np.ndarray, int]:
    unique_indices = sorted(
        {gene_to_idx[pair[0]] for pair in pairs if pair[0] in gene_to_idx}
        | {gene_to_idx[pair[1]] for pair in pairs if pair[1] in gene_to_idx}
    )
    if not unique_indices:
        return np.full(len(pairs), np.nan, dtype=float), len(pairs)

    X = adata[:, unique_indices].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0, ddof=1)
    stds[stds == 0] = np.nan
    X = (X - means) / stds

    idx_to_pos = {gene_idx: pos for pos, gene_idx in enumerate(unique_indices)}
    scores = np.full(len(pairs), np.nan, dtype=float)
    missing = 0
    for idx, (source, target) in enumerate(pairs):
        source_idx = gene_to_idx.get(source)
        target_idx = gene_to_idx.get(target)
        if source_idx not in idx_to_pos or target_idx not in idx_to_pos:
            missing += 1
            continue
        source_pos = idx_to_pos[source_idx]
        target_pos = idx_to_pos[target_idx]
        corr = np.nanmean(X[:, source_pos] * X[:, target_pos])
        if np.isnan(corr):
            missing += 1
        scores[idx] = corr
    return scores, missing


def evaluate_causal_results(
    config_path: str,
    output_path: str | None,
    extra_references: List[Tuple[str, str]],
    num_permutations: int,
    score_mode: str | None,
    attention_scores_path: str | None,
    processed_h5ad_path: str | None,
    confidence_override: List[str] | None,
    include_coexpression: bool,
) -> None:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    ci_cfg = cfg.get("causal_intervention", {})
    output_dir = Path(ci_cfg.get("output_dir") or paths.get("causal_output_dir", "outputs/causal"))
    scores_path = output_dir / "causal_scores.tsv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing causal scores: {scores_path}")

    scores_df = pd.read_csv(scores_path, sep="\t")
    alias_map = load_hgnc_alias_map(paths.get("hgnc_alias_tsv"))
    scores_df = normalize_edges(scores_df, alias_map)

    project_seed = cfg.get("project", {}).get("seed", 0)
    rng = np.random.default_rng(int(project_seed))
    score_mode = score_mode or ci_cfg.get("score_mode", "abs")

    confidence = confidence_override or ci_cfg.get("dorothea_confidence") or cfg.get(
        "evaluation", {}
    ).get("dorothea_confidence")
    references: List[Tuple[str, str]] = []
    if paths.get("dorothea_tsv"):
        primary_path = paths["dorothea_tsv"]
        references.append((Path(primary_path).stem, primary_path))
    references.extend(extra_references)

    adata, gene_to_idx = _load_processed_adata(
        processed_h5ad_path or paths.get("processed_h5ad"),
        alias_map,
    )
    attention_scores, attention_gene_to_idx = _load_attention_baseline(
        attention_scores_path or paths.get("attention_scores"),
        adata,
        gene_to_idx,
    )

    results = []
    pairs_cache: Dict[str, List[Tuple[str, str]]] = {}

    for ref_name, ref_path in references:
        edges = load_dorothea(ref_path, confidence_levels=confidence)
        edges = normalize_edges(edges, alias_map)
        edge_set = _build_edge_set(edges)

        for intervention in sorted(scores_df["intervention"].unique()):
            subset = scores_df[scores_df["intervention"] == intervention].copy()
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
                auroc_val = _maybe_auroc(labels, scores)
                p_value = _permutation_p_value(labels, scores, num_permutations, rng)

            results.append(
                {
                    "reference": ref_name,
                    "score_source": "causal",
                    "intervention": intervention,
                    "score_mode": score_mode,
                    "n_pairs": n_pairs,
                    "n_pos": labels_sum,
                    "aupr": aupr_val,
                    "auroc": auroc_val,
                    "perm_p_value": p_value,
                    "perm_n": num_permutations,
                }
            )

            pairs_cache[intervention] = pairs

        if attention_scores is not None and attention_gene_to_idx is not None:
            for intervention, pairs in pairs_cache.items():
                labels = _labels_for_pairs(pairs, edge_set)
                attention_vals, missing = _attention_scores_for_pairs(
                    attention_scores, attention_gene_to_idx, pairs
                )
                valid_mask = ~np.isnan(attention_vals)
                if not np.any(valid_mask):
                    continue
                labels = labels[valid_mask]
                scores = attention_vals[valid_mask]
                labels_sum = int(labels.sum())
                n_pairs = int(len(labels))

                if n_pairs == 0 or labels_sum == 0 or labels_sum == n_pairs:
                    aupr_val = float("nan")
                    auroc_val = float("nan")
                    p_value = float("nan")
                else:
                    aupr_val = float(aupr(scores, labels))
                    auroc_val = _maybe_auroc(labels, scores)
                    p_value = _permutation_p_value(labels, scores, num_permutations, rng)

                results.append(
                    {
                        "reference": ref_name,
                        "score_source": "attention",
                        "intervention": intervention,
                        "score_mode": "raw",
                        "n_pairs": n_pairs,
                        "n_pos": labels_sum,
                        "aupr": aupr_val,
                        "auroc": auroc_val,
                        "perm_p_value": p_value,
                        "perm_n": num_permutations,
                        "attention_missing": missing,
                    }
                )

        if include_coexpression and adata is not None and gene_to_idx is not None:
            for intervention, pairs in pairs_cache.items():
                labels = _labels_for_pairs(pairs, edge_set)
                coexpr_vals, missing = _coexpression_scores_for_pairs(
                    adata, gene_to_idx, pairs
                )
                valid_mask = ~np.isnan(coexpr_vals)
                if not np.any(valid_mask):
                    continue
                labels = labels[valid_mask]
                scores = coexpr_vals[valid_mask]
                labels_sum = int(labels.sum())
                n_pairs = int(len(labels))

                if n_pairs == 0 or labels_sum == 0 or labels_sum == n_pairs:
                    aupr_val = float("nan")
                    auroc_val = float("nan")
                    p_value = float("nan")
                else:
                    aupr_val = float(aupr(scores, labels))
                    auroc_val = _maybe_auroc(labels, scores)
                    p_value = _permutation_p_value(labels, scores, num_permutations, rng)

                results.append(
                    {
                        "reference": ref_name,
                        "score_source": "coexpression",
                        "intervention": intervention,
                        "score_mode": "raw",
                        "n_pairs": n_pairs,
                        "n_pos": labels_sum,
                        "aupr": aupr_val,
                        "auroc": auroc_val,
                        "perm_p_value": p_value,
                        "perm_n": num_permutations,
                        "coexpression_missing": missing,
                    }
                )

    results_df = pd.DataFrame(results)
    if output_path is None:
        output_path = str(output_dir / "causal_metrics.tsv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate causal intervention outputs.")
    parser.add_argument("--config", default="configs/causal_intervention.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="Optional reference edge list (name:path or path).",
    )
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--score-mode", default=None)
    parser.add_argument("--attention-scores", default=None)
    parser.add_argument("--processed-h5ad", default=None)
    parser.add_argument(
        "--confidence",
        action="append",
        default=None,
        help="Override confidence levels (repeat flag for multiple).",
    )
    parser.add_argument("--coexpression", action="store_true")
    args = parser.parse_args()

    evaluate_causal_results(
        config_path=args.config,
        output_path=args.output,
        extra_references=_parse_reference_args(args.reference),
        num_permutations=args.permutations,
        score_mode=args.score_mode,
        attention_scores_path=args.attention_scores,
        processed_h5ad_path=args.processed_h5ad,
        confidence_override=args.confidence,
        include_coexpression=args.coexpression,
    )


if __name__ == "__main__":
    main()
