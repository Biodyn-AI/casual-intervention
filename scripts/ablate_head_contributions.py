from __future__ import annotations

import argparse
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import load_hgnc_alias_map, normalize_edges, normalize_gene_names
from src.eval.metrics import aupr, precision_recall_f1
from src.network.infer import NetworkConfig, infer_edges
from src.utils.config import load_config
def _resolve_output_paths(cfg: dict) -> Path:
    atlas_cfg = cfg.get("atlas", {})
    base_dir = Path(atlas_cfg.get("output_dir", "outputs/atlas"))
    return Path(atlas_cfg.get("metrics_tsv", base_dir / "head_layer_metrics.tsv"))


def _candidate_masks(
    gene_names: np.ndarray,
    true_edges: pd.DataFrame,
    use_sources: bool,
    use_targets: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not (use_sources or use_targets):
        return None, None

    source_mask = None
    target_mask = None
    if use_sources:
        sources = set(true_edges["source"].unique())
        source_mask = np.array([name in sources for name in gene_names], dtype=bool)
    if use_targets:
        targets = set(true_edges["target"].unique())
        target_mask = np.array([name in targets for name in gene_names], dtype=bool)
    return source_mask, target_mask


def _candidate_pair_count(
    n_genes: int,
    source_mask: np.ndarray | None,
    target_mask: np.ndarray | None,
    remove_self: bool,
) -> int:
    source_count = int(source_mask.sum()) if source_mask is not None else n_genes
    target_count = int(target_mask.sum()) if target_mask is not None else n_genes
    total_pairs = source_count * target_count
    if remove_self:
        if source_mask is None and target_mask is None:
            total_pairs -= n_genes
        else:
            if source_mask is None:
                overlap = int(target_mask.sum())
            elif target_mask is None:
                overlap = int(source_mask.sum())
            else:
                overlap = int(np.sum(source_mask & target_mask))
            total_pairs -= overlap
    return max(total_pairs, 0)


def _prepare_ap_samples(
    gene_to_idx: dict[str, int],
    true_edges: pd.DataFrame,
    n_genes: int,
    source_mask: np.ndarray | None,
    target_mask: np.ndarray | None,
    max_pairs: int | None,
    seed: int,
    remove_self: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict]:
    pos_pairs = []
    for row in true_edges.itertuples(index=False):
        src = gene_to_idx.get(row.source)
        tgt = gene_to_idx.get(row.target)
        if src is None or tgt is None:
            continue
        if remove_self and src == tgt:
            continue
        if source_mask is not None and not source_mask[src]:
            continue
        if target_mask is not None and not target_mask[tgt]:
            continue
        pos_pairs.append((int(src), int(tgt)))

    pos_pairs = list(set(pos_pairs))
    pos_set = set(pos_pairs)

    candidate_pairs = _candidate_pair_count(n_genes, source_mask, target_mask, remove_self)
    baseline = float(len(pos_pairs) / candidate_pairs) if candidate_pairs else 0.0

    stats = {
        "candidate_pairs": candidate_pairs,
        "candidate_positives": len(pos_pairs),
        "candidate_positive_rate": baseline,
        "evaluated_pairs": 0,
    }

    if max_pairs is None or max_pairs <= 0 or candidate_pairs == 0:
        return None, None, None, stats

    max_pairs = min(max_pairs, candidate_pairs)
    if max_pairs <= len(pos_pairs):
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(pos_pairs), size=max_pairs, replace=False)
        pos_pairs = [pos_pairs[idx] for idx in selected]
        labels = np.ones(len(pos_pairs), dtype=bool)
        pos_pairs = np.array(pos_pairs, dtype=np.int64)
        stats["evaluated_pairs"] = int(len(pos_pairs))
        return pos_pairs[:, 0], pos_pairs[:, 1], labels, stats

    neg_needed = max_pairs - len(pos_pairs)
    rng = np.random.default_rng(seed)
    neg_pairs: set[tuple[int, int]] = set()

    source_idx = np.where(source_mask)[0] if source_mask is not None else np.arange(n_genes)
    target_idx = np.where(target_mask)[0] if target_mask is not None else np.arange(n_genes)
    if source_idx.size == 0 or target_idx.size == 0:
        return None, None, None, stats

    batch = max(10000, neg_needed)
    while len(neg_pairs) < neg_needed:
        draw = min(batch, neg_needed - len(neg_pairs))
        src_draw = rng.choice(source_idx, size=draw * 2, replace=True)
        tgt_draw = rng.choice(target_idx, size=draw * 2, replace=True)
        for src, tgt in zip(src_draw, tgt_draw):
            if remove_self and src == tgt:
                continue
            pair = (int(src), int(tgt))
            if pair in pos_set or pair in neg_pairs:
                continue
            neg_pairs.add(pair)
            if len(neg_pairs) >= neg_needed:
                break

    all_pairs = pos_pairs + list(neg_pairs)
    labels = np.concatenate(
        [
            np.ones(len(pos_pairs), dtype=bool),
            np.zeros(len(neg_pairs), dtype=bool),
        ]
    )
    all_pairs = np.array(all_pairs, dtype=np.int64)
    stats["evaluated_pairs"] = int(all_pairs.shape[0])
    return all_pairs[:, 0], all_pairs[:, 1], labels, stats


def _compute_metrics(
    scores: np.ndarray,
    gene_names: np.ndarray,
    network_cfg: NetworkConfig,
    true_edges: pd.DataFrame,
    sample_i: np.ndarray | None,
    sample_j: np.ndarray | None,
    sample_labels: np.ndarray | None,
    source_mask: np.ndarray | None,
    target_mask: np.ndarray | None,
) -> dict:
    edges_df = infer_edges(scores, gene_names, network_cfg, source_mask, target_mask)
    pred_edges = edges_df[["source", "target"]].drop_duplicates()
    metrics = precision_recall_f1(pred_edges, true_edges)
    ap_value = np.nan
    if sample_i is not None and sample_labels is not None:
        ap_value = float(aupr(scores[sample_i, sample_j], sample_labels))
    metrics.update(
        {
            "aupr": ap_value,
            "predicted_edges": int(pred_edges.shape[0]),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate head contributions and measure metric drops.")
    parser.add_argument("--config", default="configs/atlas.yaml")
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--top-k-heads", type=int, default=5)
    parser.add_argument("--random-heads", type=int, default=20)
    parser.add_argument("--pr-max-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    atlas_cfg = cfg.get("atlas", {})

    metrics_path = Path(args.metrics_path) if args.metrics_path else _resolve_output_paths(cfg)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path, sep="\t")
    score_col = "aupr_bootstrap_mean" if "aupr_bootstrap_mean" in metrics_df.columns else "aupr"
    metrics_df = metrics_df[["layer", "head", score_col]].copy()
    metrics_df[score_col] = pd.to_numeric(metrics_df[score_col], errors="coerce")
    metrics_df = metrics_df.dropna(subset=[score_col])
    metrics_df = metrics_df.sort_values([score_col, "layer", "head"], ascending=[False, True, True])
    top_heads = metrics_df.head(args.top_k_heads)[["layer", "head"]].itertuples(index=False)
    top_heads = [(int(row.layer), int(row.head)) for row in top_heads]

    adata = ad.read_h5ad(paths["processed_h5ad"])
    score_sum = np.load(paths["attention_scores_head_layer"], mmap_mode="r")
    score_count = np.load(paths["attention_counts_head_layer"], mmap_mode="r")
    shared_denom = np.maximum(score_count, 1) if score_count.ndim == 2 else None

    layers, heads, n_genes, _ = score_sum.shape
    total_heads = layers * heads
    all_heads = [(layer_idx, head_idx) for layer_idx in range(layers) for head_idx in range(heads)]

    rng = np.random.default_rng(args.seed if args.seed is not None else atlas_cfg.get("seed", 42))
    available_random = [head for head in all_heads if head not in set(top_heads)]
    if args.random_heads > 0 and available_random:
        pick_idx = rng.choice(
            len(available_random),
            size=min(args.random_heads, len(available_random)),
            replace=False,
        )
        random_heads = [available_random[int(i)] for i in pick_idx]
    else:
        random_heads = []

    heads_to_store = set(top_heads + random_heads)
    aggregate_sum = np.zeros((n_genes, n_genes), dtype=np.float32)
    head_scores: dict[tuple[int, int], np.ndarray] = {}

    for layer_idx in range(layers):
        for head_idx in range(heads):
            score_sum_slice = score_sum[layer_idx, head_idx]
            if score_count.ndim == 2:
                denom = shared_denom
            else:
                denom = np.maximum(score_count[layer_idx, head_idx], 1)
            scores = np.divide(
                score_sum_slice,
                denom,
                out=np.zeros_like(score_sum_slice, dtype=np.float32),
                where=denom > 0,
            )
            aggregate_sum += scores
            if (layer_idx, head_idx) in heads_to_store:
                head_scores[(layer_idx, head_idx)] = scores

    aggregate_scores = aggregate_sum / float(total_heads)

    alias_map = load_hgnc_alias_map(paths.get("hgnc_alias_tsv"))
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    gene_set = set(gene_names_norm)

    true_edges = load_dorothea(
        paths["dorothea_tsv"],
        confidence_levels=cfg.get("evaluation", {}).get("dorothea_confidence"),
    )
    true_edges = normalize_edges(true_edges, alias_map)
    true_edges = true_edges[
        true_edges["source"].isin(gene_set) & true_edges["target"].isin(gene_set)
    ].drop_duplicates()

    network_cfg_data = dict(cfg.get("network", {}))
    use_sources = bool(network_cfg_data.pop("candidate_sources_from_dorothea", False))
    use_targets = bool(network_cfg_data.pop("candidate_targets_from_dorothea", False))
    network_cfg = NetworkConfig(**network_cfg_data)

    source_mask, target_mask = _candidate_masks(gene_names_norm, true_edges, use_sources, use_targets)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names_norm)}

    max_pairs = args.pr_max_pairs if args.pr_max_pairs is not None else atlas_cfg.get("pr_max_pairs", 1_000_000)
    sample_i, sample_j, sample_labels, sample_stats = _prepare_ap_samples(
        gene_to_idx,
        true_edges,
        n_genes,
        source_mask,
        target_mask,
        max_pairs,
        rng.integers(0, 1_000_000),
        network_cfg.remove_self,
    )

    baseline = _compute_metrics(
        aggregate_scores,
        gene_names_norm,
        network_cfg,
        true_edges,
        sample_i,
        sample_j,
        sample_labels,
        source_mask,
        target_mask,
    )

    rows = [
        {
            "ablation_type": "baseline",
            "layer": -1,
            "head": -1,
            **baseline,
            "precision_drop": 0.0,
            "recall_drop": 0.0,
            "f1_drop": 0.0,
            "aupr_drop": 0.0,
            "predicted_edges_drop": 0,
        }
    ]

    def record_ablation(head: tuple[int, int], ablation_type: str) -> None:
        head_scores_arr = head_scores.get(head)
        if head_scores_arr is None:
            return
        ablated_scores = (aggregate_sum - head_scores_arr) / float(total_heads - 1)
        metrics = _compute_metrics(
            ablated_scores,
            gene_names_norm,
            network_cfg,
            true_edges,
            sample_i,
            sample_j,
            sample_labels,
            source_mask,
            target_mask,
        )
        rows.append(
            {
                "ablation_type": ablation_type,
                "layer": head[0],
                "head": head[1],
                **metrics,
                "precision_drop": baseline["precision"] - metrics["precision"],
                "recall_drop": baseline["recall"] - metrics["recall"],
                "f1_drop": baseline["f1"] - metrics["f1"],
                "aupr_drop": baseline["aupr"] - metrics["aupr"],
                "predicted_edges_drop": baseline["predicted_edges"] - metrics["predicted_edges"],
            }
        )

    for head in top_heads:
        record_ablation(head, "top_head")
    for head in random_heads:
        record_ablation(head, "random_head")

    output_path = metrics_path.with_name("head_ablation_metrics.tsv")
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)

    summary_rows = []
    for ablation_type in ["top_head", "random_head"]:
        subset = [row for row in rows if row["ablation_type"] == ablation_type]
        if not subset:
            continue
        summary_rows.append(
            {
                "ablation_type": ablation_type,
                "n": len(subset),
                "mean_precision_drop": float(np.mean([row["precision_drop"] for row in subset])),
                "mean_recall_drop": float(np.mean([row["recall_drop"] for row in subset])),
                "mean_f1_drop": float(np.mean([row["f1_drop"] for row in subset])),
                "mean_aupr_drop": float(np.mean([row["aupr_drop"] for row in subset])),
            }
        )

    summary_path = metrics_path.with_name("head_ablation_summary.tsv")
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)

    top_subset = [row["aupr_drop"] for row in rows if row["ablation_type"] == "top_head"]
    rand_subset = [row["aupr_drop"] for row in rows if row["ablation_type"] == "random_head"]
    effect_rows = []
    if top_subset and rand_subset:
        top_mean = float(np.mean(top_subset))
        rand_mean = float(np.mean(rand_subset))
        top_std = float(np.std(top_subset, ddof=1)) if len(top_subset) > 1 else float("nan")
        rand_std = float(np.std(rand_subset, ddof=1)) if len(rand_subset) > 1 else float("nan")
        pooled = float("nan")
        if np.isfinite(top_std) and np.isfinite(rand_std):
            pooled = float(np.sqrt((top_std**2 + rand_std**2) / 2.0))
        effect_rows.append(
            {
                "metric": "aupr_drop",
                "top_mean": top_mean,
                "random_mean": rand_mean,
                "mean_diff": top_mean - rand_mean,
                "ratio": (top_mean / rand_mean) if rand_mean else float("nan"),
                "cohens_d": (top_mean - rand_mean) / pooled if pooled and np.isfinite(pooled) else float("nan"),
                "top_n": len(top_subset),
                "random_n": len(rand_subset),
            }
        )
    effect_path = metrics_path.with_name("head_ablation_effects.tsv")
    pd.DataFrame(effect_rows).to_csv(effect_path, sep="\t", index=False)

    print(f"Wrote {output_path}, {summary_path}, and {effect_path}")


if __name__ == "__main__":
    main()
