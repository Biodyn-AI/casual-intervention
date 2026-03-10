from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class NetworkConfig:
    threshold_percentile: float = 99.0
    top_k: int | None = None
    remove_self: bool = True


def _validate_mask(mask: np.ndarray | None, n_genes: int, name: str) -> np.ndarray | None:
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (n_genes,):
        raise ValueError(f"{name} mask must have shape ({n_genes},)")
    return mask


def infer_edges(
    scores: np.ndarray,
    gene_names: Iterable[str],
    config: NetworkConfig,
    source_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
):
    gene_names = list(gene_names)
    n_genes = len(gene_names)

    if scores.shape[0] != n_genes or scores.shape[1] != n_genes:
        raise ValueError("Score matrix shape does not match gene names")

    source_mask = _validate_mask(source_mask, n_genes, "source")
    target_mask = _validate_mask(target_mask, n_genes, "target")

    edges = []
    if config.top_k is not None:
        source_indices = np.arange(n_genes)
        if source_mask is not None:
            source_indices = source_indices[source_mask]
        for i in source_indices:
            candidate_idx = np.arange(n_genes)
            if target_mask is not None:
                candidate_idx = candidate_idx[target_mask]
            if config.remove_self:
                candidate_idx = candidate_idx[candidate_idx != i]
            if candidate_idx.size == 0:
                continue
            row_scores = scores[i, candidate_idx]
            k = min(config.top_k, candidate_idx.size)
            if k <= 0:
                continue
            top_local = np.argpartition(-row_scores, k - 1)[:k]
            for j in candidate_idx[top_local]:
                score = float(scores[i, j])
                if not np.isfinite(score):
                    continue
                edges.append((gene_names[i], gene_names[j], score))
    else:
        mask = np.ones(scores.shape, dtype=bool)
        if source_mask is not None:
            mask &= source_mask[:, None]
        if target_mask is not None:
            mask &= target_mask[None, :]
        if config.remove_self:
            mask &= ~np.eye(n_genes, dtype=bool)
        if not mask.any():
            return pd.DataFrame(edges, columns=["source", "target", "score"])
        threshold = np.percentile(scores[mask], config.threshold_percentile)
        sources, targets = np.where(mask & (scores >= threshold))
        for i, j in zip(sources, targets):
            edges.append((gene_names[i], gene_names[j], float(scores[i, j])))

    return pd.DataFrame(edges, columns=["source", "target", "score"])
