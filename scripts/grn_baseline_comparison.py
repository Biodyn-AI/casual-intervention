"""GRN baseline comparison for the PLOS Computational Biology revision.

For each tissue (kidney, lung, immune) this script:

1. Loads the raw Tabula Sapiens h5ad (same files the paper's main runs used).
2. Re-applies the same preprocessing used for the main causal intervention
   runs: normalize_total=10000, log1p, HVG selection restricted to the
   scGPT whole-human vocabulary.
3. Samples a deterministic set of 120 TRRUST positive pairs + 120 random
   control pairs using seed 42, matching the sampling logic in
   scripts/run_causal_interventions.py (_build_pair_list).
4. Applies the same "evidence filter" scGPT's evaluation uses: for each of
   the 120 sampled cells (seed 42), compute the top-1200 expressed genes
   (matching max_genes=1200 in the scGPT dataset); keep only pairs where at
   least one cell contains both source and target in its top-1200 list.
5. Runs four classical GRN inference baselines on the same cell x gene
   matrix:
     - Pearson correlation (coexpression)
     - Mutual information (ARACNE / CLR family)
     - GRNBoost2 reference implementation (SGBM regressors)
     - GENIE3 reference implementation (random-forest regressors)
   The GRNBoost2 and GENIE3 implementations follow Moerman et al. 2019 and
   Huynh-Thu et al. 2010 respectively, using per-target regressors whose
   feature importances provide the edge score.
6. For every surviving pair, records each method's score, then computes
   AUPR, AUROC, and a permutation p-value (1000 label shuffles) matching
   the paper's evaluation protocol in scripts/evaluate_causal_results.py.

The resulting table is the quantitative comparison requested by the
reviewer for the PLOS Computational Biology revision.

Usage:
    python scripts/grn_baseline_comparison.py --tissue all
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import dcor
    HAVE_DCOR = True
except Exception:  # pragma: no cover
    HAVE_DCOR = False


# Raw inputs used by the paper's main runs (see configs/causal_intervention_*.yaml)
TISSUES = {
    "kidney": {
        "raw_h5ad": "data/raw/tabula_sapiens_kidney.h5ad",
        "paper_aupr_trrust_ablation": 0.6018,
    },
    "lung": {
        "raw_h5ad": "data/raw/tabula_sapiens_lung.h5ad",
        "paper_aupr_trrust_ablation": 0.7272,
    },
    "immune": {
        "raw_h5ad": "data/raw/tabula_sapiens_immune_subset_20000.h5ad",
        "paper_aupr_trrust_ablation": 0.5955,
    },
}

# Preprocessing parameters (match configs/causal_intervention_*.yaml)
NORMALIZE_TOTAL = 10_000
HVG_TOP = 5000
MAX_GENES = 1200  # scGPT dataset max_genes used for the evidence filter
MAX_CELLS = 120  # causal_intervention.max_cells used for main tissue runs
MAX_PAIRS = 120  # causal_intervention.max_pairs
RANDOM_CONTROL_PAIRS = 120  # causal_intervention.random_control_pairs

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_vocab(path: Path) -> Dict[str, int]:
    with path.open("r") as f:
        vocab = json.load(f)
    return {str(k).upper(): int(v) for k, v in vocab.items()}


def _read_raw_h5ad(path: Path) -> Tuple[sp.csr_matrix, List[str]]:
    """Read raw expression + var symbols without parsing uns metadata."""
    with h5py.File(path, "r") as f:
        # --- X ---
        if "X/data" in f:
            data = f["X/data"][:]
            indices = f["X/indices"][:]
            indptr = f["X/indptr"][:]
            shape = tuple(int(x) for x in f["X"].attrs["shape"])
            X = sp.csr_matrix((data, indices, indptr), shape=shape)
        else:
            X = np.asarray(f["X"][()])
            X = sp.csr_matrix(X)

        # --- var gene symbols ---
        # Prefer feature_name (symbol) over _index (may be ensembl id)
        gene_symbols: List[str] | None = None
        var_group = f["var"]
        if "feature_name" in var_group:
            feat = var_group["feature_name"]
            if isinstance(feat, h5py.Group) and "codes" in feat and "categories" in feat:
                codes = feat["codes"][:]
                cats = feat["categories"][:]
                cats = [c.decode() if isinstance(c, bytes) else str(c) for c in cats]
                gene_symbols = [cats[c] for c in codes]
            else:
                raw = feat[:]
                gene_symbols = [r.decode() if isinstance(r, bytes) else str(r) for r in raw]
        if gene_symbols is None:
            raw = var_group["_index"][:]
            gene_symbols = [r.decode() if isinstance(r, bytes) else str(r) for r in raw]
    gene_symbols = [g.upper() for g in gene_symbols]
    return X, gene_symbols


def _preprocess(
    X: sp.csr_matrix,
    gene_symbols: List[str],
    vocab: Dict[str, int],
) -> Tuple[sp.csr_matrix, List[str]]:
    """Normalize, log1p, restrict to vocab, and HVG-filter.

    Mirrors the preprocessing in scripts/run_causal_interventions.py and the
    preprocess block of configs/causal_intervention_*.yaml.
    """
    # Restrict columns to scGPT vocab genes (deduplicate, keep first)
    keep: List[int] = []
    seen: Dict[str, int] = {}
    for idx, gene in enumerate(gene_symbols):
        if gene in vocab and gene not in seen:
            seen[gene] = idx
            keep.append(idx)
    keep_arr = np.array(keep, dtype=np.int64)
    X = X[:, keep_arr]
    gene_symbols = [gene_symbols[i] for i in keep]

    # Drop cells with too few genes (min_genes=200) and genes with too few cells (min_cells=3)
    n_cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
    gene_mask = n_cells_per_gene >= 3
    X = X[:, gene_mask]
    gene_symbols = [g for g, m in zip(gene_symbols, gene_mask) if m]
    n_genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
    cell_mask = n_genes_per_cell >= 200
    X = X[cell_mask, :]

    # Normalize total counts to NORMALIZE_TOTAL, then log1p
    totals = np.asarray(X.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    scale = NORMALIZE_TOTAL / totals
    X = X.multiply(scale[:, None]).tocsr()
    X.data = np.log1p(X.data)

    # Highly variable genes (top HVG_TOP by variance of log-normalized values)
    n_genes = X.shape[1]
    if n_genes > HVG_TOP:
        X_dense_sq = X.multiply(X).sum(axis=0)
        sq_means = np.asarray(X_dense_sq).ravel() / X.shape[0]
        means = np.asarray(X.sum(axis=0)).ravel() / X.shape[0]
        variances = sq_means - means ** 2
        top_idx = np.argpartition(-variances, HVG_TOP - 1)[:HVG_TOP]
        top_idx = np.sort(top_idx)
        X = X[:, top_idx]
        gene_symbols = [gene_symbols[i] for i in top_idx]
    return X, gene_symbols


# ---------------------------------------------------------------------------
# Pair sampling (matches scripts/run_causal_interventions.py::_build_pair_list)
# ---------------------------------------------------------------------------


def _load_trrust(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["source", "target", "sign", "pmid"])
    df["source"] = df["source"].astype(str).str.upper()
    df["target"] = df["target"].astype(str).str.upper()
    return df[["source", "target"]].drop_duplicates().reset_index(drop=True)


def _sample_pairs(
    edges: pd.DataFrame,
    gene_set: set,
    max_pairs: int,
    random_controls: int,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[int]]:
    rng = random.Random(seed)
    edges_in = edges[edges["source"].isin(gene_set) & edges["target"].isin(gene_set)]
    true_pairs = [(row.source, row.target) for row in edges_in.itertuples(index=False)]
    if len(true_pairs) > max_pairs:
        true_pairs = rng.sample(true_pairs, max_pairs)
    true_set = set(true_pairs)

    sources = sorted({p[0] for p in true_pairs}) or sorted(gene_set)
    targets = sorted({p[1] for p in true_pairs}) or sorted(gene_set)
    candidates = [
        (s, t)
        for s in sources
        for t in targets
        if s != t and (s, t) not in true_set
    ]
    if random_controls > len(candidates):
        random_controls = len(candidates)
    random_pairs = rng.sample(candidates, random_controls) if random_controls > 0 else []

    pairs = list(true_pairs) + list(random_pairs)
    labels = [1] * len(true_pairs) + [0] * len(random_pairs)
    return pairs, labels


# ---------------------------------------------------------------------------
# Evidence filter (matches scgpt_dataset top-max_genes selection)
# ---------------------------------------------------------------------------


def _evidence_filter(
    X: sp.csr_matrix,
    gene_symbols: List[str],
    cell_indices: np.ndarray,
    pairs: List[Tuple[str, str]],
    max_genes: int = MAX_GENES,
) -> List[bool]:
    """Return a mask over pairs: True if both genes co-occur in the top-max_genes
    expressed genes of at least one of the selected cells.

    Mirrors ScGPTDataset.__getitem__ which keeps nonzero values and then
    truncates to max_genes by expression rank.
    """
    gene_to_col = {g: i for i, g in enumerate(gene_symbols)}
    pair_cols = []
    for src, tgt in pairs:
        si = gene_to_col.get(src)
        ti = gene_to_col.get(tgt)
        pair_cols.append((si, ti))

    n_pairs = len(pairs)
    found = np.zeros(n_pairs, dtype=bool)

    for row_idx in cell_indices:
        row = X[row_idx]
        row_data = row.data
        row_cols = row.indices
        if row_data.size == 0:
            continue
        # nonzero only (matches include_zero=false)
        nz_mask = row_data > 0
        row_data = row_data[nz_mask]
        row_cols = row_cols[nz_mask]
        if row_data.size == 0:
            continue
        # top-max_genes by expression
        if row_data.size > max_genes:
            top = np.argpartition(-row_data, max_genes - 1)[:max_genes]
            cell_top_cols = set(int(c) for c in row_cols[top])
        else:
            cell_top_cols = set(int(c) for c in row_cols)

        for i, (si, ti) in enumerate(pair_cols):
            if found[i]:
                continue
            if si is None or ti is None:
                continue
            if si in cell_top_cols and ti in cell_top_cols:
                found[i] = True

    return list(found)


# ---------------------------------------------------------------------------
# GRN baselines
# ---------------------------------------------------------------------------


def _pearson_scores(expr_df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> np.ndarray:
    X = expr_df.values
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds_nan = stds.copy()
    stds_nan[stds_nan == 0] = np.nan
    Z = (X - means) / stds_nan
    col_idx = {g: i for i, g in enumerate(expr_df.columns)}
    scores = np.full(len(pairs), np.nan, dtype=float)
    for i, (src, tgt) in enumerate(pairs):
        ci = col_idx.get(src)
        cj = col_idx.get(tgt)
        if ci is None or cj is None:
            continue
        zi = Z[:, ci]
        zj = Z[:, cj]
        corr = float(np.nanmean(zi * zj))
        scores[i] = abs(corr) if not np.isnan(corr) else np.nan
    return scores


def _build_mi_cache(expr_df: pd.DataFrame, sources: List[str], seed: int) -> Dict[str, np.ndarray]:
    col_idx = {g: i for i, g in enumerate(expr_df.columns)}
    X = expr_df.values
    cache: Dict[str, np.ndarray] = {}
    for src in sources:
        if src not in col_idx:
            continue
        si = col_idx[src]
        try:
            mi = mutual_info_regression(X, X[:, si], random_state=seed)
        except Exception:
            mi = np.full(X.shape[1], np.nan, dtype=float)
        cache[src] = mi
    return cache


def _mi_scores(
    expr_df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    mi_cache: Dict[str, np.ndarray],
) -> np.ndarray:
    col_idx = {g: i for i, g in enumerate(expr_df.columns)}
    scores = np.full(len(pairs), np.nan, dtype=float)
    for i, (src, tgt) in enumerate(pairs):
        if src not in mi_cache:
            continue
        ci = col_idx.get(tgt)
        if ci is None:
            continue
        scores[i] = float(mi_cache[src][ci])
    return scores


def _clr_scores(
    expr_df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    mi_cache: Dict[str, np.ndarray],
) -> np.ndarray:
    """Context Likelihood of Relatedness (Faith et al. 2007).

    CLR[i, j] = sqrt(max(0, z_i)^2 + max(0, z_j)^2), where
    z_i = (MI[i, j] - mean(MI[i, :])) / std(MI[i, :]) and similarly for z_j.
    This variant uses the row-z-score for the source and treats the target
    column as its own normalization by rerunning MI with the target as
    'query' when possible; when the target is not in the MI cache we fall
    back to the source-side z-score only.
    """
    col_idx = {g: i for i, g in enumerate(expr_df.columns)}
    scores = np.full(len(pairs), np.nan, dtype=float)
    row_stats: Dict[str, Tuple[float, float]] = {}
    for src, mi_row in mi_cache.items():
        mu = float(np.nanmean(mi_row))
        sig = float(np.nanstd(mi_row))
        row_stats[src] = (mu, sig if sig > 0 else 1.0)

    for i, (src, tgt) in enumerate(pairs):
        if src not in mi_cache:
            continue
        ci = col_idx.get(tgt)
        if ci is None:
            continue
        mi_val = float(mi_cache[src][ci])
        mu_s, sig_s = row_stats[src]
        z_src = max(0.0, (mi_val - mu_s) / sig_s)
        # target-side z-score if we also computed MI with target as source
        if tgt in mi_cache and src in col_idx:
            mi_row_t = mi_cache[tgt]
            mu_t = float(np.nanmean(mi_row_t))
            sig_t = float(np.nanstd(mi_row_t)) or 1.0
            z_tgt = max(0.0, (mi_val - mu_t) / sig_t)
        else:
            z_tgt = 0.0
        scores[i] = float(np.sqrt(z_src ** 2 + z_tgt ** 2))
    return scores


def _aracne_scores(
    expr_df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    mi_cache: Dict[str, np.ndarray],
    eps: float = 0.0,
) -> np.ndarray:
    """ARACNE (Margolin et al. 2006) with data processing inequality.

    For each pair (i, j) with mutual information M[i, j], search for a
    mediator k such that M[i, j] < min(M[i, k], M[k, j]) - eps. If such
    a k exists the edge is pruned to 0; otherwise it retains M[i, j].
    Candidate mediators are every gene that appears as a source in the
    MI cache (i.e. every TF already processed by the MI step).
    """
    col_idx = {g: i for i, g in enumerate(expr_df.columns)}
    mediators = list(mi_cache.keys())
    scores = np.full(len(pairs), np.nan, dtype=float)
    for i, (src, tgt) in enumerate(pairs):
        if src not in mi_cache:
            continue
        ci = col_idx.get(tgt)
        if ci is None:
            continue
        m_st = float(mi_cache[src][ci])
        pruned = False
        for k in mediators:
            if k == src or k == tgt:
                continue
            ki = col_idx.get(k)
            if ki is None:
                continue
            m_sk = float(mi_cache[src][ki])
            if tgt in mi_cache:
                m_kt = float(mi_cache[tgt][ki])
            else:
                m_kt = float(mi_cache[k][ci])
            triplet_min = min(m_sk, m_kt)
            if m_st < triplet_min - eps:
                pruned = True
                break
        scores[i] = 0.0 if pruned else m_st
    return scores


def _spearman_scores(expr_df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> np.ndarray:
    col_idx = {g: i for i, g in enumerate(expr_df.columns)}
    X = expr_df.values
    scores = np.full(len(pairs), np.nan, dtype=float)
    for i, (src, tgt) in enumerate(pairs):
        ci = col_idx.get(src)
        cj = col_idx.get(tgt)
        if ci is None or cj is None:
            continue
        a = X[:, ci]
        b = X[:, cj]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        rho, _ = spearmanr(a, b)
        scores[i] = abs(float(rho)) if not np.isnan(rho) else np.nan
    return scores


def _partial_correlation_scores(
    expr_df: pd.DataFrame,
    tf_list: List[str],
    pairs: List[Tuple[str, str]],
    ridge: float = 1e-2,
) -> np.ndarray:
    """Gaussian partial correlation on the TF union, with a ridge-regularized
    precision matrix (small multiple of identity added before inversion).

    For each unique target, we form the covariance of [TFs union target],
    invert it with ridge shrinkage, and extract the partial correlation
    between every source in the TF set and the target.
    """
    col_idx = {g: i for i, g in enumerate(expr_df.columns)}
    tf_present = [tf for tf in tf_list if tf in col_idx]
    if not tf_present:
        return np.full(len(pairs), np.nan, dtype=float)
    tf_pos = {tf: i for i, tf in enumerate(tf_present)}
    X_tf = expr_df.loc[:, tf_present].to_numpy(dtype=np.float64)
    X_tf -= X_tf.mean(axis=0)

    scores = np.full(len(pairs), np.nan, dtype=float)
    per_target: Dict[str, List[int]] = {}
    for i, (_, tgt) in enumerate(pairs):
        per_target.setdefault(tgt, []).append(i)

    for tgt, slots in per_target.items():
        if tgt not in col_idx:
            continue
        y = expr_df[tgt].to_numpy(dtype=np.float64)
        y = y - y.mean()
        if tgt in tf_pos:
            # drop duplicate column so the target appears only once
            Xuse = np.delete(X_tf, tf_pos[tgt], axis=1)
            tfs_use = [tf for tf in tf_present if tf != tgt]
        else:
            Xuse = X_tf
            tfs_use = tf_present
        M = np.column_stack([Xuse, y])
        cov = np.cov(M, rowvar=False)
        cov += ridge * np.eye(cov.shape[0])
        try:
            prec = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        tgt_row = prec.shape[0] - 1
        tgt_tgt = prec[tgt_row, tgt_row]
        tf_to_row = {tf: idx for idx, tf in enumerate(tfs_use)}
        for slot in slots:
            src = pairs[slot][0]
            if src not in tf_to_row:
                continue
            r = tf_to_row[src]
            p_ss = prec[r, r]
            p_st = prec[r, tgt_row]
            if p_ss <= 0 or tgt_tgt <= 0:
                continue
            partial = -p_st / float(np.sqrt(p_ss * tgt_tgt))
            scores[slot] = abs(partial)
    return scores


def _regularized_regression_scores(
    method: str,
    expr_df: pd.DataFrame,
    tf_list: List[str],
    pairs: List[Tuple[str, str]],
    seed: int,
) -> np.ndarray:
    """LASSO (Tibshirani 1996) or Elastic Net (Zou & Hastie 2005) per target.

    Represents the regularized regression family (the algorithmic ancestry of
    the Inferelator series), with absolute value of the source TF coefficient
    as the edge score.
    """
    col_idx = {g: i for i, g in enumerate(expr_df.columns)}
    tf_present = [tf for tf in tf_list if tf in col_idx]
    if not tf_present:
        return np.full(len(pairs), np.nan, dtype=float)
    tf_pos = {tf: i for i, tf in enumerate(tf_present)}
    X_tf = expr_df.loc[:, tf_present].to_numpy(dtype=np.float32)
    # Standardize features for regularized regression
    mu = X_tf.mean(axis=0)
    sd = X_tf.std(axis=0)
    sd[sd == 0] = 1.0
    X_tf = (X_tf - mu) / sd

    scores = np.full(len(pairs), np.nan, dtype=float)
    per_target: Dict[str, List[int]] = {}
    for i, (_, tgt) in enumerate(pairs):
        per_target.setdefault(tgt, []).append(i)

    for tgt, slots in per_target.items():
        if tgt not in col_idx:
            continue
        y = expr_df[tgt].to_numpy(dtype=np.float32)
        if y.std() == 0:
            continue
        if tgt in tf_pos:
            X_use = X_tf.copy()
            X_use[:, tf_pos[tgt]] = 0.0
        else:
            X_use = X_tf

        if method == "lasso":
            reg = LassoCV(cv=5, random_state=seed, n_jobs=1, max_iter=5000)
        elif method == "elasticnet":
            reg = ElasticNetCV(
                cv=5,
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
                random_state=seed,
                n_jobs=1,
                max_iter=5000,
            )
        else:
            raise ValueError(method)
        try:
            reg.fit(X_use, y)
            coefs = np.abs(reg.coef_)
        except Exception as exc:
            print(f"[warn] {method} fit failed for {tgt}: {exc}", file=sys.stderr)
            continue
        for slot in slots:
            src = pairs[slot][0]
            if src in tf_pos:
                scores[slot] = float(coefs[tf_pos[src]])
            else:
                scores[slot] = 0.0
    return scores


def _distance_correlation_scores(
    expr_df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
) -> np.ndarray:
    """Székely et al. (2007) distance correlation, nonlinear dependence."""
    if not HAVE_DCOR:
        return np.full(len(pairs), np.nan, dtype=float)
    col_idx = {g: i for i, g in enumerate(expr_df.columns)}
    X = expr_df.values
    scores = np.full(len(pairs), np.nan, dtype=float)
    for i, (src, tgt) in enumerate(pairs):
        ci = col_idx.get(src)
        cj = col_idx.get(tgt)
        if ci is None or cj is None:
            continue
        a = X[:, ci]
        b = X[:, cj]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        try:
            scores[i] = float(dcor.distance_correlation(a, b))
        except Exception:
            continue
    return scores


def _tree_scores(
    method: str,
    expr_df: pd.DataFrame,
    tf_list: List[str],
    pairs: List[Tuple[str, str]],
    seed: int,
    n_estimators: int = 200,
) -> np.ndarray:
    """Reference GENIE3 / GRNBoost2 implementation.

    For each target gene referenced by a pair, fit a regressor predicting
    that target from every TF in `tf_list`, then read off the feature
    importance of the source TF for each pair.
    """
    tf_present = [tf for tf in tf_list if tf in expr_df.columns]
    if not tf_present:
        return np.full(len(pairs), np.nan, dtype=float)
    tf_idx = {tf: i for i, tf in enumerate(tf_present)}
    X_tf = expr_df.loc[:, tf_present].to_numpy(dtype=np.float32)

    needed_targets = sorted({tgt for _, tgt in pairs if tgt in expr_df.columns})
    scores = np.full(len(pairs), np.nan, dtype=float)
    per_target: Dict[str, List[int]] = {}
    for i, (_src, tgt) in enumerate(pairs):
        per_target.setdefault(tgt, []).append(i)

    for tgt in needed_targets:
        y = expr_df[tgt].to_numpy(dtype=np.float32)
        if y.std() == 0:
            continue
        if tgt in tf_idx:
            X_use = X_tf.copy()
            X_use[:, tf_idx[tgt]] = 0.0
        else:
            X_use = X_tf

        if method == "genie3":
            reg = RandomForestRegressor(
                n_estimators=n_estimators,
                max_features="sqrt",
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=seed,
            )
        elif method == "grnboost2":
            reg = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=0.01,
                max_depth=3,
                subsample=0.9,
                max_features="sqrt",
                random_state=seed,
            )
        else:
            raise ValueError(method)

        try:
            reg.fit(X_use, y)
            importances = reg.feature_importances_
        except Exception as exc:
            print(f"[warn] {method} fit failed for {tgt}: {exc}", file=sys.stderr)
            continue

        for slot in per_target[tgt]:
            src = pairs[slot][0]
            if src in tf_idx:
                scores[slot] = float(importances[tf_idx[src]])
            else:
                scores[slot] = 0.0

    return scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _metric(scores: np.ndarray, labels: np.ndarray, seed: int, permutations: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(scores)
    scores = scores[mask]
    labels = labels[mask]
    n = int(len(labels))
    n_pos = int(labels.sum())
    if n == 0 or n_pos == 0 or n_pos == n:
        return {"n_pairs": n, "n_pos": n_pos, "aupr": float("nan"), "auroc": float("nan"), "perm_p": float("nan")}
    aupr = float(average_precision_score(labels, scores))
    try:
        auroc = float(roc_auc_score(labels, scores))
    except ValueError:
        auroc = float("nan")
    ge = 1
    for _ in range(permutations):
        perm = rng.permutation(labels)
        if average_precision_score(perm, scores) >= aupr:
            ge += 1
    return {
        "n_pairs": n,
        "n_pos": n_pos,
        "aupr": aupr,
        "auroc": auroc,
        "perm_p": ge / (permutations + 1),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_one_tissue(tissue: str, repo_root: Path, seed: int, permutations: int) -> pd.DataFrame:
    info = TISSUES[tissue]
    raw_path = repo_root / info["raw_h5ad"]
    vocab = _load_vocab(repo_root / "external/scGPT_checkpoints/whole-human/vocab.json")
    trrust_path = repo_root / "external/networks/trrust_human.tsv"

    print(f"[{tissue}] loading {raw_path}", flush=True)
    t0 = time.time()
    X_raw, gene_symbols = _read_raw_h5ad(raw_path)
    print(f"[{tissue}] raw shape {X_raw.shape}, load in {time.time() - t0:.1f}s", flush=True)

    print(f"[{tissue}] preprocessing (normalize, log1p, HVG={HVG_TOP}, vocab-restrict)", flush=True)
    X_proc, gene_symbols = _preprocess(X_raw, gene_symbols, vocab)
    print(f"[{tissue}] processed shape {X_proc.shape}", flush=True)

    # Sample a fixed subset of MAX_CELLS cells for all methods (matches max_cells=120)
    rng_cells = np.random.default_rng(seed)
    n_total = X_proc.shape[0]
    if n_total > MAX_CELLS:
        cell_idx = np.sort(rng_cells.choice(n_total, size=MAX_CELLS, replace=False))
    else:
        cell_idx = np.arange(n_total)
    X_cells = X_proc[cell_idx]

    # Sample pair list the same way run_causal_interventions.py does
    trrust = _load_trrust(trrust_path)
    gene_set = set(gene_symbols)
    pairs, labels = _sample_pairs(trrust, gene_set, MAX_PAIRS, RANDOM_CONTROL_PAIRS, seed)
    labels_arr = np.asarray(labels, dtype=int)
    print(f"[{tissue}] sampled {len(pairs)} pairs "
          f"({int(labels_arr.sum())} positive, {int((labels_arr == 0).sum())} negative)", flush=True)

    # Evidence filter: pair survives if both genes are in top-MAX_GENES of >=1 cell
    evidence = _evidence_filter(X_proc, gene_symbols, cell_idx, pairs, max_genes=MAX_GENES)
    kept_idx = [i for i, e in enumerate(evidence) if e]
    pairs_kept = [pairs[i] for i in kept_idx]
    labels_kept = labels_arr[kept_idx]
    print(f"[{tissue}] evidence-filtered to {len(pairs_kept)} pairs "
          f"({int(labels_kept.sum())} pos)", flush=True)

    # Build dense DataFrame of MAX_CELLS x G for sklearn-based baselines
    expr_dense = np.asarray(X_cells.toarray(), dtype=np.float32)
    expr_df = pd.DataFrame(expr_dense, columns=gene_symbols)
    const_mask = expr_df.std(axis=0) > 0
    expr_df = expr_df.loc[:, const_mask]

    # TF list = sources in TRRUST, restricted to genes present in expr_df
    tf_list = sorted({s for s in trrust["source"].tolist() if s in expr_df.columns})
    print(f"[{tissue}] expr matrix {expr_df.shape[0]} x {expr_df.shape[1]} "
          f"TFs available: {len(tf_list)}", flush=True)

    # --- run methods ---
    results: List[Dict[str, object]] = []

    def _record(method: str, raw_scores: np.ndarray) -> None:
        m = _metric(raw_scores, labels_kept, seed, permutations)
        m.update({"tissue": tissue, "method": method})
        results.append(m)
        print(f"[{tissue}] {method:<22s} AUPR={m['aupr']:.4f} "
              f"AUROC={m['auroc']:.4f} p={m['perm_p']:.4f} n={m['n_pairs']}",
              flush=True)

    print(f"[{tissue}] Pearson coexpression...", flush=True)
    _record("pearson_coexpr", _pearson_scores(expr_df, pairs_kept))

    print(f"[{tissue}] Spearman coexpression (rank)...", flush=True)
    _record("spearman_coexpr", _spearman_scores(expr_df, pairs_kept))

    if HAVE_DCOR:
        print(f"[{tissue}] Distance correlation (Szekely et al. 2007)...", flush=True)
        _record("distance_correlation", _distance_correlation_scores(expr_df, pairs_kept))

    print(f"[{tissue}] Partial correlation (GGM / PC family, ridge=1e-2)...", flush=True)
    t0 = time.time()
    _record("partial_correlation",
            _partial_correlation_scores(expr_df, tf_list, pairs_kept))
    print(f"[{tissue}] partial corr done in {time.time() - t0:.1f}s", flush=True)

    # Mutual information: compute MI cache for both sources AND targets used by
    # the pair set, so CLR and ARACNE get two-sided normalization.
    unique_sources = sorted({s for s, _ in pairs_kept})
    unique_targets = sorted({t for _, t in pairs_kept})
    mi_query_set = sorted(set(unique_sources) | set(unique_targets))
    print(f"[{tissue}] Mutual information over {len(mi_query_set)} query genes "
          f"(used by MI, CLR, ARACNE)...", flush=True)
    t0 = time.time()
    mi_cache = _build_mi_cache(expr_df, mi_query_set, seed)
    print(f"[{tissue}] MI cache done in {time.time() - t0:.1f}s", flush=True)
    _record("mutual_information", _mi_scores(expr_df, pairs_kept, mi_cache))
    _record("clr", _clr_scores(expr_df, pairs_kept, mi_cache))
    _record("aracne", _aracne_scores(expr_df, pairs_kept, mi_cache))

    print(f"[{tissue}] LASSO regression (LassoCV per target)...", flush=True)
    t0 = time.time()
    _record("lasso", _regularized_regression_scores("lasso", expr_df, tf_list, pairs_kept, seed))
    print(f"[{tissue}] lasso done in {time.time() - t0:.1f}s", flush=True)

    print(f"[{tissue}] Elastic Net (ElasticNetCV per target)...", flush=True)
    t0 = time.time()
    _record("elasticnet", _regularized_regression_scores("elasticnet", expr_df, tf_list, pairs_kept, seed))
    print(f"[{tissue}] elasticnet done in {time.time() - t0:.1f}s", flush=True)

    print(f"[{tissue}] GRNBoost2 (SGBM per target)...", flush=True)
    t0 = time.time()
    grnb = _tree_scores("grnboost2", expr_df, tf_list, pairs_kept, seed)
    print(f"[{tissue}] GRNBoost2 done in {time.time() - t0:.1f}s", flush=True)
    _record("grnboost2", grnb)

    print(f"[{tissue}] GENIE3 (random forest per target)...", flush=True)
    t0 = time.time()
    gen = _tree_scores("genie3", expr_df, tf_list, pairs_kept, seed)
    print(f"[{tissue}] GENIE3 done in {time.time() - t0:.1f}s", flush=True)
    _record("genie3", gen)

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", default="all", choices=["all", *TISSUES.keys()])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="outputs/grn_baseline_comparison")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tissues = list(TISSUES.keys()) if args.tissue == "all" else [args.tissue]
    combined: List[pd.DataFrame] = []
    for tissue in tissues:
        df = run_one_tissue(tissue, repo_root, args.seed, args.permutations)
        df.to_csv(out_dir / f"{tissue}_metrics.tsv", sep="\t", index=False)
        combined.append(df)

    summary = pd.concat(combined, ignore_index=True)
    summary = summary[["tissue", "method", "n_pairs", "n_pos", "aupr", "auroc", "perm_p"]]
    summary.to_csv(out_dir / "summary.tsv", sep="\t", index=False)
    print("\n=== summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
