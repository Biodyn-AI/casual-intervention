from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scanpy as sc


@dataclass
class PreprocessConfig:
    min_genes: int = 200
    min_cells: int = 3
    normalize_total: float = 1e4
    log1p: bool = True
    hvg: int | None = 3000
    hvg_flavor: str = "seurat_v3"
    max_cells: int | None = None
    retain_genes: list[str] | None = None


def load_h5ad(path: str | Path):
    return sc.read_h5ad(str(path))


def preprocess_anndata(adata, config: PreprocessConfig):
    retain_set = {str(gene).upper() for gene in (config.retain_genes or []) if gene}
    if config.max_cells and adata.n_obs > config.max_cells:
        adata = adata[: config.max_cells].copy()
    if config.min_genes:
        sc.pp.filter_cells(adata, min_genes=config.min_genes)
    if config.min_cells:
        sc.pp.filter_genes(adata, min_cells=config.min_cells)
    if config.normalize_total:
        sc.pp.normalize_total(adata, target_sum=config.normalize_total)
    if config.log1p:
        sc.pp.log1p(adata)
    if config.hvg:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=config.hvg,
            subset=False,
            flavor=config.hvg_flavor,
        )
        hvg_mask = adata.var["highly_variable"].to_numpy()
        if retain_set:
            retain_mask = np.asarray(adata.var_names.str.upper().isin(retain_set))
            hvg_mask = hvg_mask | retain_mask
        adata = adata[:, hvg_mask].copy()
    elif retain_set:
        retain_mask = np.asarray(adata.var_names.str.upper().isin(retain_set))
        adata = adata[:, retain_mask].copy()
    return adata


def subset_to_vocab(adata, vocab_genes: Iterable[str]):
    vocab_list = list(vocab_genes)
    vocab_set = set(vocab_list)
    mask = adata.var_names.isin(vocab_set)
    filtered = adata[:, mask].copy()
    adata_genes = set(adata.var_names)
    missing_in_vocab = [gene for gene in vocab_list if gene not in adata_genes]
    return filtered, missing_in_vocab


def map_ensembl_to_symbol(
    adata,
    mapping_path: str | Path,
    id_col: str = "feature_id",
    name_col: str = "feature_name",
):
    mapping_path = Path(mapping_path)
    mapping_df = pd.read_csv(mapping_path, usecols=[id_col, name_col])
    mapping = dict(zip(mapping_df[id_col], mapping_df[name_col]))

    original = adata.var_names.to_list()
    mapped = [mapping.get(gene, gene) for gene in original]
    adata.var["ensembl_id"] = original
    adata.var_names = mapped
    missing = [gene for gene in original if gene not in mapping]
    return adata, missing
