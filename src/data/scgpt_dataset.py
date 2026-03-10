from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


@dataclass
class ScGPTDatasetConfig:
    max_genes: int = 2048
    include_zero: bool = False
    sort_by_expression: bool = True
    pad_token_id: int | None = None
    cls_token_id: int | None = None
    force_gene_names: list[str] | None = None


class ScGPTDataset(Dataset):
    def __init__(
        self,
        adata,
        gene_to_id: Dict[str, int],
        config: ScGPTDatasetConfig,
    ):
        self.adata = adata
        self.config = config
        self.gene_to_id = gene_to_id
        if self.config.pad_token_id is None:
            raise ValueError("pad_token_id must be set before constructing the dataset")
        self.gene_names = list(adata.var_names)
        missing = [name for name in self.gene_names if name not in gene_to_id]
        if missing:
            raise ValueError(f"Genes missing from vocab: {len(missing)}")
        self.gene_ids = np.array([gene_to_id[name] for name in self.gene_names], dtype=np.int64)
        self.gene_indices = np.arange(len(self.gene_names), dtype=np.int64)
        self.force_gene_indices = None
        if self.config.force_gene_names:
            name_to_idx = {name: idx for idx, name in enumerate(self.gene_names)}
            indices = [
                name_to_idx[name]
                for name in self.config.force_gene_names
                if name in name_to_idx
            ]
            if indices:
                self.force_gene_indices = np.array(sorted(set(indices)), dtype=np.int64)

    def __len__(self) -> int:
        return self.adata.n_obs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.adata.X[idx]
        if sp.issparse(row):
            values = row.toarray().ravel()
        else:
            values = np.asarray(row).ravel()

        if self.config.include_zero:
            mask = np.ones_like(values, dtype=bool)
        else:
            mask = values > 0
        if self.force_gene_indices is not None:
            mask[self.force_gene_indices] = True

        gene_ids = self.gene_ids[mask]
        gene_indices = self.gene_indices[mask]
        expr_values = values[mask]

        if gene_ids.size == 0:
            gene_ids = np.array([], dtype=np.int64)
            gene_indices = np.array([], dtype=np.int64)
            expr_values = np.array([], dtype=np.float32)

        if self.config.sort_by_expression and gene_ids.size > 1:
            order = np.argsort(-expr_values)
            gene_ids = gene_ids[order]
            gene_indices = gene_indices[order]
            expr_values = expr_values[order]

        max_genes = self.config.max_genes
        if self.config.cls_token_id is not None and max_genes > 0:
            max_genes = max_genes - 1
        if max_genes > 0 and self.force_gene_indices is not None and gene_ids.size > 0:
            forced_mask = np.isin(gene_indices, self.force_gene_indices)
            forced_ids = gene_ids[forced_mask]
            forced_indices = gene_indices[forced_mask]
            forced_values = expr_values[forced_mask]
            if forced_ids.size > max_genes:
                forced_ids = forced_ids[:max_genes]
                forced_indices = forced_indices[:max_genes]
                forced_values = forced_values[:max_genes]
                gene_ids = forced_ids
                gene_indices = forced_indices
                expr_values = forced_values
            else:
                remaining_mask = ~forced_mask
                remaining_ids = gene_ids[remaining_mask]
                remaining_indices = gene_indices[remaining_mask]
                remaining_values = expr_values[remaining_mask]
                remaining_count = max_genes - forced_ids.size
                if remaining_count < 0:
                    remaining_count = 0
                gene_ids = np.concatenate([forced_ids, remaining_ids[:remaining_count]])
                gene_indices = np.concatenate([forced_indices, remaining_indices[:remaining_count]])
                expr_values = np.concatenate([forced_values, remaining_values[:remaining_count]])
        elif gene_ids.size > max_genes > 0:
            gene_ids = gene_ids[:max_genes]
            gene_indices = gene_indices[:max_genes]
            expr_values = expr_values[:max_genes]

        if self.config.cls_token_id is not None:
            gene_ids = np.concatenate(([self.config.cls_token_id], gene_ids))
            gene_indices = np.concatenate(([-1], gene_indices))
            expr_values = np.concatenate(([0.0], expr_values))

        pad_len = self.config.max_genes - gene_ids.size
        if pad_len < 0:
            pad_len = 0

        if pad_len:
            gene_ids = np.pad(
                gene_ids,
                (0, pad_len),
                mode="constant",
                constant_values=self.config.pad_token_id,
            )
            gene_indices = np.pad(
                gene_indices,
                (0, pad_len),
                mode="constant",
                constant_values=-1,
            )
            expr_values = np.pad(expr_values, (0, pad_len), mode="constant", constant_values=0.0)

        src_key_padding_mask = gene_ids == self.config.pad_token_id

        return {
            "gene_ids": torch.tensor(gene_ids, dtype=torch.long),
            "gene_values": torch.tensor(expr_values, dtype=torch.float32),
            "gene_indices": torch.tensor(gene_indices, dtype=torch.long),
            "src_key_padding_mask": torch.tensor(src_key_padding_mask, dtype=torch.bool),
        }


def collate_scgpt(batch: Iterable[Dict[str, torch.Tensor]]):
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch]) for key in keys}
