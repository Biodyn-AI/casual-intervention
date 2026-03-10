from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch

from src.utils.torch_utils import move_to_device


def reduce_attentions(attentions, reduce_layers: bool = True, reduce_heads: bool = True):
    if attentions is None:
        raise ValueError("No attentions were provided by the model")

    if isinstance(attentions, torch.Tensor):
        attn_list = [attentions]
    else:
        attn_list = list(attentions)

    normalized = []
    for attn in attn_list:
        if attn.dim() == 2:
            attn = attn.unsqueeze(0).unsqueeze(0)
        elif attn.dim() == 3:
            attn = attn.unsqueeze(0)
        elif attn.dim() != 4:
            raise ValueError(f"Unsupported attention shape: {attn.shape}")
        normalized.append(attn)

    stacked = torch.stack(normalized, dim=0)  # layers, batch, heads, seq, seq
    attn = stacked
    if reduce_layers:
        attn = attn.mean(dim=0)  # batch, heads, seq, seq
    if reduce_heads:
        if reduce_layers:
            attn = attn.mean(dim=1)  # batch, seq, seq
        else:
            attn = attn.mean(dim=2)  # layers, batch, seq, seq
    return attn


def _make_array(
    path: str | Path | None,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=dtype)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _init_score_arrays(
    reduced: torch.Tensor,
    n_genes: int,
    reduce_layers: bool,
    reduce_heads: bool,
    score_dtype: np.dtype,
    count_dtype: np.dtype,
    score_sum_path: str | Path | None,
    score_count_path: str | Path | None,
    share_counts: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if reduce_layers and reduce_heads:
        score_shape = (n_genes, n_genes)
    elif reduce_layers and not reduce_heads:
        if reduced.dim() != 4:
            raise ValueError("Expected attention shape (batch, heads, seq, seq)")
        score_shape = (reduced.shape[1], n_genes, n_genes)
    elif not reduce_layers and reduce_heads:
        if reduced.dim() != 4:
            raise ValueError("Expected attention shape (layers, batch, seq, seq)")
        score_shape = (reduced.shape[0], n_genes, n_genes)
    else:
        if reduced.dim() != 5:
            raise ValueError("Expected attention shape (layers, batch, heads, seq, seq)")
        score_shape = (reduced.shape[0], reduced.shape[2], n_genes, n_genes)

    count_shape = (n_genes, n_genes) if share_counts or len(score_shape) == 2 else score_shape
    score_sum = _make_array(score_sum_path, score_shape, score_dtype)
    score_count = _make_array(score_count_path, count_shape, count_dtype)
    return score_sum, score_count


def _accumulate_attention_scores_aggregate(
    score_sum: np.ndarray,
    score_count: np.ndarray,
    attn: torch.Tensor,
    gene_indices: torch.Tensor,
    ignore_index: int = -1,
):
    attn_np = attn.detach().cpu().numpy()
    gene_indices_np = gene_indices.detach().cpu().numpy()

    batch_size = attn_np.shape[0]
    for i in range(batch_size):
        indices = gene_indices_np[i]
        valid = indices != ignore_index
        if not np.any(valid):
            continue
        kept = indices[valid]
        attn_cell = attn_np[i]
        attn_cell = attn_cell[np.ix_(valid, valid)]
        score_sum[np.ix_(kept, kept)] += attn_cell
        score_count[np.ix_(kept, kept)] += 1


def _accumulate_attention_scores_heads(
    score_sum: np.ndarray,
    score_count: np.ndarray,
    attn: torch.Tensor,
    gene_indices: torch.Tensor,
    ignore_index: int = -1,
) -> None:
    attn_np = attn.detach().cpu().numpy()
    gene_indices_np = gene_indices.detach().cpu().numpy()

    batch_size = attn_np.shape[0]
    for i in range(batch_size):
        indices = gene_indices_np[i]
        valid = indices != ignore_index
        if not np.any(valid):
            continue
        kept = indices[valid]
        attn_cell = attn_np[i][:, valid][:, :, valid]
        for head_idx, attn_head in enumerate(attn_cell):
            score_sum[head_idx][np.ix_(kept, kept)] += attn_head
        if score_count.ndim == 2:
            score_count[np.ix_(kept, kept)] += 1
        else:
            for head_idx in range(score_sum.shape[0]):
                score_count[head_idx][np.ix_(kept, kept)] += 1


def _accumulate_attention_scores_layers(
    score_sum: np.ndarray,
    score_count: np.ndarray,
    attn: torch.Tensor,
    gene_indices: torch.Tensor,
    ignore_index: int = -1,
) -> None:
    attn_np = attn.detach().cpu().numpy()
    gene_indices_np = gene_indices.detach().cpu().numpy()

    batch_size = attn_np.shape[1]
    for i in range(batch_size):
        indices = gene_indices_np[i]
        valid = indices != ignore_index
        if not np.any(valid):
            continue
        kept = indices[valid]
        attn_cell = attn_np[:, i][:, valid][:, :, valid]
        for layer_idx, attn_layer in enumerate(attn_cell):
            score_sum[layer_idx][np.ix_(kept, kept)] += attn_layer
        if score_count.ndim == 2:
            score_count[np.ix_(kept, kept)] += 1
        else:
            for layer_idx in range(score_sum.shape[0]):
                score_count[layer_idx][np.ix_(kept, kept)] += 1


def _accumulate_attention_scores_layers_heads(
    score_sum: np.ndarray,
    score_count: np.ndarray,
    attn: torch.Tensor,
    gene_indices: torch.Tensor,
    ignore_index: int = -1,
) -> None:
    attn_np = attn.detach().cpu().numpy()
    gene_indices_np = gene_indices.detach().cpu().numpy()

    batch_size = attn_np.shape[1]
    for i in range(batch_size):
        indices = gene_indices_np[i]
        valid = indices != ignore_index
        if not np.any(valid):
            continue
        kept = indices[valid]
        attn_cell = attn_np[:, i][:, :, valid][:, :, :, valid]
        for layer_idx, layer_attn in enumerate(attn_cell):
            for head_idx, attn_head in enumerate(layer_attn):
                score_sum[layer_idx, head_idx][np.ix_(kept, kept)] += attn_head
        if score_count.ndim == 2:
            score_count[np.ix_(kept, kept)] += 1
        else:
            for layer_idx in range(score_sum.shape[0]):
                for head_idx in range(score_sum.shape[1]):
                    score_count[layer_idx, head_idx][np.ix_(kept, kept)] += 1


def finalize_attention_scores(score_sum: np.ndarray, score_count: np.ndarray) -> np.ndarray:
    denom = np.maximum(score_count, 1)
    return score_sum / denom


def extract_attention_scores(
    model,
    dataloader: Iterable,
    n_genes: int,
    device: str,
    reduce_layers: bool = True,
    reduce_heads: bool = True,
    ignore_index: int = -1,
    score_sum_path: str | Path | None = None,
    score_count_path: str | Path | None = None,
    score_dtype: np.dtype | str = np.float32,
    count_dtype: np.dtype | str = np.int32,
    share_counts: bool | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    score_sum = None
    score_count = None
    score_dtype = np.dtype(score_dtype)
    count_dtype = np.dtype(count_dtype)
    if share_counts is None:
        share_counts = not (reduce_layers and reduce_heads)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            _, attentions = model.forward_with_attentions(batch)
            if attentions is None:
                raise RuntimeError("Model did not return attention weights")
            reduced = reduce_attentions(attentions, reduce_layers=reduce_layers, reduce_heads=reduce_heads)
            if score_sum is None or score_count is None:
                score_sum, score_count = _init_score_arrays(
                    reduced,
                    n_genes,
                    reduce_layers,
                    reduce_heads,
                    score_dtype,
                    count_dtype,
                    score_sum_path,
                    score_count_path,
                    share_counts,
                )
            if reduce_layers and reduce_heads:
                _accumulate_attention_scores_aggregate(
                    score_sum,
                    score_count,
                    reduced,
                    batch["gene_indices"],
                    ignore_index,
                )
            elif reduce_layers and not reduce_heads:
                _accumulate_attention_scores_heads(
                    score_sum,
                    score_count,
                    reduced,
                    batch["gene_indices"],
                    ignore_index,
                )
            elif not reduce_layers and reduce_heads:
                _accumulate_attention_scores_layers(
                    score_sum,
                    score_count,
                    reduced,
                    batch["gene_indices"],
                    ignore_index,
                )
            else:
                _accumulate_attention_scores_layers_heads(
                    score_sum,
                    score_count,
                    reduced,
                    batch["gene_indices"],
                    ignore_index,
                )

    if score_sum is None or score_count is None:
        raise ValueError("No attention batches were processed")
    return score_sum, score_count
