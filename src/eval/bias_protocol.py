from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.eval.gene_symbols import SymbolMapper, normalize_symbol


@dataclass
class CandidateSet:
    name: str
    sources: List[str]
    targets: List[str]
    allow_self_edges: bool

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def target_count(self) -> int:
        return len(self.targets)

    @property
    def size(self) -> int:
        size = self.source_count * self.target_count
        if self.allow_self_edges:
            return size
        shared = len(set(self.sources) & set(self.targets))
        return size - shared


def read_gene_set(h5ad_path: str | Path) -> List[str]:
    path = Path(h5ad_path)
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            if "var/_index" in handle:
                data = handle["var/_index"][()]
                return [
                    value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
                    for value in data
                ]
    except Exception:
        pass

    import anndata as ad

    adata = ad.read_h5ad(str(path), backed="r")
    gene_names = list(adata.var_names)
    if getattr(adata, "file", None) is not None:
        adata.file.close()
    return gene_names


def basic_normalize_symbols(symbols: Iterable[str | None]) -> List[str]:
    return [normalize_symbol(symbol) for symbol in symbols]


def map_symbols_for_context(
    symbols: Iterable[str | None],
    mapper: SymbolMapper,
    context: str,
) -> Tuple[List[str], pd.DataFrame]:
    unique_symbols = list(dict.fromkeys(symbols))
    mapped, report = mapper.map_symbols(unique_symbols)
    report.insert(0, "context", context)
    mapping = dict(zip(unique_symbols, mapped))
    mapped_list = [mapping[symbol] for symbol in symbols]
    return mapped_list, report


def map_edges_for_context(
    edges: pd.DataFrame,
    mapper: SymbolMapper,
    context: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sources, source_report = map_symbols_for_context(edges["source"].tolist(), mapper, f"{context}:source")
    targets, target_report = map_symbols_for_context(edges["target"].tolist(), mapper, f"{context}:target")
    mapped = edges.copy()
    mapped["source"] = sources
    mapped["target"] = targets
    mapped = mapped[(mapped["source"] != "") & (mapped["target"] != "")]
    mapped = mapped.drop_duplicates(subset=["source", "target"]).reset_index(drop=True)
    report = pd.concat([source_report, target_report], ignore_index=True)
    return mapped, report


def load_edge_list(path: str | Path, source_col: str | None = None, target_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if source_col is None or target_col is None:
        columns = {col.lower(): col for col in df.columns}
        if "source" in columns and "target" in columns:
            source_col = columns["source"]
            target_col = columns["target"]
        elif "tf" in columns and "target" in columns:
            source_col = columns["tf"]
            target_col = columns["target"]
        else:
            df = pd.read_csv(path, sep="\t", header=None, dtype=str)
            if df.shape[1] < 2:
                raise ValueError(f"Edge list at {path} must have at least two columns")
            source_col = 0
            target_col = 1
    edges = df[[source_col, target_col]].dropna().copy()
    edges.columns = ["source", "target"]
    return edges.drop_duplicates().reset_index(drop=True)


def dedupe_edges_with_score(edges: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if score_col not in edges.columns:
        edges[score_col] = 1.0
    edges[score_col] = edges[score_col].astype(float)
    grouped = edges.groupby(["source", "target"], as_index=False)[score_col].max()
    return grouped


def build_candidate_sets(
    gene_set: Sequence[str],
    gold_standards: Dict[str, pd.DataFrame],
    candidate_specs: Sequence[Dict],
    allow_self_edges: bool,
) -> List[CandidateSet]:
    gene_set_clean = sorted({symbol for symbol in gene_set if symbol})
    gene_set_values = set(gene_set_clean)
    candidates: List[CandidateSet] = []
    for spec in candidate_specs:
        name = spec.get("name") or spec.get("type")
        spec_type = spec.get("type", "all_pairs")
        if spec_type == "all_pairs":
            sources = gene_set_clean
            targets = gene_set_clean
        elif spec_type == "tf_sources":
            source_names = spec.get("sources_from") or list(gold_standards.keys())
            source_set: set[str] = set()
            for gs_name in source_names:
                if gs_name in gold_standards:
                    source_set |= set(gold_standards[gs_name]["source"].unique())
            sources = sorted(source_set & gene_set_values)
            targets = gene_set_clean
        elif spec_type == "tf_sources_targets":
            source_names = spec.get("sources_from") or list(gold_standards.keys())
            target_names = spec.get("targets_from") or list(gold_standards.keys())
            source_set: set[str] = set()
            target_set: set[str] = set()
            for gs_name in source_names:
                if gs_name in gold_standards:
                    source_set |= set(gold_standards[gs_name]["source"].unique())
            for gs_name in target_names:
                if gs_name in gold_standards:
                    target_set |= set(gold_standards[gs_name]["target"].unique())
            sources = sorted(source_set & gene_set_values)
            targets = sorted(target_set & gene_set_values)
        else:
            raise ValueError(f"Unknown candidate set type: {spec_type}")
        candidates.append(CandidateSet(name=name, sources=sources, targets=targets, allow_self_edges=allow_self_edges))
    return candidates


def filter_edges_to_candidate(edges: pd.DataFrame, candidate: CandidateSet) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    filtered = edges[
        edges["source"].isin(candidate.sources) & edges["target"].isin(candidate.targets)
    ].copy()
    if not candidate.allow_self_edges:
        filtered = filtered[filtered["source"] != filtered["target"]]
    return filtered.reset_index(drop=True)


def _ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    total_pos: int,
    total_neg: int,
) -> Dict[str, float]:
    if total_pos <= 0 or total_neg < 0:
        return {"aupr": 0.0, "auroc": 0.0}
    if scores.size == 0:
        return {"aupr": 0.0, "auroc": 0.0}

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)

    precision = tp / (tp + fp)
    recall = tp / total_pos
    tpr = recall
    fpr = fp / total_neg if total_neg > 0 else np.zeros_like(fp, dtype=float)

    if tp[-1] < total_pos or (total_neg > 0 and fp[-1] < total_neg):
        base_rate = total_pos / (total_pos + total_neg) if total_pos + total_neg else 0.0
        precision = np.append(precision, base_rate)
        recall = np.append(recall, 1.0)
        tpr = np.append(tpr, 1.0)
        fpr = np.append(fpr, 1.0 if total_neg > 0 else 0.0)

    aupr = float(np.trapz(precision, recall)) if precision.size > 1 else float(precision[0])
    auroc = float(np.trapz(tpr, fpr)) if tpr.size > 1 else float(tpr[0])

    return {"aupr": aupr, "auroc": auroc}


def _top_k_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    total_pos: int,
    top_k: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if labels.size == 0:
        for k in top_k:
            metrics[f"precision@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
        return metrics
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    for k in top_k:
        k_eff = min(k, sorted_labels.size)
        if k_eff == 0:
            metrics[f"precision@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
            continue
        top_labels = sorted_labels[:k_eff]
        tp = int(top_labels.sum())
        precision = tp / k_eff
        recall = tp / total_pos if total_pos else 0.0
        metrics[f"precision@{k}"] = precision
        metrics[f"recall@{k}"] = recall
    return metrics


def _prepare_candidate_edges(
    pred_edges: pd.DataFrame,
    true_edges: pd.DataFrame,
    candidate: CandidateSet,
) -> Tuple[pd.DataFrame, set[Tuple[str, str]]]:
    pred_filtered = filter_edges_to_candidate(pred_edges, candidate)
    true_filtered = filter_edges_to_candidate(true_edges, candidate)
    true_set = set(zip(true_filtered["source"], true_filtered["target"]))
    return pred_filtered, true_set


def prepare_candidate_edges(
    pred_edges: pd.DataFrame,
    true_edges: pd.DataFrame,
    candidate: CandidateSet,
) -> Tuple[pd.DataFrame, set[Tuple[str, str]]]:
    return _prepare_candidate_edges(pred_edges, true_edges, candidate)


def evaluate_from_sets(
    pred_filtered: pd.DataFrame,
    true_set: set[Tuple[str, str]],
    candidate: CandidateSet,
    score_col: str,
    top_k: Sequence[int],
) -> Dict[str, float]:
    if pred_filtered.empty:
        scores = np.array([], dtype=float)
        labels = np.array([], dtype=int)
    else:
        scores = pred_filtered[score_col].to_numpy(dtype=float)
        labels = np.array(
            [1 if (src, tgt) in true_set else 0 for src, tgt in zip(pred_filtered["source"], pred_filtered["target"])],
            dtype=int,
        )

    total_pos = len(true_set)
    total_neg = max(candidate.size - total_pos, 0)

    tp = int(labels.sum()) if labels.size else 0
    fp = int(labels.size - tp)
    fn = int(total_pos - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    metrics: Dict[str, float] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    metrics.update(_ranking_metrics(scores, labels, total_pos, total_neg))
    metrics.update(_top_k_metrics(labels, scores, total_pos, top_k))
    metrics["predicted_edges"] = int(len(pred_filtered))
    metrics["true_edges"] = int(total_pos)
    metrics["candidate_size"] = int(candidate.size)
    metrics["base_rate"] = total_pos / candidate.size if candidate.size else 0.0

    return metrics


def bootstrap_auc_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    total_pos: int,
    total_neg: int,
    n_resamples: int,
    seed: int,
) -> Dict[str, float]:
    if n_resamples <= 0 or scores.size == 0:
        return {}
    rng = np.random.default_rng(seed)
    n = scores.size
    aupr_values = []
    auroc_values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample_scores = scores[idx]
        sample_labels = labels[idx]
        metrics = _ranking_metrics(sample_scores, sample_labels, total_pos, total_neg)
        aupr_values.append(metrics["aupr"])
        auroc_values.append(metrics["auroc"])
    aupr_arr = np.array(aupr_values, dtype=float)
    auroc_arr = np.array(auroc_values, dtype=float)
    return {
        "aupr_ci_lower": float(np.quantile(aupr_arr, 0.025)),
        "aupr_ci_upper": float(np.quantile(aupr_arr, 0.975)),
        "auroc_ci_lower": float(np.quantile(auroc_arr, 0.025)),
        "auroc_ci_upper": float(np.quantile(auroc_arr, 0.975)),
    }


def evaluate_predictions(
    pred_edges: pd.DataFrame,
    true_edges: pd.DataFrame,
    candidate: CandidateSet,
    score_col: str,
    top_k: Sequence[int],
) -> Dict[str, float]:
    pred_filtered, true_set = _prepare_candidate_edges(pred_edges, true_edges, candidate)
    return evaluate_from_sets(pred_filtered, true_set, candidate, score_col, top_k)


def _sample_negative_edges(
    sources: np.ndarray,
    targets: np.ndarray,
    existing: set[Tuple[str, str]],
    count: int,
    allow_self_edges: bool,
    rng: np.random.Generator,
) -> set[Tuple[str, str]]:
    sampled: set[Tuple[str, str]] = set()
    if count <= 0:
        return sampled
    attempts = 0
    max_attempts = 10
    source_len = len(sources)
    target_len = len(targets)
    if source_len == 0 or target_len == 0:
        return sampled

    batch = max(count * 5, 1000)
    while len(sampled) < count and attempts < max_attempts:
        size = min(batch, max((count - len(sampled)) * 10, 1000))
        src_idx = rng.integers(0, source_len, size=size)
        tgt_idx = rng.integers(0, target_len, size=size)
        for src, tgt in zip(sources[src_idx], targets[tgt_idx]):
            if not allow_self_edges and src == tgt:
                continue
            edge = (src, tgt)
            if edge in existing or edge in sampled:
                continue
            sampled.add(edge)
            if len(sampled) >= count:
                break
        attempts += 1
    return sampled


def simulate_noise(
    pred_edges: pd.DataFrame,
    true_edges: pd.DataFrame,
    candidate: CandidateSet,
    score_col: str,
    top_k: Sequence[int],
    rng: np.random.Generator,
    rates: Sequence[float],
    repeats: int,
    tf_dropout_rates: Sequence[float],
    target_dropout_rates: Sequence[float],
    structured_repeats: int = 1,
) -> List[Dict[str, float]]:
    pred_filtered, true_set = _prepare_candidate_edges(pred_edges, true_edges, candidate)
    sources = np.array(candidate.sources, dtype=object)
    targets = np.array(candidate.targets, dtype=object)

    results: List[Dict[str, float]] = []

    for rate in rates:
        for _ in range(repeats):
            rate = float(rate)
            edges = list(true_set)
            remove_count = int(round(rate * len(edges)))
            if remove_count > 0:
                remove_idx = rng.choice(len(edges), size=remove_count, replace=False)
                removed = {edges[idx] for idx in remove_idx}
                remaining = true_set - removed
            else:
                remaining = set(true_set)
            add_count = int(round(rate * len(edges)))
            additions = _sample_negative_edges(sources, targets, remaining, add_count, candidate.allow_self_edges, rng)
            perturbed = remaining | additions
            metrics = evaluate_from_sets(pred_filtered, perturbed, candidate, score_col, top_k)
            metrics.update({"noise_type": "random", "noise_rate": rate})
            results.append(metrics)

    if tf_dropout_rates:
        tf_sources = sorted({src for src, _ in true_set})
        for rate in tf_dropout_rates:
            if rate <= 0 or not tf_sources:
                continue
            for _ in range(max(structured_repeats, 1)):
                drop_count = max(1, int(round(rate * len(tf_sources))))
                drop_sources = set(rng.choice(tf_sources, size=drop_count, replace=False))
                perturbed = {edge for edge in true_set if edge[0] not in drop_sources}
                metrics = evaluate_from_sets(pred_filtered, perturbed, candidate, score_col, top_k)
                metrics.update({"noise_type": "tf_dropout", "noise_rate": float(rate)})
                results.append(metrics)

    if target_dropout_rates:
        tf_targets = sorted({tgt for _, tgt in true_set})
        for rate in target_dropout_rates:
            if rate <= 0 or not tf_targets:
                continue
            for _ in range(max(structured_repeats, 1)):
                drop_count = max(1, int(round(rate * len(tf_targets))))
                drop_targets = set(rng.choice(tf_targets, size=drop_count, replace=False))
                perturbed = {edge for edge in true_set if edge[1] not in drop_targets}
                metrics = evaluate_from_sets(pred_filtered, perturbed, candidate, score_col, top_k)
                metrics.update({"noise_type": "target_dropout", "noise_rate": float(rate)})
                results.append(metrics)

    return results


def summarize_noise_results(records: List[Dict[str, float]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    metric_cols = [col for col in df.columns if col not in {"noise_type", "noise_rate"}]
    group_cols = ["noise_type", "noise_rate"]
    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary
