from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from src.data.scgpt_dataset import ScGPTDataset, ScGPTDatasetConfig, collate_scgpt
from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import canonical_symbol, load_hgnc_alias_map, normalize_edges, normalize_gene_names
from src.eval.metrics import aupr
from src.interpret.causal_intervention import (
    OnlineStats,
    align_output_to_batch_seq,
    apply_pad_ablation,
    apply_value_ablation,
    attention_head_info,
    attention_head_slice,
    capture_module_outputs,
    extract_output_tensor,
    find_attention_modules,
    find_gene_positions,
    find_mlp_modules,
    find_transformer_layers,
    patch_module_output,
    reduce_output,
    swap_gene_values,
)
from src.model.scgpt_loader import load_scgpt_model
from src.model.vocab import load_vocab
from src.model.wrapper import ScGPTWrapper
from src.utils.config import load_config
from src.utils.torch_utils import move_to_device


def _device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_scgpt_args(args_path: Path) -> dict:
    with args_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_scgpt_model_args(scgpt_args: dict, vocab) -> dict:
    vocab_map = vocab.gene_to_id
    pad_token = scgpt_args.get("pad_token") or vocab.pad_token
    if not pad_token or pad_token not in vocab_map:
        for token in ("<pad>", "[PAD]", "<PAD>", "PAD"):
            if token in vocab_map:
                pad_token = token
                break
    if not pad_token or pad_token not in vocab_map:
        raise ValueError("Pad token not found in vocab; cannot build model args.")

    return {
        "ntoken": len(vocab_map),
        "d_model": scgpt_args["embsize"],
        "nhead": scgpt_args["nheads"],
        "d_hid": scgpt_args["d_hid"],
        "nlayers": scgpt_args["nlayers"],
        "nlayers_cls": scgpt_args.get("n_layers_cls", 3),
        "n_cls": 1,
        "vocab": vocab_map,
        "dropout": scgpt_args.get("dropout", 0.5),
        "pad_token": pad_token,
        "pad_value": scgpt_args.get("pad_value", 0),
        "do_mvc": bool(scgpt_args.get("MVC", False)),
        "do_dab": False,
        "use_batch_labels": False,
        "domain_spec_batchnorm": False,
        "input_emb_style": scgpt_args.get("input_emb_style", "continuous"),
        "n_input_bins": scgpt_args.get("n_bins"),
        "cell_emb_style": "avg-pool" if scgpt_args.get("no_cls") else "cls",
        "explicit_zero_prob": False,
        "use_fast_transformer": bool(scgpt_args.get("fast_transformer", False)),
        "fast_transformer_backend": "flash",
        "pre_norm": False,
    }


def _load_gene_list(path: str | None, alias_map: Dict[str, str]) -> List[str]:
    if not path:
        return []
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    genes = []
    for line in path_obj.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        gene = canonical_symbol(line.split()[0], alias_map)
        if gene:
            genes.append(gene)
    return genes


def _filter_edges(
    edges: pd.DataFrame,
    allowed_sources: Iterable[str],
    allowed_targets: Iterable[str],
) -> pd.DataFrame:
    if allowed_sources:
        edges = edges[edges["source"].isin(set(allowed_sources))]
    if allowed_targets:
        edges = edges[edges["target"].isin(set(allowed_targets))]
    return edges


def _sample_pairs(
    rng: random.Random,
    pairs: List[Tuple[str, str]],
    max_pairs: int | None,
) -> List[Tuple[str, str]]:
    if max_pairs is None or max_pairs <= 0 or len(pairs) <= max_pairs:
        return pairs
    return rng.sample(pairs, max_pairs)


def _build_pair_list(
    cfg: dict,
    paths: dict,
    gene_to_idx: Dict[str, int],
    alias_map: Dict[str, str],
    rng: random.Random,
) -> List[Dict[str, object]]:
    ci_cfg = cfg.get("causal_intervention", {})
    pairs_path = ci_cfg.get("pairs_path")
    confidence = ci_cfg.get("dorothea_confidence") or cfg.get("evaluation", {}).get(
        "dorothea_confidence"
    )

    edges = None
    if pairs_path:
        pairs_df = pd.read_csv(pairs_path, sep=None, engine="python")
        if {"source", "target"}.issubset(pairs_df.columns):
            edges = pairs_df[["source", "target"]].copy()
        else:
            edges = pairs_df.iloc[:, :2].copy()
            edges.columns = ["source", "target"]
    else:
        edges = load_dorothea(paths["dorothea_tsv"], confidence_levels=confidence)

    edges = normalize_edges(edges, alias_map)
    candidate_sources = _load_gene_list(ci_cfg.get("candidate_sources_path"), alias_map)
    candidate_targets = _load_gene_list(ci_cfg.get("candidate_targets_path"), alias_map)
    edges = _filter_edges(edges, candidate_sources, candidate_targets)
    edges = edges[
        edges["source"].isin(gene_to_idx.keys()) & edges["target"].isin(gene_to_idx.keys())
    ].drop_duplicates()

    true_pairs = [(row["source"], row["target"]) for _, row in edges.iterrows()]
    max_pairs = ci_cfg.get("max_pairs")
    if max_pairs is not None:
        max_pairs = int(max_pairs)
    true_pairs = _sample_pairs(rng, true_pairs, max_pairs)
    true_set = set(true_pairs)

    sources = sorted({pair[0] for pair in true_pairs}) or sorted(gene_to_idx.keys())
    targets = sorted({pair[1] for pair in true_pairs}) or sorted(gene_to_idx.keys())

    random_pairs: List[Tuple[str, str]] = []
    n_random = int(ci_cfg.get("random_control_pairs", 0) or 0)
    if n_random > 0:
        candidate_random_pairs = [
            (source, target)
            for source in sources
            for target in targets
            if source != target and (source, target) not in true_set
        ]
        if len(candidate_random_pairs) < n_random:
            print(
                "[causal-warning] random_control_pairs exceeds feasible non-true pairs; "
                f"requested={n_random}, feasible={len(candidate_random_pairs)}. "
                "Using all feasible pairs.",
                flush=True,
            )
            n_random = len(candidate_random_pairs)
        if n_random > 0:
            if n_random < len(candidate_random_pairs):
                random_pairs = rng.sample(candidate_random_pairs, n_random)
            else:
                random_pairs = candidate_random_pairs

    pairs = []
    for source, target in true_pairs:
        pairs.append(
            {
                "source": source,
                "target": target,
                "label": 1,
                "source_idx": gene_to_idx[source],
                "target_idx": gene_to_idx[target],
            }
        )
    for source, target in random_pairs:
        pairs.append(
            {
                "source": source,
                "target": target,
                "label": 0,
                "source_idx": gene_to_idx[source],
                "target_idx": gene_to_idx[target],
            }
        )
    return pairs


def _readout_value(
    wrapper: ScGPTWrapper,
    sample: Dict[str, torch.Tensor],
    target_pos: int,
    output_key: str | None,
    output_reduce: str | None,
) -> float:
    outputs = wrapper.forward(sample)
    return _readout_from_outputs(outputs, sample, target_pos, output_key, output_reduce)


def _readout_from_outputs(
    outputs,
    sample: Dict[str, torch.Tensor],
    target_pos: int,
    output_key: str | None,
    output_reduce: str | None,
) -> float:
    output_tensor = extract_output_tensor(outputs, output_key)
    output_tensor = reduce_output(output_tensor, output_reduce)
    output_tensor = align_output_to_batch_seq(
        output_tensor, batch_size=sample["gene_ids"].shape[0], seq_len=sample["gene_ids"].shape[1]
    )
    return float(output_tensor[0, target_pos].detach().cpu().item())


def _capture_clean_outputs(
    wrapper: ScGPTWrapper,
    sample: Dict[str, torch.Tensor],
    modules: List[torch.nn.Module],
) -> List[torch.Tensor | None]:
    clean_outputs: List[torch.Tensor | None] = [None for _ in modules]
    hooks = capture_module_outputs(modules, clean_outputs)
    try:
        wrapper.forward(sample)
    finally:
        for hook in hooks:
            hook.remove()
    return clean_outputs


def _forward_with_patch(
    wrapper: ScGPTWrapper,
    sample: Dict[str, torch.Tensor],
    module: torch.nn.Module,
    clean_output: torch.Tensor,
    positions: Iterable[int],
    batch_size: int,
    seq_len: int,
    head_slice: Tuple[int, int] | None = None,
):
    def _patch_hook(_module, _inputs, layer_output):
        return patch_module_output(
            layer_output,
            clean_output,
            positions,
            batch_size=batch_size,
            seq_len=seq_len,
            head_slice=head_slice,
        )

    handle = module.register_forward_hook(_patch_hook)
    try:
        return wrapper.forward(sample)
    finally:
        handle.remove()


def _pair_key(source: str, target: str) -> str:
    return f"{source}||{target}"


def _split_pair_key(key: str) -> Tuple[str, str]:
    return tuple(key.split("||", 1))  # type: ignore[return-value]


def _component_key(component: str, layer_idx: int, head_idx: int | None = None) -> str:
    if component == "head" and head_idx is not None:
        return f"{component}:{layer_idx}:{head_idx}"
    return f"{component}:{layer_idx}"


def _split_component_key(key: str) -> Tuple[str, int, int | None]:
    parts = key.split(":")
    component = parts[0]
    layer_idx = int(parts[1]) if len(parts) > 1 else -1
    head_idx = int(parts[2]) if len(parts) > 2 else None
    return component, layer_idx, head_idx


def _mermaid_id(label: str, suffix: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", label)
    return f"n_{safe}_{suffix}"


def _component_label(component: str, layer_idx: int, head_idx: int | None) -> str:
    if component == "head" and head_idx is not None:
        return f"L{layer_idx}-H{head_idx}"
    if component == "mlp":
        return f"L{layer_idx}-MLP"
    return f"L{layer_idx}"


def _stats_to_dict(stats: OnlineStats) -> dict:
    return {"count": stats.count, "mean": stats.mean, "m2": stats.m2}


def _stats_from_dict(data: dict) -> OnlineStats:
    return OnlineStats(count=int(data.get("count", 0)), mean=float(data.get("mean", 0.0)), m2=float(data.get("m2", 0.0)))


def _load_checkpoint(path: Path) -> tuple[set[Tuple[str, str]], dict, dict]:
    if not path.exists():
        return set(), {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    processed_pairs = set()
    for key in payload.get("processed_pairs", []):
        if isinstance(key, str) and "||" in key:
            processed_pairs.add(_split_pair_key(key))
    component_stats = payload.get("component_stats")
    if component_stats is None:
        layer_stats = payload.get("layer_stats", {})
        component_stats = {}
        for pair_key, layer_map in layer_stats.items():
            converted = {f"layer:{layer_idx}": stats for layer_idx, stats in layer_map.items()}
            component_stats[pair_key] = converted
    return processed_pairs, payload.get("pair_stats", {}), component_stats


def _save_checkpoint(
    path: Path,
    processed_pairs: set[Tuple[str, str]],
    pair_stats: Dict[Tuple[str, str], Dict[str, OnlineStats]],
    component_stats: Dict[Tuple[str, str], Dict[str, OnlineStats]],
) -> None:
    processed_keys = {_pair_key(*pair) for pair in processed_pairs}
    pair_payload = {}
    for pair_key, stats_map in pair_stats.items():
        key_str = _pair_key(*pair_key)
        if key_str not in processed_keys:
            continue
        serialized = {name: _stats_to_dict(stats) for name, stats in stats_map.items() if stats.count > 0}
        if serialized:
            pair_payload[key_str] = serialized

    component_payload = {}
    for pair_key, component_map in component_stats.items():
        key_str = _pair_key(*pair_key)
        if key_str not in processed_keys:
            continue
        serialized_components = {
            component_key: _stats_to_dict(stats)
            for component_key, stats in component_map.items()
            if stats.count > 0
        }
        if serialized_components:
            component_payload[key_str] = serialized_components

    payload = {
        "processed_pairs": sorted(processed_keys),
        "pair_stats": pair_payload,
        "component_stats": component_payload,
        "saved_at": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_causal_interventions(config_path: str, device: str | None) -> None:
    cfg = load_config(config_path)
    config_base = Path(config_path).resolve().parent
    project_seed = cfg.get("project", {}).get("seed")
    if project_seed is not None:
        np.random.seed(project_seed)
        torch.manual_seed(project_seed)

    paths = cfg["paths"]
    ci_cfg = cfg.get("causal_intervention", {})
    debug_log = bool(ci_cfg.get("debug_log", False))
    debug_stop = ci_cfg.get("debug_stop_stage")
    progress_every_pairs = int(ci_cfg.get("progress_every_pairs", 0) or 0)
    checkpoint_every_pairs = int(ci_cfg.get("checkpoint_every_pairs", 0) or 0)
    max_cells_per_pair = ci_cfg.get("max_cells_per_pair")
    if max_cells_per_pair is not None:
        max_cells_per_pair = int(max_cells_per_pair)

    def _log(message: str) -> None:
        if debug_log:
            print(f"[causal-debug] {message}", flush=True)

    torch_num_threads = ci_cfg.get("torch_num_threads")
    if torch_num_threads is not None:
        torch.set_num_threads(int(torch_num_threads))
    torch_num_interop = ci_cfg.get("torch_num_interop_threads")
    if torch_num_interop is not None:
        torch.set_num_interop_threads(int(torch_num_interop))
    output_dir = Path(ci_cfg.get("output_dir") or paths.get("causal_output_dir", "outputs/causal"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(ci_cfg.get("checkpoint_path") or (output_dir / "checkpoint.json"))
    resume = bool(ci_cfg.get("resume", True))

    _log("load processed h5ad")
    adata = sc.read_h5ad(paths["processed_h5ad"])
    max_cells = ci_cfg.get("max_cells")
    if max_cells:
        max_cells = int(max_cells)
        if adata.n_obs > max_cells:
            sample_mode = ci_cfg.get("cell_sample_mode", "sequential")
            if sample_mode == "random":
                sample_seed = ci_cfg.get("cell_sample_seed", project_seed or 0)
                rng = np.random.default_rng(int(sample_seed))
                indices = rng.choice(adata.n_obs, size=max_cells, replace=False)
                adata = adata[indices].copy()
            else:
                adata = adata[:max_cells].copy()
    if debug_stop == "after_load_data":
        _log("stop after load data")
        return

    _log("load vocab + dataset")
    vocab = load_vocab(paths["scgpt_vocab"])
    if hasattr(adata, "var_names"):
        valid_mask = [name in vocab.gene_to_id for name in adata.var_names]
        if not all(valid_mask):
            adata = adata[:, valid_mask].copy()
    dataset_cfg_dict = dict(cfg["scgpt_dataset"])
    force_genes_path = dataset_cfg_dict.pop("force_genes_path", None)
    if force_genes_path:
        force_path = Path(force_genes_path)
        if not force_path.is_absolute():
            force_path = config_base / force_path
        force_names = []
        for line in force_path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name and not name.startswith("#"):
                force_names.append(name)
        if force_names:
            dataset_cfg_dict["force_gene_names"] = force_names
    dataset_cfg = ScGPTDatasetConfig(**dataset_cfg_dict)
    if dataset_cfg.pad_token_id is None:
        if vocab.pad_id is None:
            raise ValueError("pad_token_id is missing and no pad token found in vocab")
        dataset_cfg.pad_token_id = vocab.pad_id

    batch_size = int(ci_cfg.get("batch_size", 1))
    tracing_cfg = ci_cfg.get("tracing", {})
    tracing_enabled = bool(tracing_cfg.get("enabled", False))
    if tracing_enabled and batch_size != 1:
        raise ValueError("Tracing requires batch_size=1 for deterministic patching")

    dataset = ScGPTDataset(adata, vocab.gene_to_id, dataset_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_scgpt,
    )
    if debug_stop == "after_dataloader":
        _log("stop after dataloader")
        return

    model_cfg = cfg.get("model", {})
    model_args = model_cfg.get("model_args") or {}
    if not model_args:
        args_path = model_cfg.get("args_path")
        if args_path:
            args_path = Path(args_path)
        else:
            args_path = Path(paths["scgpt_checkpoint"]).with_name("args.json")
        if not args_path.exists():
            raise FileNotFoundError(f"scGPT args file not found: {args_path}")
        scgpt_args = _load_scgpt_args(args_path)
        model_args = _build_scgpt_model_args(scgpt_args, vocab)
    if ci_cfg.get("disable_fast_transformer", False):
        model_args["use_fast_transformer"] = False

    device = _device(device)
    _log(f"load model on device={device}")
    model, missing, unexpected = load_scgpt_model(
        entrypoint=model_cfg["entrypoint"],
        repo_path=paths["scgpt_repo"],
        checkpoint_path=paths["scgpt_checkpoint"],
        device=device,
        model_args=model_args,
        prefix_to_strip=model_cfg.get("prefix_to_strip"),
    )
    if missing or unexpected:
        print(f"Model load missing keys: {len(missing)} unexpected: {len(unexpected)}")
    model.to(device)
    for module in model.modules():
        if hasattr(module, "enable_nested_tensor"):
            module.enable_nested_tensor = False
        if hasattr(module, "use_nested_tensor"):
            module.use_nested_tensor = False
    wrapper = ScGPTWrapper(model, model_cfg["forward_key_map"])
    if debug_stop == "after_model":
        _log("stop after model load")
        return

    _log("build pair list")
    alias_map = load_hgnc_alias_map(paths.get("hgnc_alias_tsv"))
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    gene_to_idx: Dict[str, int] = {}
    for idx, name in enumerate(gene_names_norm):
        if name and name not in gene_to_idx:
            gene_to_idx[name] = idx

    rng = random.Random(ci_cfg.get("random_seed", project_seed or 0))
    pairs = _build_pair_list(cfg, paths, gene_to_idx, alias_map, rng)
    if not pairs:
        raise ValueError("No candidate pairs found for causal interventions")
    _log(f"pairs={len(pairs)}")
    if debug_stop == "after_pairs":
        _log("stop after pair list")
        return

    interventions = ci_cfg.get("interventions", ["ablation", "swap"])
    ablation_mode = ci_cfg.get("ablation_mode", "value")
    baseline_value = float(ci_cfg.get("baseline_value", 0.0))
    swap_strategy = ci_cfg.get("swap_strategy", "random")
    output_key = ci_cfg.get("output_key")
    output_reduce = ci_cfg.get("output_reduce")

    layers = find_transformer_layers(model) if tracing_enabled else []
    trace_pairs_set: set[tuple[str, str]] | None = None
    trace_granularity = tracing_cfg.get("granularity", "layer")
    if trace_granularity == "head":
        trace_granularity = "attention_head"
    trace_modules: List[torch.nn.Module] = []
    trace_components: List[Dict[str, object]] = []
    if tracing_enabled:
        _log(f"tracing layers={len(layers)}")
        trace_pairs_path = tracing_cfg.get("pairs_path")
        if trace_pairs_path:
            trace_pairs_df = pd.read_csv(trace_pairs_path, sep=None, engine="python")
            if {"source", "target"}.issubset(trace_pairs_df.columns):
                trace_pairs_df = trace_pairs_df[["source", "target"]]
            else:
                trace_pairs_df = trace_pairs_df.iloc[:, :2]
                trace_pairs_df.columns = ["source", "target"]
            trace_pairs_df = normalize_edges(trace_pairs_df, alias_map)
            trace_pairs_df = trace_pairs_df[
                trace_pairs_df["source"].isin(gene_to_idx.keys())
                & trace_pairs_df["target"].isin(gene_to_idx.keys())
            ].drop_duplicates()
            trace_pairs_set = {
                (row["source"], row["target"]) for _, row in trace_pairs_df.iterrows()
            }
        else:
            max_trace_pairs = tracing_cfg.get("max_pairs")
            if max_trace_pairs:
                max_trace_pairs = int(max_trace_pairs)
                trace_pairs_set = set()
                for pair in pairs:
                    if len(trace_pairs_set) >= max_trace_pairs:
                        break
                    trace_pairs_set.add((pair["source"], pair["target"]))
        if trace_pairs_set is not None:
            _log(f"tracing subset size={len(trace_pairs_set)}")
        if not layers:
            raise ValueError("Tracing enabled but no transformer layers were detected")
        if trace_granularity in ("attention_head", "mlp"):
            if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
                torch.backends.mha.set_fastpath_enabled(False)
        if trace_granularity == "layer":
            trace_modules = layers
            for idx in range(len(layers)):
                trace_components.append(
                    {
                        "key": _component_key("layer", idx),
                        "component": "layer",
                        "layer_idx": idx,
                        "head_idx": None,
                        "module_idx": idx,
                        "head_slice": None,
                    }
                )
        elif trace_granularity == "mlp":
            mlp_modules = find_mlp_modules(layers)
            trace_modules = mlp_modules
            for idx in range(len(mlp_modules)):
                trace_components.append(
                    {
                        "key": _component_key("mlp", idx),
                        "component": "mlp",
                        "layer_idx": idx,
                        "head_idx": None,
                        "module_idx": idx,
                        "head_slice": None,
                    }
                )
        elif trace_granularity == "attention_head":
            attn_modules = find_attention_modules(layers)
            trace_modules = attn_modules
            head_indices = tracing_cfg.get("head_indices")
            max_heads_per_layer = tracing_cfg.get("max_heads_per_layer")
            if head_indices is not None:
                if isinstance(head_indices, (int, str)):
                    head_indices = [int(head_indices)]
                else:
                    head_indices = [int(head) for head in head_indices]
            for layer_idx, module in enumerate(attn_modules):
                num_heads, _ = attention_head_info(module)
                if head_indices:
                    selected_heads = [head for head in head_indices if 0 <= head < num_heads]
                elif max_heads_per_layer:
                    selected_heads = list(range(min(num_heads, int(max_heads_per_layer))))
                else:
                    selected_heads = list(range(num_heads))
                for head_idx in selected_heads:
                    head_slice = attention_head_slice(module, head_idx)
                    trace_components.append(
                        {
                            "key": _component_key("head", layer_idx, head_idx),
                            "component": "head",
                            "layer_idx": layer_idx,
                            "head_idx": head_idx,
                            "module_idx": layer_idx,
                            "head_slice": head_slice,
                        }
                    )
        else:
            raise ValueError(f"Unknown tracing granularity: {trace_granularity}")
        if not trace_components:
            raise ValueError("Tracing enabled but no trace components were configured")
        _log(f"tracing granularity={trace_granularity} components={len(trace_components)}")

    processed_pairs, pair_stats_ckpt, component_stats_ckpt = _load_checkpoint(checkpoint_path) if resume else (set(), {}, {})
    if processed_pairs:
        _log(f"resume: processed pairs={len(processed_pairs)}")

    pair_stats: Dict[Tuple[str, str], Dict[str, OnlineStats]] = {}
    component_stats: Dict[Tuple[str, str], Dict[str, OnlineStats]] = {}
    for pair in pairs:
        key = (pair["source"], pair["target"])
        pair_key_str = _pair_key(*key)
        pair_stats[key] = {}
        ckpt_entry = pair_stats_ckpt.get(pair_key_str, {})
        for name in interventions:
            if name in ckpt_entry:
                pair_stats[key][name] = _stats_from_dict(ckpt_entry[name])
            else:
                pair_stats[key][name] = OnlineStats()
        if tracing_enabled and (trace_pairs_set is None or key in trace_pairs_set):
            component_stats[key] = {}
            component_entry = component_stats_ckpt.get(pair_key_str, {})
            for component in trace_components:
                component_key = component["key"]
                if component_key in component_entry:
                    component_stats[key][component_key] = _stats_from_dict(component_entry[component_key])
                else:
                    component_stats[key][component_key] = OnlineStats()
    if tracing_enabled and debug_log:
        _log(f"component_stats pairs={len(component_stats)}")

    wrapper.eval()
    processed_count = len(processed_pairs)
    with torch.no_grad():
        for pair in pairs:
            key = (pair["source"], pair["target"])
            if key in processed_pairs:
                continue
            source_idx = pair["source_idx"]
            target_idx = pair["target_idx"]
            cell_hits = 0
            for batch in dataloader:
                batch = move_to_device(batch, device)
                for idx in range(batch["gene_ids"].shape[0]):
                    sample = {k: v[idx : idx + 1] for k, v in batch.items()}
                    gene_indices = sample["gene_indices"][0]
                    source_positions = find_gene_positions(gene_indices, source_idx)
                    target_positions = find_gene_positions(gene_indices, target_idx)
                    if not source_positions or not target_positions:
                        continue
                    cell_hits += 1
                    _log("processing cell with source/target positions")
                    source_pos = source_positions[0]
                    target_pos = target_positions[0]

                    baseline_val = _readout_value(wrapper, sample, target_pos, output_key, output_reduce)

                    ablated_sample = (
                        apply_pad_ablation(sample, source_positions, dataset_cfg.pad_token_id)
                        if ablation_mode == "pad"
                        else apply_value_ablation(sample, source_positions, baseline_value)
                    )
                    ablated_val = _readout_value(
                        wrapper, ablated_sample, target_pos, output_key, output_reduce
                    )
                    effect = baseline_val - ablated_val
                    if "ablation" in interventions:
                        pair_stats[key]["ablation"].update(effect)

                    if "swap" in interventions:
                        if swap_strategy == "target":
                            swap_pos = target_pos
                        else:
                            valid_positions = [
                                int(pos)
                                for pos in torch.nonzero(gene_indices != -1, as_tuple=False)
                                .squeeze(1)
                                .tolist()
                                if int(pos) != source_pos
                            ]
                            swap_pos = rng.choice(valid_positions) if valid_positions else None
                        if swap_pos is not None:
                            swapped_sample = swap_gene_values(sample, source_pos, swap_pos)
                            swapped_val = _readout_value(
                                wrapper, swapped_sample, target_pos, output_key, output_reduce
                            )
                            pair_stats[key]["swap"].update(baseline_val - swapped_val)

                    if tracing_enabled and (trace_pairs_set is None or key in trace_pairs_set):
                        clean_outputs = _capture_clean_outputs(wrapper, sample, trace_modules)
                        if debug_log:
                            captured = sum(output is not None for output in clean_outputs)
                            _log(f"captured trace outputs={captured}")
                        patch_positions = (
                            source_positions
                            if tracing_cfg.get("patch_position", "source") == "source"
                            else target_positions
                        )
                        seq_len = sample["gene_ids"].shape[1]
                        for component in trace_components:
                            module_idx = int(component["module_idx"])
                            clean_output = clean_outputs[module_idx]
                            if clean_output is None:
                                continue
                            patched_outputs = _forward_with_patch(
                                wrapper,
                                ablated_sample,
                                trace_modules[module_idx],
                                clean_output,
                                patch_positions,
                                batch_size=1,
                                seq_len=seq_len,
                                head_slice=component["head_slice"],
                            )
                            patched_val = _readout_from_outputs(
                                patched_outputs, ablated_sample, target_pos, output_key, output_reduce
                            )
                            restoration = patched_val - ablated_val
                            if tracing_cfg.get("normalize_restoration", False) and effect != 0:
                                restoration = restoration / effect
                            component_key = component["key"]
                            component_stats[key][component_key].update(restoration)
                    if debug_stop == "after_first_cell":
                        _log("stop after first cell")
                        return
                    if max_cells_per_pair and cell_hits >= max_cells_per_pair:
                        break
                if debug_stop == "after_first_batch":
                    _log("stop after first batch")
                    return
                if max_cells_per_pair and cell_hits >= max_cells_per_pair:
                    break
            if debug_stop == "after_first_pair":
                _log("stop after first pair")
                return
            processed_pairs.add(key)
            processed_count += 1
            if progress_every_pairs and processed_count % progress_every_pairs == 0:
                print(
                    f"[causal-progress] processed pairs={processed_count}/{len(pairs)}",
                    flush=True,
                )
            if checkpoint_every_pairs and processed_count % checkpoint_every_pairs == 0:
                _save_checkpoint(checkpoint_path, processed_pairs, pair_stats, component_stats)

    if checkpoint_every_pairs:
        _save_checkpoint(checkpoint_path, processed_pairs, pair_stats, component_stats)

    score_rows = []
    for pair in pairs:
        key = (pair["source"], pair["target"])
        label = pair["label"]
        for intervention, stats in pair_stats[key].items():
            mean, std, n = stats.finalize()
            if n == 0:
                continue
            score_rows.append(
                {
                    "source": key[0],
                    "target": key[1],
                    "label": label,
                    "intervention": intervention,
                    "effect_mean": mean,
                    "effect_std": std,
                    "n_cells": n,
                }
            )

    scores_df = pd.DataFrame(score_rows)
    scores_path = output_dir / "causal_scores.tsv"
    _log(f"writing scores to {scores_path}")
    scores_df.to_csv(scores_path, sep="\t", index=False)

    circuit_rows = []
    if tracing_enabled:
        for pair in pairs:
            key = (pair["source"], pair["target"])
            if key not in component_stats:
                continue
            for component_key, stats in component_stats[key].items():
                mean, std, n = stats.finalize()
                if n == 0:
                    continue
                component, layer_idx, head_idx = _split_component_key(component_key)
                circuit_rows.append(
                    {
                        "source": key[0],
                        "target": key[1],
                        "component": component,
                        "layer": layer_idx,
                        "head": head_idx,
                        "restoration_mean": mean,
                        "restoration_std": std,
                        "n_cells": n,
                    }
                )
    circuit_df = pd.DataFrame(circuit_rows)
    circuit_path = output_dir / "circuit_maps.tsv"
    _log(f"writing circuits to {circuit_path}")
    circuit_df.to_csv(circuit_path, sep="\t", index=False)

    case_studies_path = output_dir / "case_studies.md"
    _log(f"writing case studies to {case_studies_path}")
    case_lines = ["# Causal Intervention Case Studies", ""]
    if not scores_df.empty:
        score_mode = ci_cfg.get("score_mode", "abs")
        for intervention in scores_df["intervention"].unique():
            subset = scores_df[scores_df["intervention"] == intervention]
            labels = subset["label"].to_numpy()
            scores = subset["effect_mean"].to_numpy()
            if score_mode == "abs":
                scores = np.abs(scores)
            if labels.sum() > 0 and labels.sum() < len(labels):
                case_lines.append(
                    f"- {intervention} AUPR: {aupr(scores, labels):.4f} "
                    f"(score_mode={score_mode})"
                )
        case_lines.append("")

    primary_intervention = ci_cfg.get("primary_intervention", "ablation")
    primary = scores_df[scores_df["intervention"] == primary_intervention] if not scores_df.empty else pd.DataFrame()
    if not primary.empty:
        primary = primary.copy()
        primary["abs_effect"] = primary["effect_mean"].abs()
        known = primary[primary["label"] == 1]
        if known.empty:
            known = primary
        if tracing_enabled and not circuit_df.empty:
            trace_keys = set(
                circuit_df["source"].astype(str) + "||" + circuit_df["target"].astype(str)
            )
            primary_keys = known["source"].astype(str) + "||" + known["target"].astype(str)
            traced_primary = known[primary_keys.isin(trace_keys)]
            if not traced_primary.empty:
                known = traced_primary
        top_n = int(ci_cfg.get("case_study_count", 3))
        top_pairs = known.sort_values("abs_effect", ascending=False).head(top_n)
        for _, row in top_pairs.iterrows():
            case_lines.extend(
                [
                    f"## {row['source']} -> {row['target']}",
                    f"- mean effect: {row['effect_mean']:.4f} ± {row['effect_std']:.4f} (n={int(row['n_cells'])})",
                    f"- label: {int(row['label'])}",
                ]
            )
            if tracing_enabled and not circuit_df.empty:
                pair_mask = (circuit_df["source"] == row["source"]) & (
                    circuit_df["target"] == row["target"]
                )
                pair_circuit = circuit_df[pair_mask].copy()
                if not pair_circuit.empty:
                    if "component" not in pair_circuit.columns:
                        pair_circuit["component"] = "layer"
                    if "head" not in pair_circuit.columns:
                        pair_circuit["head"] = np.nan
                    pair_circuit["abs_restoration"] = pair_circuit["restoration_mean"].abs()
                    top_n_components = int(ci_cfg.get("case_study_component_count", 3))
                    top_components = pair_circuit.sort_values("abs_restoration", ascending=False).head(
                        top_n_components
                    )
                    component_labels = []
                    for _, comp_row in top_components.iterrows():
                        component = str(comp_row.get("component", "layer"))
                        layer_idx = int(comp_row["layer"]) if not pd.isna(comp_row["layer"]) else -1
                        head_idx = int(comp_row["head"]) if not pd.isna(comp_row["head"]) else None
                        label = _component_label(component, layer_idx, head_idx)
                        component_labels.append(f"{label}:{comp_row['restoration_mean']:.3f}")
                    case_lines.append(f"- top components: {', '.join(component_labels)}")
                    src_id = _mermaid_id(str(row["source"]), 0)
                    tgt_id = _mermaid_id(str(row["target"]), 1)
                    case_lines.append("```mermaid")
                    case_lines.append("graph LR")
                    case_lines.append(f"  {src_id}[{row['source']}] --> {tgt_id}[{row['target']}]")
                    for idx, comp_row in enumerate(top_components.itertuples(index=False), start=2):
                        component = str(getattr(comp_row, "component", "layer"))
                        layer_val = getattr(comp_row, "layer", float("nan"))
                        layer_idx = int(layer_val) if not pd.isna(layer_val) else -1
                        head_val = getattr(comp_row, "head", float("nan"))
                        head_idx = int(head_val) if not pd.isna(head_val) else None
                        label = _component_label(component, layer_idx, head_idx)
                        comp_id = _mermaid_id(label, idx)
                        case_lines.append(f"  {comp_id}[{label}] --> {tgt_id}")
                    case_lines.append("```")
            case_lines.append("")
    case_studies_path.write_text("\n".join(case_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal interventions for scGPT")
    parser.add_argument("--config", default="configs/causal_intervention.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_causal_interventions(args.config, args.device)


if __name__ == "__main__":
    main()
