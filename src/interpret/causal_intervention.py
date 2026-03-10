from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


@dataclass
class OnlineStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def finalize(self) -> tuple[float, float, int]:
        if self.count < 2:
            return self.mean, 0.0, self.count
        variance = self.m2 / (self.count - 1)
        return self.mean, variance**0.5, self.count


def clone_sample(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in sample.items()}


def find_gene_positions(gene_indices: torch.Tensor, gene_idx: int) -> List[int]:
    matches = torch.nonzero(gene_indices == gene_idx, as_tuple=False).squeeze(1)
    return matches.detach().cpu().tolist()


def apply_value_ablation(
    sample: Dict[str, torch.Tensor],
    positions: Iterable[int],
    baseline_value: float,
) -> Dict[str, torch.Tensor]:
    edited = clone_sample(sample)
    if positions:
        edited["gene_values"][0, list(positions)] = baseline_value
    return edited


def apply_pad_ablation(
    sample: Dict[str, torch.Tensor],
    positions: Iterable[int],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    edited = clone_sample(sample)
    if positions:
        pos = list(positions)
        edited["gene_ids"][0, pos] = pad_token_id
        edited["gene_values"][0, pos] = 0.0
        edited["src_key_padding_mask"] = edited["gene_ids"] == pad_token_id
    return edited


def swap_gene_values(
    sample: Dict[str, torch.Tensor],
    pos_a: int,
    pos_b: int,
) -> Dict[str, torch.Tensor]:
    edited = clone_sample(sample)
    values = edited["gene_values"]
    tmp = values[0, pos_a].clone()
    values[0, pos_a] = values[0, pos_b]
    values[0, pos_b] = tmp
    return edited


def extract_output_tensor(outputs, output_key: str | None) -> torch.Tensor:
    if output_key:
        if isinstance(outputs, dict) and output_key in outputs:
            return outputs[output_key]
        if hasattr(outputs, output_key):
            return getattr(outputs, output_key)
        raise KeyError(f"Output key '{output_key}' not found in model outputs")

    if isinstance(outputs, dict):
        for key in ("mlm_output", "mvc_output", "logits", "pred", "output"):
            value = outputs.get(key)
            if torch.is_tensor(value):
                return value
        for value in outputs.values():
            if torch.is_tensor(value):
                return value
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)):
        for value in outputs:
            if torch.is_tensor(value):
                return value
    raise ValueError("Unable to extract a tensor output from the model outputs")


def reduce_output(output: torch.Tensor, reduce_mode: str | None) -> torch.Tensor:
    if output.dim() == 2:
        return output
    if output.dim() != 3:
        raise ValueError(f"Expected output with 2 or 3 dims, got {output.shape}")

    if reduce_mode is None or reduce_mode == "none":
        raise ValueError("output_reduce must be set for 3D outputs")
    if reduce_mode == "mean":
        return output.mean(dim=-1)
    if reduce_mode == "sum":
        return output.sum(dim=-1)
    raise ValueError(f"Unsupported output_reduce: {reduce_mode}")


def align_output_to_batch_seq(
    output: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    if output.dim() == 2:
        if output.shape == (batch_size, seq_len):
            return output
        if output.shape == (seq_len, batch_size):
            return output.t()
    elif output.dim() == 3:
        if output.shape[0] == batch_size and output.shape[1] == seq_len:
            return output
        if output.shape[0] == seq_len and output.shape[1] == batch_size:
            return output.permute(1, 0, 2)
    raise ValueError(
        f"Output shape {tuple(output.shape)} does not align with "
        f"batch_size={batch_size} seq_len={seq_len}"
    )


def find_transformer_layers(model) -> List[torch.nn.Module]:
    layers = [module for module in model.modules() if isinstance(module, torch.nn.TransformerEncoderLayer)]
    if layers:
        return layers

    for attr in ("transformer_encoder", "encoder", "transformer"):
        parent = getattr(model, attr, None)
        if parent is not None and hasattr(parent, "layers"):
            return list(parent.layers)

    custom_layers = []
    for module in model.modules():
        name = module.__class__.__name__
        if "EncoderLayer" in name and hasattr(module, "self_attn"):
            custom_layers.append(module)
    return custom_layers


def find_attention_modules(layers: List[torch.nn.Module]) -> List[torch.nn.Module]:
    modules: List[torch.nn.Module] = []
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            for name, module in layer.named_modules():
                if not name:
                    continue
                lname = name.lower()
                if "attn" in lname or "attention" in lname:
                    attn = module
                    break
        if attn is None:
            raise ValueError("Unable to locate attention module for transformer layer")
        modules.append(attn)
    return modules


def find_mlp_modules(layers: List[torch.nn.Module]) -> List[torch.nn.Module]:
    modules: List[torch.nn.Module] = []
    for layer in layers:
        mlp = getattr(layer, "linear2", None)
        if mlp is None:
            for name, module in layer.named_modules():
                if not name:
                    continue
                if "linear2" in name.lower():
                    mlp = module
                    break
        if mlp is None:
            for name, module in layer.named_modules():
                if not name:
                    continue
                lname = name.lower()
                if "mlp" in lname or "ff" in lname or "feedforward" in lname:
                    mlp = module
                    break
        if mlp is None:
            raise ValueError("Unable to locate MLP module for transformer layer")
        modules.append(mlp)
    return modules


def attention_head_info(attn_module: torch.nn.Module) -> Tuple[int, int]:
    num_heads = getattr(attn_module, "num_heads", None) or getattr(attn_module, "nhead", None)
    if num_heads is None:
        raise ValueError("Attention module missing num_heads")
    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None:
        embed_dim = getattr(attn_module, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("Attention module missing embed_dim")
        head_dim = embed_dim // int(num_heads)
    return int(num_heads), int(head_dim)


def attention_head_slice(attn_module: torch.nn.Module, head_idx: int) -> Tuple[int, int]:
    num_heads, head_dim = attention_head_info(attn_module)
    if head_idx < 0 or head_idx >= num_heads:
        raise ValueError(f"head_idx={head_idx} out of range for {num_heads} heads")
    start = head_idx * head_dim
    end = start + head_dim
    return start, end


def capture_layer_outputs(
    layers: List[torch.nn.Module],
    outputs: List[torch.Tensor | None],
) -> List[torch.utils.hooks.RemovableHandle]:
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    for idx, layer in enumerate(layers):

        def _hook(module, _inputs, layer_output, index=idx):
            if isinstance(layer_output, (tuple, list)):
                layer_tensor = layer_output[0]
            else:
                layer_tensor = layer_output
            outputs[index] = layer_tensor

        hooks.append(layer.register_forward_hook(_hook))

    return hooks


def _unwrap_module_output(module_output: Any) -> torch.Tensor:
    if isinstance(module_output, (tuple, list)):
        return module_output[0]
    if torch.is_tensor(module_output):
        return module_output
    raise ValueError("Module output did not contain a tensor")


def capture_module_outputs(
    modules: List[torch.nn.Module],
    outputs: List[torch.Tensor | None],
) -> List[torch.utils.hooks.RemovableHandle]:
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    for idx, module in enumerate(modules):

        def _hook(_module, _inputs, module_output, index=idx):
            outputs[index] = _unwrap_module_output(module_output)

        hooks.append(module.register_forward_hook(_hook))

    return hooks


def patch_layer_output(
    layer_output: torch.Tensor,
    clean_output: torch.Tensor,
    positions: Iterable[int],
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    if batch_size != 1:
        raise ValueError("Layer patching expects batch_size=1 for position-based edits")
    pos = list(positions)
    if not pos:
        return layer_output

    if layer_output.dim() < 3:
        raise ValueError("Layer output must be 3D for sequence patching")

    if layer_output.shape[0] == seq_len and layer_output.shape[1] == batch_size:
        patched = layer_output.clone()
        patched[pos, :, :] = clean_output[pos, :, :]
        return patched
    if layer_output.shape[0] == batch_size and layer_output.shape[1] == seq_len:
        patched = layer_output.clone()
        patched[:, pos, :] = clean_output[:, pos, :]
        return patched
    raise ValueError(
        "Layer output shape does not align with batch/sequence for patching: "
        f"{tuple(layer_output.shape)}"
    )


def patch_module_output(
    module_output: Any,
    clean_output: torch.Tensor,
    positions: Iterable[int],
    batch_size: int,
    seq_len: int,
    head_slice: Optional[Tuple[int, int]] = None,
) -> Any:
    if batch_size != 1:
        raise ValueError("Module patching expects batch_size=1 for position-based edits")
    pos = list(positions)
    if not pos:
        return module_output

    tensor = _unwrap_module_output(module_output)
    if tensor.dim() < 3:
        raise ValueError("Module output must be 3D for sequence patching")

    def _apply_patch(patched: torch.Tensor) -> torch.Tensor:
        if head_slice is None:
            if patched.shape[0] == batch_size and patched.shape[1] == seq_len:
                patched[:, pos, :] = clean_output[:, pos, :]
                return patched
            if patched.shape[0] == seq_len and patched.shape[1] == batch_size:
                patched[pos, :, :] = clean_output[pos, :, :]
                return patched
        else:
            start, end = head_slice
            if patched.shape[0] == batch_size and patched.shape[1] == seq_len:
                patched[:, pos, start:end] = clean_output[:, pos, start:end]
                return patched
            if patched.shape[0] == seq_len and patched.shape[1] == batch_size:
                patched[pos, :, start:end] = clean_output[pos, :, start:end]
                return patched
        raise ValueError(
            "Module output shape does not align with batch/sequence for patching: "
            f"{tuple(patched.shape)}"
        )

    patched_tensor = _apply_patch(tensor.clone())
    if isinstance(module_output, tuple):
        return (patched_tensor,) + tuple(module_output[1:])
    if isinstance(module_output, list):
        return [patched_tensor] + list(module_output[1:])
    return patched_tensor
