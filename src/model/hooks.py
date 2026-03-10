from __future__ import annotations

from typing import Callable, List, Tuple

import torch


def register_attention_hooks(
    model,
    module_name_filter: Callable[[str], bool] | None = None,
) -> Tuple[List[torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    cache: List[torch.Tensor] = []
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def hook(_module, _inputs, outputs):
        attn = _extract_attention_from_output(outputs)
        if attn is not None:
            cache.append(attn.detach().cpu())

    for name, module in model.named_modules():
        if module_name_filter and not module_name_filter(name):
            continue
        if "attn" in name.lower() or "attention" in name.lower():
            handles.append(module.register_forward_hook(hook))

    return cache, handles


def _extract_attention_from_output(outputs):
    if outputs is None:
        return None
    if isinstance(outputs, tuple) and len(outputs) > 1:
        candidate = outputs[1]
        if torch.is_tensor(candidate):
            return candidate
    if hasattr(outputs, "attn_weights"):
        return outputs.attn_weights
    return None


def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]):
    for handle in handles:
        handle.remove()
