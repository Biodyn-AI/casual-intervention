from __future__ import annotations

from typing import Any, Dict, List

import torch
import types


class ScGPTWrapper:
    def __init__(
        self,
        model,
        forward_key_map: Dict[str, str],
        forward_kwargs: Dict[str, Any] | None = None,
    ):
        self.model = model
        self.forward_key_map = forward_key_map
        self.forward_kwargs = forward_kwargs or {}

    def forward(self, batch: Dict[str, Any]):
        inputs = {
            model_key: batch[data_key]
            for data_key, model_key in self.forward_key_map.items()
            if data_key in batch
        }
        return self.model(**inputs, **self.forward_kwargs)

    def eval(self):
        self.model.eval()
        return self

    def forward_with_attentions(self, batch: Dict[str, Any], output_attentions: bool = True):
        inputs = {
            model_key: batch[data_key]
            for data_key, model_key in self.forward_key_map.items()
            if data_key in batch
        }
        if output_attentions:
            try:
                outputs = self.model(**inputs, output_attentions=True, **self.forward_kwargs)
                attentions = _extract_attentions(outputs)
                if attentions is not None:
                    return outputs, attentions
            except TypeError:
                pass
        _enable_torch_attention_capture(self.model)
        _clear_torch_attentions(self.model)
        outputs = self.model(**inputs, **self.forward_kwargs)
        attentions = _extract_attentions(outputs)
        if attentions is None:
            attentions = _collect_torch_attentions(self.model)
        return outputs, attentions


def _extract_attentions(outputs):
    if outputs is None:
        return None
    if isinstance(outputs, dict) and "attentions" in outputs:
        return outputs["attentions"]
    if hasattr(outputs, "attentions"):
        return outputs.attentions
    if isinstance(outputs, tuple) and len(outputs) > 1:
        return outputs[1]
    return None


def _enable_torch_attention_capture(model) -> None:
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)

    for module in model.modules():
        if isinstance(module, torch.nn.TransformerEncoderLayer):
            if hasattr(module, "_scgpt_original_sa_block"):
                continue
            module._scgpt_original_sa_block = module._sa_block

            def _sa_block_with_attn(self, x, attn_mask, key_padding_mask, is_causal=False):
                attn_output, attn_weights = self.self_attn(
                    x,
                    x,
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                    is_causal=is_causal,
                )
                self._last_attn_weights = attn_weights
                return self.dropout1(attn_output)

            module._sa_block = types.MethodType(_sa_block_with_attn, module)


def _clear_torch_attentions(model) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.TransformerEncoderLayer) and hasattr(
            module, "_last_attn_weights"
        ):
            delattr(module, "_last_attn_weights")


def _collect_torch_attentions(model) -> List[torch.Tensor] | None:
    attentions = []
    for module in model.modules():
        if isinstance(module, torch.nn.TransformerEncoderLayer):
            attn = getattr(module, "_last_attn_weights", None)
            if attn is not None:
                attentions.append(attn)
    return attentions if attentions else None
