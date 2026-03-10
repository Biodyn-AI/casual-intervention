from __future__ import annotations

from typing import Callable

import torch


def integrated_gradients(
    inputs: torch.Tensor,
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    baseline: torch.Tensor | None = None,
    steps: int = 50,
):
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(1, steps + 1)]
    grads = []

    for scaled in scaled_inputs:
        scaled = scaled.clone().detach().requires_grad_(True)
        outputs = model_fn(scaled)
        if outputs.dim() != 0:
            outputs = outputs.sum()
        outputs.backward()
        grads.append(scaled.grad.detach())

    avg_grads = torch.stack(grads, dim=0).mean(dim=0)
    return (inputs - baseline) * avg_grads
