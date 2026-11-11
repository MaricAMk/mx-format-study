from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from ..formats import QuantConfig, quantize_dequantize_fast, quantize_dequantize_prefix_fast


def target_linears(model: nn.Module) -> dict[str, nn.Linear]:
    """Every nn.Linear except the LM head."""
    return {
        name: m
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and "lm_head" not in name
    }


@dataclass
class WeightStats:
    """Diagnostics summed over all quantized weights."""

    groups: int = 0
    clipped: int = 0
    zeroed: int = 0
    elements: int = 0
    mse: float = 0.0  # sum of squared error
    energy: float = 0.0  # sum of squared weights
    spread_log2: list[torch.Tensor] = field(default_factory=list)

    @property
    def relative_mse(self) -> float:
        return self.mse / self.energy


@torch.no_grad()
def quantize_weights(model: nn.Module, config: QuantConfig, device: str = "cpu") -> WeightStats:
    """Quantize each target weight in place; `device` is where the
    quantizer runs."""
    stats = WeightStats()
    for m in target_linears(model).values():
        w = m.weight.data
        r = quantize_dequantize_fast(w.to(device), config)
        q = r.output.to(w.device, w.dtype)
        stats.groups += r.scale.numel()
        stats.clipped += int(r.clipped.sum())
        stats.zeroed += int(r.zeroed.sum())
        stats.elements += w.numel()
        stats.mse += float(((q.double() - w.double()) ** 2).sum())
        stats.energy += float((w.double() ** 2).sum())
        spread = r.magnitude_spread
        stats.spread_log2.append(torch.log2(spread[torch.isfinite(spread)]).float().cpu())
        w.copy_(q)
    return stats


class ActivationQuantizer:
    """Context manager that quantizes each target linear's input.
    causal=True uses quantize_dequantize_prefix_fast."""

    def __init__(self, model: nn.Module, config: QuantConfig, causal: bool = False):
        self.model = model
        self.config = config
        self.causal = causal
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, module: nn.Module, args: tuple) -> tuple:
        (x, *rest) = args
        if self.causal:
            q = quantize_dequantize_prefix_fast(x, self.config)
        else:
            q = quantize_dequantize_fast(x, self.config, diagnostics=False).output
        return (q, *rest)

    def __enter__(self) -> ActivationQuantizer:
        for m in target_linears(self.model).values():
            self._handles.append(m.register_forward_pre_hook(self._hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
