from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from .config import QuantConfig
from .minifloat import max_magnitude, round_element, round_scale


@dataclass
class GroupDiagnostics:
    """Per-group statistics."""

    scale: float
    magnitude_spread: float  # max / min nonzero magnitude
    clipped: int  # scaled magnitude above the element max
    zeroed: int  # nonzero inputs that became zero
    scale_utilization: float  # block max / (scale * element max)
    poisoned: bool  # NaN/Inf input made the whole group NaN


@dataclass
class QuantResult:
    output: torch.Tensor
    diagnostics: list[GroupDiagnostics]


def _magnitude_spread(finite_mags: list[float]) -> float:
    nonzero = [m for m in finite_mags if m > 0.0]
    if len(nonzero) < 2:
        return float("nan")
    return max(nonzero) / min(nonzero)


def _quantize_group(
    values: list[float], config: QuantConfig, max_mag: float, rng: random.Random
) -> tuple[list[float], GroupDiagnostics]:
    finite_mags = [abs(v) for v in values if math.isfinite(v)]

    if len(finite_mags) != len(values):
        # NaN/Inf poisons the group: the shared max reads every element.
        return [float("nan")] * len(values), GroupDiagnostics(
            scale=float("nan"),
            magnitude_spread=float("nan"),
            clipped=0,
            zeroed=0,
            scale_utilization=float("nan"),
            poisoned=True,
        )

    block_max = max(finite_mags) if finite_mags else 0.0
    if block_max == 0.0:
        return [0.0] * len(values), GroupDiagnostics(
            scale=0.0,
            magnitude_spread=float("nan"),
            clipped=0,
            zeroed=0,
            scale_utilization=0.0,
            poisoned=False,
        )

    scale = round_scale(block_max, max_mag, config.scale_format, config.scale_round)

    nonzero_inputs = sum(1 for v in values if v != 0.0)
    if scale == 0.0:
        # The scale underflowed.
        return [0.0] * len(values), GroupDiagnostics(
            scale=0.0,
            magnitude_spread=_magnitude_spread(finite_mags),
            clipped=0,
            zeroed=nonzero_inputs,
            scale_utilization=0.0,
            poisoned=False,
        )

    out = []
    clipped = 0
    zeroed = 0
    for v in values:
        if v == 0.0:
            out.append(0.0)
            continue
        scaled = v / scale
        if abs(scaled) > max_mag:
            clipped += 1
        q = round_element(
            scaled, config.element_format, config.element_round, config.overflow_policy, rng
        )
        dq = q * scale
        if dq == 0.0:
            zeroed += 1
        out.append(dq)

    utilization = block_max / (scale * max_mag)
    return out, GroupDiagnostics(
        scale=scale,
        magnitude_spread=_magnitude_spread(finite_mags),
        clipped=clipped,
        zeroed=zeroed,
        scale_utilization=utilization,
        poisoned=False,
    )


def quantize_dequantize(tensor: torch.Tensor, config: QuantConfig) -> QuantResult:
    """Quantize and dequantize `tensor`, grouping along `config.axis`.
    The axis size must be divisible by group_size (no padding)."""
    axis_size = tensor.shape[config.axis]
    if axis_size % config.group_size != 0:
        raise ValueError(
            f"axis {config.axis} has size {axis_size}, not divisible by "
            f"group_size {config.group_size}"
        )

    moved = tensor.movedim(config.axis, -1)
    flat = moved.reshape(-1, config.group_size)

    rng = random.Random(config.seed)
    max_mag = max_magnitude(config.element_format)

    out_rows: list[list[float]] = []
    diagnostics: list[GroupDiagnostics] = []
    for row in flat:
        out_row, diag = _quantize_group(row.tolist(), config, max_mag, rng)
        out_rows.append(out_row)
        diagnostics.append(diag)

    out_flat = torch.tensor(out_rows, dtype=torch.float64).reshape(moved.shape)
    output = out_flat.to(tensor.dtype).movedim(-1, config.axis)
    return QuantResult(output=output, diagnostics=diagnostics)
