from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import (
    ElementFormat,
    ElementRound,
    OverflowPolicy,
    QuantConfig,
    ScaleFormat,
    ScaleRound,
)
from .minifloat import max_magnitude


@dataclass
class FastQuantResult:
    """Per-group diagnostics as tensors, in reference.py's group order."""

    output: torch.Tensor
    scale: torch.Tensor
    magnitude_spread: torch.Tensor
    clipped: torch.Tensor
    zeroed: torch.Tensor
    scale_utilization: torch.Tensor
    poisoned: torch.Tensor


def _floor_log2(x: torch.Tensor) -> torch.Tensor:
    _, e = torch.frexp(x)
    return (e - 1).to(x.dtype)


def _round(units: torch.Tensor, mode: ScaleRound) -> torch.Tensor:
    if mode is ScaleRound.FLOOR:
        return torch.floor(units)
    if mode is ScaleRound.CEIL:
        return torch.ceil(units)
    if mode is ScaleRound.NEAREST:
        return torch.round(units)  # half-to-even
    raise ValueError(f"unhandled ScaleRound: {mode}")


def _step_and_max(
    mag: torch.Tensor, exp_bits: int, mantissa_bits: int, bias: int, max_value: float | None = None
) -> tuple[torch.Tensor, float]:
    """Vectorized minifloat._step_and_max."""
    if mantissa_bits == 0:
        max_exp = (1 << exp_bits) - 2 - bias
    else:
        max_exp = (1 << exp_bits) - 1 - bias
    min_normal_exp = 1 - bias
    max_val = (2.0**max_exp) * (2.0 - 2.0**-mantissa_bits)
    if max_value is not None:
        max_val = max_value

    exp = torch.clamp(_floor_log2(mag), max=max_exp)
    normal_step = torch.exp2(exp - mantissa_bits)
    subnormal_step = 2.0 ** (min_normal_exp - mantissa_bits)
    step = torch.where(
        mag >= 2.0**min_normal_exp, normal_step, torch.full_like(mag, subnormal_step)
    )
    return step, max_val


def _round_scale(
    block_max: torch.Tensor, max_mag: float, fmt: ScaleFormat, mode: ScaleRound
) -> torch.Tensor:
    """Vectorized minifloat.round_scale. Groups with block_max == 0 get 0."""
    positive = block_max > 0.0
    safe_max = torch.where(positive, block_max, torch.ones_like(block_max))

    if fmt.mantissa_bits == 0:
        emax_elem = _floor_log2(torch.tensor(max_mag, dtype=torch.float64)).item()
        floor_exp = _floor_log2(safe_max)
        if mode is ScaleRound.FLOOR:
            n = floor_exp
        elif mode is ScaleRound.CEIL:
            n = floor_exp + (safe_max != torch.exp2(floor_exp)).to(floor_exp.dtype)
        else:
            n = _round(torch.log2(safe_max), mode)
        max_exp = (1 << fmt.exp_bits) - 2 - fmt.bias
        shared_exp = torch.clamp(n - emax_elem, min=-fmt.bias, max=max_exp)
        scale = torch.exp2(shared_exp)
    else:
        value = safe_max / max_mag
        step, max_val = _step_and_max(
            value, fmt.exp_bits, fmt.mantissa_bits, fmt.bias, fmt.max_value
        )
        scale = torch.clamp(_round(value / step, mode) * step, max=max_val)

    return torch.where(positive, scale, torch.zeros_like(scale))


def _round_elements(
    scaled: torch.Tensor, fmt: ElementFormat, mode: ElementRound, overflow: OverflowPolicy
) -> torch.Tensor:
    """Vectorized minifloat.round_element."""
    mag = scaled.abs()
    if fmt.is_minifloat:
        step, max_val = _step_and_max(mag, fmt.exp_bits, fmt.mantissa_bits, fmt.bias, fmt.max_value)
    else:
        step, max_val = torch.ones_like(mag), float((1 << (fmt.bits - 1)) - 1)

    units = mag / step
    if mode is ElementRound.RNE:
        units = torch.round(units)
    elif mode is ElementRound.TOWARD_ZERO:
        units = torch.floor(units)
    else:
        raise ValueError(f"fast path does not support ElementRound {mode}; use reference.py")
    quantized = units * step

    over = quantized > max_val
    if overflow is OverflowPolicy.SATURATE:
        quantized = torch.where(over, torch.full_like(quantized, max_val), quantized)
    else:
        quantized = torch.where(over, torch.full_like(quantized, float("nan")), quantized)
    return torch.sign(scaled) * quantized


def quantize_dequantize_fast(
    tensor: torch.Tensor, config: QuantConfig, diagnostics: bool = True
) -> FastQuantResult:
    """Same as reference.quantize_dequantize. With diagnostics=False only
    `output` is filled in."""
    axis_size = tensor.shape[config.axis]
    if axis_size % config.group_size != 0:
        raise ValueError(
            f"axis {config.axis} has size {axis_size}, not divisible by "
            f"group_size {config.group_size}"
        )

    moved = tensor.movedim(config.axis, -1)
    x = moved.reshape(-1, config.group_size).to(torch.float64)
    max_mag = max_magnitude(config.element_format)

    finite = torch.isfinite(x)
    poisoned = ~finite.all(dim=1)
    mags = torch.where(finite, x.abs(), torch.zeros_like(x))
    block_max = mags.amax(dim=1)

    scale = _round_scale(block_max, max_mag, config.scale_format, config.scale_round)
    live = (scale > 0.0) & ~poisoned
    safe_scale = torch.where(live, scale, torch.ones_like(scale)).unsqueeze(1)

    scaled = torch.where(finite, x, torch.zeros_like(x)) / safe_scale
    q = _round_elements(scaled, config.element_format, config.element_round, config.overflow_policy)
    out = q * safe_scale
    out = torch.where(live.unsqueeze(1), out, torch.zeros_like(out))
    out = torch.where(poisoned.unsqueeze(1), torch.full_like(out, float("nan")), out)

    output = out.reshape(moved.shape).to(tensor.dtype).movedim(-1, config.axis)
    if not diagnostics:
        empty = torch.empty(0)
        return FastQuantResult(output, empty, empty, empty, empty, empty, empty)

    nonzero = (x != 0.0) & finite
    clipped = ((scaled.abs() > max_mag) & nonzero & live.unsqueeze(1)).sum(dim=1)
    zeroed = torch.where(live.unsqueeze(1), (out == 0.0) & nonzero, nonzero).sum(dim=1)

    nz_count = (mags > 0.0).sum(dim=1)
    nz_min = torch.where(mags > 0.0, mags, torch.full_like(mags, float("inf"))).amin(dim=1)
    spread = torch.where(
        nz_count >= 2, block_max / nz_min, torch.full_like(block_max, float("nan"))
    )

    utilization = torch.where(
        live, block_max / (safe_scale.squeeze(1) * max_mag), torch.zeros_like(block_max)
    )

    nan = torch.full_like(block_max, float("nan"))
    scale = torch.where(poisoned, nan, scale)
    spread = torch.where(poisoned, nan, spread)
    utilization = torch.where(poisoned, nan, utilization)
    clipped = torch.where(poisoned, torch.zeros_like(clipped), clipped)
    zeroed = torch.where(poisoned | (block_max == 0.0), torch.zeros_like(zeroed), zeroed)

    return FastQuantResult(output, scale, spread, clipped, zeroed, utilization, poisoned)


def quantize_dequantize_prefix_fast(tensor: torch.Tensor, config: QuantConfig) -> torch.Tensor:
    """Causal variant: each element's scale comes from the running max of
    its group up to and including itself, so no element depends on later
    ones. Returns the output only; inputs must be finite."""
    axis_size = tensor.shape[config.axis]
    if axis_size % config.group_size != 0:
        raise ValueError(
            f"axis {config.axis} has size {axis_size}, not divisible by "
            f"group_size {config.group_size}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("quantize_dequantize_prefix_fast requires finite inputs")

    moved = tensor.movedim(config.axis, -1)
    x = moved.reshape(-1, config.group_size).to(torch.float64)
    max_mag = max_magnitude(config.element_format)

    prefix_max = torch.cummax(x.abs(), dim=1).values
    scale = _round_scale(prefix_max.reshape(-1), max_mag, config.scale_format, config.scale_round)
    scale = scale.reshape(prefix_max.shape)
    live = scale > 0.0
    safe_scale = torch.where(live, scale, torch.ones_like(scale))

    q = _round_elements(
        x / safe_scale, config.element_format, config.element_round, config.overflow_policy
    )
    out = torch.where(live, q * safe_scale, torch.zeros_like(x))
    return out.reshape(moved.shape).to(tensor.dtype).movedim(-1, config.axis)
