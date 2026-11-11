from __future__ import annotations

import math
import random

from .config import ElementFormat, ElementRound, OverflowPolicy, ScaleFormat, ScaleRound


def _floor_log2(x: float) -> int:
    """Exact floor(log2(x)) for x > 0 (math.log2 can round up just below a
    power of two)."""
    return math.frexp(x)[1] - 1


def _round_units(units: float, mode: ElementRound, rng: random.Random) -> int:
    """Round a non-negative number of steps to an integer."""
    if mode is ElementRound.RNE:
        return round(units)  # half-to-even
    if mode is ElementRound.TOWARD_ZERO:
        return math.floor(units)
    if mode is ElementRound.STOCHASTIC:
        lower = math.floor(units)
        frac = units - lower
        return lower + 1 if rng.random() < frac else lower
    raise ValueError(f"unhandled ElementRound: {mode}")


def _step_and_max(
    mag: float, exp_bits: int, mantissa_bits: int, bias: int, max_value: float | None = None
) -> tuple[float, float]:
    """Quantization step at magnitude `mag`, and the format's max.
    `max_value` lowers the max for formats with reserved top codes; the step
    is unaffected."""
    if mantissa_bits == 0:
        # E8M0: the top code is NaN.
        max_exp = (1 << exp_bits) - 2 - bias
    else:
        max_exp = (1 << exp_bits) - 1 - bias
    min_normal_exp = 1 - bias
    max_val = (2.0**max_exp) * (2.0 - 2.0**-mantissa_bits)
    if max_value is not None:
        max_val = max_value

    if mag >= 2.0**min_normal_exp:
        exp = min(_floor_log2(mag), max_exp)
        step = 2.0 ** (exp - mantissa_bits)
    else:
        step = 2.0 ** (min_normal_exp - mantissa_bits)
    return step, max_val


def max_magnitude(fmt: ElementFormat) -> float:
    """Largest magnitude `fmt` can represent."""
    if fmt.is_minifloat:
        return _step_and_max(1.0, fmt.exp_bits, fmt.mantissa_bits, fmt.bias, fmt.max_value)[1]
    return float((1 << (fmt.bits - 1)) - 1)


def round_scale(block_max: float, max_mag: float, fmt: ScaleFormat, mode: ScaleRound) -> float:
    """Shared scale for a group with largest magnitude `block_max`.

    Power-of-two scales follow the MX rule: subtract only floor(log2(max_mag))
    from block_max's exponent and leave the element's mantissa headroom
    (E2M1: 6 = 1.5 * 2**2) to element rounding. Scales with mantissa bits
    round block_max / max_mag directly. Always saturates.
    """
    if block_max <= 0.0:
        return 0.0

    if fmt.mantissa_bits == 0:
        emax_elem = _floor_log2(max_mag)
        floor_exp = _floor_log2(block_max)
        if mode is ScaleRound.FLOOR:
            n = floor_exp
        elif mode is ScaleRound.CEIL:
            n = floor_exp + (block_max != 2.0**floor_exp)
        elif mode is ScaleRound.NEAREST:
            n = round(math.log2(block_max))
        else:
            raise ValueError(f"unhandled ScaleRound: {mode}")
        shared_exp = n - emax_elem

        max_exp = (1 << fmt.exp_bits) - 2 - fmt.bias  # top code is NaN
        min_exp = -fmt.bias
        shared_exp = max(min(shared_exp, max_exp), min_exp)
        return 2.0**shared_exp

    value = block_max / max_mag
    step, max_val = _step_and_max(value, fmt.exp_bits, fmt.mantissa_bits, fmt.bias, fmt.max_value)
    units = value / step

    if mode is ScaleRound.FLOOR:
        n = math.floor(units)
    elif mode is ScaleRound.CEIL:
        n = math.ceil(units)
    elif mode is ScaleRound.NEAREST:
        n = round(units)
    else:
        raise ValueError(f"unhandled ScaleRound: {mode}")

    return min(n * step, max_val)


def round_element(
    value: float,
    fmt: ElementFormat,
    mode: ElementRound,
    overflow: OverflowPolicy,
    rng: random.Random,
) -> float:
    """Round an already-scaled value to an element code of `fmt`."""
    if value == 0.0:
        return 0.0

    sign = -1.0 if value < 0.0 else 1.0
    mag = abs(value)

    if fmt.is_minifloat:
        step, max_val = _step_and_max(mag, fmt.exp_bits, fmt.mantissa_bits, fmt.bias, fmt.max_value)
    else:
        step, max_val = 1.0, float((1 << (fmt.bits - 1)) - 1)

    quantized = _round_units(mag / step, mode, rng) * step

    if quantized > max_val:
        if overflow is OverflowPolicy.SATURATE:
            quantized = max_val
        else:
            return float("nan")

    return sign * quantized
