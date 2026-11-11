from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScaleRound(Enum):
    """How the ideal (real-valued) scale is rounded to a representable one."""

    FLOOR = "floor"
    CEIL = "ceil"
    NEAREST = "nearest"


class ElementRound(Enum):
    """How an individual scaled element is rounded to a representable code."""

    RNE = "rne"  # MX spec default
    TOWARD_ZERO = "toward_zero"
    STOCHASTIC = "stochastic"


class OverflowPolicy(Enum):
    """Element overflow beyond the format's max. Scales always saturate."""

    SATURATE = "saturate"
    NAN = "nan"


@dataclass(frozen=True)
class ScaleFormat:
    """Format of the scale shared by a group. mantissa_bits=0 means a
    power-of-two scale (E8M0)."""

    name: str
    exp_bits: int
    mantissa_bits: int
    bias: int
    # Largest finite value when the spec reserves top codes for Inf/NaN;
    # None means every code is finite.
    max_value: float | None = None


# FP32 is the unrounded upper bound, not a candidate format.
E8M0 = ScaleFormat("E8M0", exp_bits=8, mantissa_bits=0, bias=127)
E4M3_SCALE = ScaleFormat("E4M3", exp_bits=4, mantissa_bits=3, bias=7, max_value=448.0)
FP16_SCALE = ScaleFormat("FP16", exp_bits=5, mantissa_bits=10, bias=15, max_value=65504.0)
FP32_SCALE = ScaleFormat(
    "FP32", exp_bits=8, mantissa_bits=23, bias=127, max_value=(2.0 - 2.0**-23) * 2.0**127
)

SCALE_FORMATS = {f.name: f for f in (E8M0, E4M3_SCALE, FP16_SCALE, FP32_SCALE)}


@dataclass(frozen=True)
class ElementFormat:
    """Format of one element: pure mantissa (exp_bits=0, uniform grid, as in
    BFP) or minifloat (exp_bits>0, as in MX/NVFP4)."""

    name: str
    bits: int
    exp_bits: int = 0
    mantissa_bits: int = 0
    bias: int = 0
    max_value: float | None = None  # as ScaleFormat.max_value

    @property
    def is_minifloat(self) -> bool:
        return self.exp_bits > 0


def pure_mantissa(bits: int) -> ElementFormat:
    """A classic BFP element: sign + (bits-1) magnitude bits, no exponent."""
    return ElementFormat(name=f"mantissa{bits}", bits=bits)


MANTISSA2 = pure_mantissa(2)
MANTISSA4 = pure_mantissa(4)
MANTISSA6 = pure_mantissa(6)
MANTISSA8 = pure_mantissa(8)

# FP4/FP6 have no reserved codes (OCP MX v1.0). OCP FP8 does: E4M3's
# S.1111.111 is NaN (max 448) and E5M2's top exponent is Inf/NaN (max 57344).
E2M1 = ElementFormat("E2M1", bits=4, exp_bits=2, mantissa_bits=1, bias=1)
E3M2 = ElementFormat("E3M2", bits=6, exp_bits=3, mantissa_bits=2, bias=3)
E2M3 = ElementFormat("E2M3", bits=6, exp_bits=2, mantissa_bits=3, bias=1)
E4M3 = ElementFormat("E4M3", bits=8, exp_bits=4, mantissa_bits=3, bias=7, max_value=448.0)
E5M2 = ElementFormat("E5M2", bits=8, exp_bits=5, mantissa_bits=2, bias=15, max_value=57344.0)

ELEMENT_FORMATS = {
    f.name: f for f in (MANTISSA2, MANTISSA4, MANTISSA6, MANTISSA8, E2M1, E3M2, E2M3, E4M3, E5M2)
}


@dataclass(frozen=True)
class QuantConfig:
    """One point in the design space."""

    group_size: int
    scale_format: ScaleFormat
    element_format: ElementFormat
    axis: int = -1
    scale_round: ScaleRound = ScaleRound.FLOOR
    element_round: ElementRound = ElementRound.RNE
    overflow_policy: OverflowPolicy = OverflowPolicy.SATURATE
    seed: int | None = None  # for STOCHASTIC rounding
