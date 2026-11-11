import itertools
import math

import pytest
import torch

from mx_format_study.formats import (
    ELEMENT_FORMATS,
    SCALE_FORMATS,
    ElementRound,
    OverflowPolicy,
    QuantConfig,
    ScaleRound,
    quantize_dequantize,
)
from mx_format_study.formats.fast import quantize_dequantize_fast, quantize_dequantize_prefix_fast

GROUP = 16


def directed_rows() -> torch.Tensor:
    """Rows hitting the special cases reference.py handles explicitly."""
    rows = [
        [0.0] * GROUP,  # all-zero block
        [float("nan"), 1.0] + [0.0] * (GROUP - 2),  # poisoned
        [float("inf"), -2.0] + [0.0] * (GROUP - 2),
        [1e-40, -1e-41] + [0.0] * (GROUP - 2),  # scale underflow
        [2e38, 1.0] + [0.0] * (GROUP - 2),  # scale saturation
        [4.0, 2.5, 1.5, -2.5, 0.5, 0.25, 0.75, -0.125] + [0.0] * (GROUP - 8),  # ties
        [2.0**k for k in range(GROUP)],  # exact powers of two, wide spread
        [(1 - 2.0**-24) * 2.0**k for k in range(-8, GROUP - 8)],  # just below powers of two
        [7.9, -7.9, 6.0, 5.5] + [0.0] * (GROUP - 4),
    ]
    return torch.tensor(rows, dtype=torch.float32)


def random_rows(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(32, GROUP, generator=g) * torch.exp(torch.randn(32, 1, generator=g) * 4)
    # Heavy-tailed outliers.
    x[:, 0] *= 50.0
    return x


def assert_same(fast, ref):
    assert torch.equal(torch.isnan(fast.output), torch.isnan(ref.output))
    mask = ~torch.isnan(ref.output)
    assert torch.equal(fast.output[mask], ref.output[mask])

    def check(name, fast_vals):
        ref_vals = torch.tensor([getattr(d, name) for d in ref.diagnostics], dtype=torch.float64)
        fast_vals = fast_vals.to(torch.float64)
        assert torch.equal(torch.isnan(fast_vals), torch.isnan(ref_vals)), name
        m = ~torch.isnan(ref_vals)
        assert torch.allclose(fast_vals[m], ref_vals[m], rtol=1e-12, atol=0.0), name

    check("scale", fast.scale)
    check("magnitude_spread", fast.magnitude_spread)
    check("clipped", fast.clipped)
    check("zeroed", fast.zeroed)
    check("scale_utilization", fast.scale_utilization)
    check("poisoned", fast.poisoned)


GRID = list(itertools.product(SCALE_FORMATS.values(), ELEMENT_FORMATS.values()))


@pytest.mark.parametrize("scale_fmt,elem_fmt", GRID, ids=[f"{s.name}-{e.name}" for s, e in GRID])
@pytest.mark.parametrize("scale_round", list(ScaleRound))
@pytest.mark.parametrize("element_round", [ElementRound.RNE, ElementRound.TOWARD_ZERO])
def test_agrees_with_reference(scale_fmt, elem_fmt, scale_round, element_round):
    config = QuantConfig(
        group_size=GROUP,
        scale_format=scale_fmt,
        element_format=elem_fmt,
        scale_round=scale_round,
        element_round=element_round,
    )
    for x in (directed_rows(), random_rows(0), random_rows(1)):
        assert_same(quantize_dequantize_fast(x, config), quantize_dequantize(x, config))


@pytest.mark.parametrize("overflow", list(OverflowPolicy))
def test_agrees_on_overflow_policy(overflow):
    # FLOOR on an E4M3 scale rounds the scale down, which pushes the block
    # max past the element format's top code -- the case the policy decides.
    config = QuantConfig(
        group_size=GROUP,
        scale_format=SCALE_FORMATS["E4M3"],
        element_format=ELEMENT_FORMATS["E2M1"],
        scale_round=ScaleRound.FLOOR,
        overflow_policy=overflow,
    )
    x = random_rows(2)
    assert_same(quantize_dequantize_fast(x, config), quantize_dequantize(x, config))


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_agrees_on_every_grouping_axis(axis):
    config = QuantConfig(
        group_size=8,
        scale_format=SCALE_FORMATS["E8M0"],
        element_format=ELEMENT_FORMATS["E2M1"],
        axis=axis,
    )
    torch.manual_seed(0)
    x = torch.randn(16, 24, 8)
    fast = quantize_dequantize_fast(x, config)
    ref = quantize_dequantize(x, config)
    assert fast.output.shape == x.shape
    assert_same(fast, ref)


def test_preserves_dtype():
    config = QuantConfig(
        group_size=8, scale_format=SCALE_FORMATS["E8M0"], element_format=ELEMENT_FORMATS["E2M1"]
    )
    x = torch.randn(4, 8, dtype=torch.float32)
    assert quantize_dequantize_fast(x, config).output.dtype == torch.float32


def test_rejects_stochastic_rounding():
    config = QuantConfig(
        group_size=8,
        scale_format=SCALE_FORMATS["E8M0"],
        element_format=ELEMENT_FORMATS["E2M1"],
        element_round=ElementRound.STOCHASTIC,
    )
    with pytest.raises(ValueError):
        quantize_dequantize_fast(torch.randn(8), config)


def test_rejects_indivisible_axis():
    config = QuantConfig(
        group_size=32, scale_format=SCALE_FORMATS["E8M0"], element_format=ELEMENT_FORMATS["E2M1"]
    )
    with pytest.raises(ValueError):
        quantize_dequantize_fast(torch.randn(48), config)


def test_has_no_non_finite_output_on_finite_input():
    config = QuantConfig(
        group_size=GROUP, scale_format=SCALE_FORMATS["E8M0"], element_format=ELEMENT_FORMATS["E2M1"]
    )
    out = quantize_dequantize_fast(random_rows(3), config).output
    assert all(math.isfinite(v) for v in out.flatten().tolist())


# --- Causal (prefix-max) variant ---

PREFIX_CONFIGS = [
    QuantConfig(
        group_size=8, scale_format=SCALE_FORMATS["E8M0"], element_format=ELEMENT_FORMATS["E2M1"]
    ),
    QuantConfig(
        group_size=8,
        scale_format=SCALE_FORMATS["E8M0"],
        element_format=ELEMENT_FORMATS["mantissa4"],
    ),
    QuantConfig(
        group_size=8, scale_format=SCALE_FORMATS["E4M3"], element_format=ELEMENT_FORMATS["E2M1"]
    ),
    QuantConfig(
        group_size=8,
        scale_format=SCALE_FORMATS["E4M3"],
        element_format=ELEMENT_FORMATS["E2M1"],
        scale_round=ScaleRound.NEAREST,
    ),
]


def prefix_input() -> torch.Tensor:
    g = torch.Generator().manual_seed(4)
    x = torch.randn(6, 32, generator=g) * torch.exp(torch.randn(6, 1, generator=g) * 3)
    x[:, 5] *= 40.0  # an outlier part-way through a group
    return x


@pytest.mark.parametrize(
    "config",
    PREFIX_CONFIGS,
    ids=lambda c: f"{c.scale_format.name}-{c.element_format.name}-{c.scale_round.value}",
)
def test_prefix_element_equals_standard_on_truncated_group(config):
    # Element i of the causal quantizer must equal the standard quantizer's
    # element i on the same group with every later element zeroed: zeros do
    # not change a group's maximum, so both see the same scale.
    x = prefix_input()
    out = quantize_dequantize_prefix_fast(x, config)
    k = config.group_size
    for start in range(0, x.shape[1], k):
        for i in range(k):
            truncated = x.clone()
            truncated[:, start + i + 1 : start + k] = 0.0
            expected = quantize_dequantize_fast(truncated, config).output
            assert torch.equal(out[:, start + i], expected[:, start + i])


@pytest.mark.parametrize(
    "config",
    PREFIX_CONFIGS,
    ids=lambda c: f"{c.scale_format.name}-{c.element_format.name}-{c.scale_round.value}",
)
def test_prefix_last_element_matches_standard(config):
    x = prefix_input()
    out = quantize_dequantize_prefix_fast(x, config)
    std = quantize_dequantize_fast(x, config).output
    k = config.group_size
    assert torch.equal(out[:, k - 1 :: k], std[:, k - 1 :: k])


def test_prefix_is_causal_along_the_grouping_axis():
    # Changing later tokens must not change earlier outputs, on the token
    # axis of an activation-shaped tensor.
    config = QuantConfig(
        group_size=8,
        scale_format=SCALE_FORMATS["E8M0"],
        element_format=ELEMENT_FORMATS["E2M1"],
        axis=-2,
    )
    torch.manual_seed(0)
    x = torch.randn(1, 32, 16)
    y = x.clone()
    y[:, 20:, :] *= 100.0
    a = quantize_dequantize_prefix_fast(x, config)
    b = quantize_dequantize_prefix_fast(y, config)
    assert torch.equal(a[:, :20, :], b[:, :20, :])
    # The standard quantizer is not causal: token 16..19 share a group with 20..23.
    sa = quantize_dequantize_fast(x, config).output
    sb = quantize_dequantize_fast(y, config).output
    assert not torch.equal(sa[:, 16:20, :], sb[:, 16:20, :])


def test_prefix_rejects_non_finite_input():
    config = QuantConfig(
        group_size=8, scale_format=SCALE_FORMATS["E8M0"], element_format=ELEMENT_FORMATS["E2M1"]
    )
    x = torch.randn(8)
    x[3] = float("nan")
    with pytest.raises(ValueError):
        quantize_dequantize_prefix_fast(x, config)


@pytest.mark.parametrize("impl", [quantize_dequantize, quantize_dequantize_fast])
@pytest.mark.parametrize(
    ("block_max", "mode", "expected_exp"),
    [
        # float64 neighbours of 8.0, where math.log2 rounds to exactly 3.0.
        (math.nextafter(8.0, 0.0), ScaleRound.FLOOR, 2),
        (8.0, ScaleRound.FLOOR, 3),
        (8.0, ScaleRound.CEIL, 3),
        (math.nextafter(8.0, math.inf), ScaleRound.CEIL, 4),
    ],
)
def test_power_of_two_scale_exact_for_float64(impl, block_max, mode, expected_exp):
    """E8M0 scale exponent next to a power of two must not depend on log2's
    rounding; E2M1's max 6.0 has exponent 2, subtracted from the result."""
    config = QuantConfig(
        group_size=GROUP,
        scale_format=SCALE_FORMATS["E8M0"],
        element_format=ELEMENT_FORMATS["E2M1"],
        scale_round=mode,
    )
    x = torch.tensor([block_max] + [0.0] * (GROUP - 1), dtype=torch.float64)
    result = impl(x, config)
    if impl is quantize_dequantize:
        scale = result.diagnostics[0].scale
    else:
        scale = float(result.scale[0])
    assert scale == 2.0 ** (expected_exp - 2)
