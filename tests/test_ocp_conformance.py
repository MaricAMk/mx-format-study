import math
from pathlib import Path

import pytest
import torch
from mx.mx_ops import quantize_mx_op
from mx.specs import MxSpecs

from mx_format_study.formats import (
    E4M3_SCALE,
    E8M0,
    ELEMENT_FORMATS,
    OverflowPolicy,
    QuantConfig,
    ScaleRound,
    quantize_dequantize,
    quantize_dequantize_fast,
)
from mx_format_study.formats.minifloat import max_magnitude, round_scale

TABLE = Path(__file__).resolve().parents[1] / "conformance_vectors" / "ocp_element_codes.txt"

# Published anchors: OCP MX v1.0 table 1 (FP4/FP6), OFP8 v1.0 table 1 (FP8).
SPEC_ANCHORS = {
    "E2M1": (6.0, 0.5),
    "E2M3": (7.5, 0.125),
    "E3M2": (28.0, 0.0625),
    "E4M3": (448.0, 2.0**-9),
    "E5M2": (57344.0, 2.0**-16),
}
UPSTREAM_NAMES = {
    "E2M1": "fp4_e2m1",
    "E2M3": "fp6_e2m3",
    "E3M2": "fp6_e3m2",
    "E4M3": "fp8_e4m3",
    "E5M2": "fp8_e5m2",
}


def load_table() -> dict[str, list[tuple[int, float]]]:
    table: dict[str, list[tuple[int, float]]] = {}
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, code, value = line.split()
        v = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}.get(value)
        table.setdefault(name, []).append((int(code, 16), float.fromhex(value) if v is None else v))
    return table


TABLE_DATA = load_table()


def finite_grid(name: str) -> list[tuple[int, float]]:
    """(code, value) for the non-negative finite codes, ascending by value;
    +0 only (the -0 code decodes to the same value)."""
    pos = [(c, v) for c, v in TABLE_DATA[name] if math.isfinite(v) and math.copysign(1.0, v) > 0]
    return sorted(pos, key=lambda cv: cv[1])


def grid_block(values: list[float], name: str) -> torch.Tensor:
    """One group per value: [value, spec max, 0...]. The spec max pins the
    E8M0 scale to exactly 1.0, so the element grid is tested directly."""
    spec_max = SPEC_ANCHORS[name][0]
    rows = [[v, spec_max] + [0.0] * 6 for v in values]
    return torch.tensor(rows, dtype=torch.float64)


def config(name: str, overflow=OverflowPolicy.SATURATE) -> QuantConfig:
    return QuantConfig(
        group_size=8,
        scale_format=E8M0,
        element_format=ELEMENT_FORMATS[name],
        overflow_policy=overflow,
    )


@pytest.mark.parametrize("name", list(SPEC_ANCHORS))
class TestElementCodes:
    def test_table_matches_spec_anchors(self, name):
        grid = [v for _, v in finite_grid(name)]
        spec_max, spec_min_sub = SPEC_ANCHORS[name]
        assert max(grid) == spec_max
        assert min(v for v in grid if v > 0) == spec_min_sub

    def test_format_max_matches_spec(self, name):
        assert max_magnitude(ELEMENT_FORMATS[name]) == SPEC_ANCHORS[name][0]

    def test_every_spec_value_round_trips(self, name):
        values = [v for _, v in finite_grid(name)]
        x = grid_block(values + [-v for v in values], name)
        for quantize in (quantize_dequantize, quantize_dequantize_fast):
            out = quantize(x, config(name)).output
            assert torch.equal(out[:, 0], x[:, 0]), quantize.__name__

    def test_midpoints_round_to_even_code(self, name):
        grid = finite_grid(name)
        mids, expected = [], []
        for (c_lo, lo), (c_hi, hi) in zip(grid, grid[1:]):
            mids.append((lo + hi) / 2)
            expected.append(lo if c_lo % 2 == 0 else hi)
        x = grid_block(mids, name)
        for quantize in (quantize_dequantize, quantize_dequantize_fast):
            out = quantize(x, config(name)).output
            assert out[:, 0].tolist() == expected, quantize.__name__

    def test_past_max_saturates_or_nans(self, name):
        spec_max = SPEC_ANCHORS[name][0]
        grid = [v for _, v in finite_grid(name)]
        top_step = spec_max - grid[-2]
        # Past the halfway point to the next (nonexistent) code: rounds up
        # beyond max, so the overflow policy decides.
        beyond = spec_max + 0.75 * top_step
        x = grid_block([beyond], name)
        # Keep the scale at 1.0: floor(log2(beyond)) == floor(log2(max)).
        assert math.floor(math.log2(beyond)) == math.floor(math.log2(spec_max))
        for quantize in (quantize_dequantize, quantize_dequantize_fast):
            assert quantize(x, config(name)).output[0, 0].item() == spec_max
            nan_out = quantize(x, config(name, OverflowPolicy.NAN)).output
            assert math.isnan(nan_out[0, 0].item())

    def test_microxcaling_agrees_on_grid_and_midpoints(self, name):
        grid = finite_grid(name)
        values = [v for _, v in grid] + [(a[1] + b[1]) / 2 for a, b in zip(grid, grid[1:])]
        x = grid_block(values, name).to(torch.float32)
        s = MxSpecs()
        s["block_size"] = 8
        s["scale_bits"] = 8
        s["shared_exp_method"] = "max"
        upstream = quantize_mx_op(
            x.clone(), s, elem_format=UPSTREAM_NAMES[name], axes=-1, round="even"
        )
        ours = quantize_dequantize_fast(x, config(name)).output
        assert torch.equal(ours, upstream)


class TestScaleCodes:
    def test_e8m0_table(self):
        codes = dict(TABLE_DATA["E8M0"])
        assert math.isnan(codes[0xFF])
        assert codes[0x00] == 2.0**-127
        assert codes[0x7F] == 1.0
        assert codes[0xFE] == 2.0**127

    def test_e8m0_scales_are_table_values(self):
        legal = {v for _, v in TABLE_DATA["E8M0"] if math.isfinite(v)}
        for block_max in (1e-40, 1e-30, 0.3, 1.0, 6.0, 1e10, 3e38):
            s = round_scale(block_max, 6.0, E8M0, ScaleRound.FLOOR)
            assert s in legal, block_max

    @pytest.mark.parametrize("mode", list(ScaleRound))
    def test_e4m3_scales_are_table_values(self, mode):
        # NVFP4's scale: every scale chosen must be a finite E4M3 code value,
        # including at saturation (448, not 480).
        legal = {v for _, v in TABLE_DATA["E4M3"] if math.isfinite(v) and v > 0}
        torch.manual_seed(0)
        maxes = (torch.exp(torch.randn(2000) * 6)).tolist() + [6.0 * 448.0, 6.0 * 470.0, 1e6]
        for block_max in maxes:
            s = round_scale(block_max, 6.0, E4M3_SCALE, mode)
            assert s == 0.0 or s in legal, (block_max, s)
        assert round_scale(1e6, 6.0, E4M3_SCALE, mode) == 448.0
