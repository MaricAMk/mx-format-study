import math

import pytest
import torch
from mx.mx_ops import quantize_mx_op
from mx.specs import MxSpecs

from mx_format_study.formats import (
    E2M1,
    E8M0,
    ElementRound,
    OverflowPolicy,
    QuantConfig,
    ScaleRound,
    quantize_dequantize,
)

MXFP4 = QuantConfig(
    group_size=32,
    scale_format=E8M0,
    element_format=E2M1,
    scale_round=ScaleRound.FLOOR,
    element_round=ElementRound.RNE,
    overflow_policy=OverflowPolicy.SATURATE,
)


def padded(*head: float) -> list[float]:
    return list(head) + [0.0] * (MXFP4.group_size - len(head))


def quantize(values: list[float], config: QuantConfig = MXFP4) -> torch.Tensor:
    assert len(values) == config.group_size
    x = torch.tensor(values, dtype=torch.float32)
    return quantize_dequantize(x, config).output


class TestDirectedSpecialValues:
    def test_all_zero_block(self):
        out = quantize([0.0] * MXFP4.group_size)
        assert torch.all(out == 0.0)

    def test_nan_poisons_entire_block(self):
        out = quantize(padded(float("nan"), 1.0, 2.0))
        assert torch.all(torch.isnan(out))

    def test_inf_poisons_entire_block(self):
        out = quantize(padded(float("inf"), 1.0, 2.0))
        assert torch.all(torch.isnan(out))

    def test_scale_saturates_at_e8m0_max(self):
        out = quantize(padded(2e38))
        assert torch.isfinite(out[0])
        assert torch.isclose(out[0], torch.tensor(2.0**127), rtol=1e-3)

    def test_extreme_denormal_flushes_to_zero(self):
        out = quantize(padded(1e-40))
        assert torch.all(out == 0.0)

    def test_sign_symmetry(self):
        pos = quantize(padded(3.0, 1.0, 0.4))
        neg = quantize(padded(-3.0, -1.0, -0.4))
        assert torch.equal(neg, -pos)


class TestProperties:
    def test_round_trip_idempotence(self):
        torch.manual_seed(0)
        x = torch.randn(MXFP4.group_size) * 3.0
        once = quantize_dequantize(x, MXFP4).output
        twice = quantize_dequantize(once, MXFP4).output
        assert torch.equal(once, twice)

    def test_monotonicity_within_group(self):
        xs = sorted([0.1, 0.4, 0.6, 0.9, 1.2, 1.7, 2.2, 2.8, 3.5, 4.5])
        out = quantize(padded(*xs))
        ordered = out[: len(xs)]
        assert torch.all(ordered[1:] >= ordered[:-1])

    @pytest.mark.parametrize("round_mode", [ElementRound.RNE, ElementRound.TOWARD_ZERO])
    def test_zero_maps_to_zero_under_every_rounding_mode(self, round_mode):
        config = QuantConfig(**{**MXFP4.__dict__, "element_round": round_mode})
        out = quantize(padded(0.0, 3.0, 0.0, 1.5, 0.0), config)
        assert out[0] == 0.0
        assert out[2] == 0.0
        assert out[4] == 0.0

    def test_exhaustive_exponent_delta_coverage(self):
        prev = float("inf")
        zero_seen = False
        for delta in range(256):
            probe_value = 2.0 ** (-delta)
            out = quantize(padded(1.0, probe_value))
            probe_out = out[1].item()

            assert math.isfinite(probe_out), f"delta={delta} produced non-finite output"
            assert probe_out <= prev + 1e-30, f"delta={delta} increased magnitude"
            prev = probe_out
            if probe_out == 0.0:
                zero_seen = True
            if zero_seen:
                assert probe_out == 0.0, f"delta={delta} un-flushed after reaching zero"


class TestUpstreamAgreement:
    """MXFP4 must match microxcaling exactly."""

    @pytest.fixture
    def upstream_specs(self):
        s = MxSpecs()
        s["block_size"] = MXFP4.group_size
        s["scale_bits"] = 8
        s["shared_exp_method"] = "max"
        return s

    def upstream_quantize(self, values, specs, round="even"):
        x = torch.tensor(values, dtype=torch.float32)
        return quantize_mx_op(x, specs, elem_format="fp4_e2m1", axes=0, round=round)

    def test_agrees_on_rounding_tie(self, upstream_specs):
        # 2.5 is a tie between E2M1's 2.0 and 3.0; both round half to even.
        values = padded(2.5, 4.0)
        ours = quantize(values)
        upstream = self.upstream_quantize(values, upstream_specs, round="even")
        assert torch.equal(ours, upstream)

    def test_agrees_on_scale_saturation(self, upstream_specs):
        values = padded(2e38)
        ours = quantize(values)
        upstream = self.upstream_quantize(values, upstream_specs, round="even")
        assert torch.equal(ours, upstream)

    def test_agrees_on_denormal_flush(self, upstream_specs):
        values = padded(1e-40)
        ours = quantize(values)
        upstream = self.upstream_quantize(values, upstream_specs, round="even")
        assert torch.equal(ours, upstream)

    def test_agrees_on_random_tensor(self, upstream_specs):
        torch.manual_seed(1)
        values = (torch.randn(MXFP4.group_size) * 2.0).tolist()
        ours = quantize(values)
        upstream = self.upstream_quantize(values, upstream_specs, round="even")
        assert torch.equal(ours, upstream)
