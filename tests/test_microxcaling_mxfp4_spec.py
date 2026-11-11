import pytest
import torch
from mx.mx_ops import quantize_mx_op
from mx.specs import MxSpecs

BLOCK_SIZE = 32  # OCP MX block size
ELEM_FORMAT = "fp4_e2m1"  # E2M1: 1 sign, 2 exponent, 1 mantissa bit
E8M0_MAX_SCALE = 2.0**127  # 8-bit unsigned exponent, no bias headroom beyond this


@pytest.fixture
def specs():
    s = MxSpecs()
    s["block_size"] = BLOCK_SIZE
    s["scale_bits"] = 8
    s["shared_exp_method"] = "max"
    return s


def quantize(values, specs, round="nearest"):
    assert len(values) == BLOCK_SIZE
    x = torch.tensor(values, dtype=torch.float32)
    return quantize_mx_op(x, specs, elem_format=ELEM_FORMAT, axes=0, round=round)


def padded(*head):
    return list(head) + [0.0] * (BLOCK_SIZE - len(head))


class TestDirectedSpecialValues:
    def test_all_zero_block(self, specs):
        """Spec is silent; observed convention: all zero, no NaN."""
        out = quantize([0.0] * BLOCK_SIZE, specs)
        assert torch.all(out == 0.0)

    def test_nan_poisons_entire_block(self, specs):
        """One NaN makes the whole block NaN, through the shared max."""
        out = quantize(padded(float("nan"), 1.0, 2.0), specs)
        assert torch.all(torch.isnan(out))

    def test_inf_poisons_entire_block(self, specs):
        """+Inf doesn't clip to the max: the whole block becomes NaN."""
        out = quantize(padded(float("inf"), 1.0, 2.0), specs)
        assert torch.all(torch.isnan(out))

    def test_scale_saturates_at_e8m0_max(self, specs):
        """2e38 (finite in float32, above 2**127) saturates the scale at
        2**127 instead of overflowing."""
        out = quantize(padded(2e38), specs)
        assert torch.isfinite(out[0])
        assert torch.isclose(out[0], torch.tensor(E8M0_MAX_SCALE), rtol=1e-3)

    def test_extreme_denormal_flushes_to_zero(self, specs):
        """A block of float32 denormals (1e-40) flushes to zero."""
        out = quantize(padded(1e-40), specs)
        assert torch.all(out == 0.0)

    def test_rounding_boundary_nearest_vs_even_vs_floor(self, specs):
        """2.5 is a tie between E2M1's 2.0 and 3.0 (4.0 pins the scale to 1):
        nearest -> 3.0, even -> 2.0, floor -> 2.0."""
        assert quantize(padded(2.5, 4.0), specs, round="nearest")[0] == 3.0
        assert quantize(padded(2.5, 4.0), specs, round="even")[0] == 2.0
        assert quantize(padded(2.5, 4.0), specs, round="floor")[0] == 2.0

    def test_sign_symmetry(self, specs):
        """Negating the input negates the output."""
        pos = quantize(padded(3.0, 1.0, 0.4), specs)
        neg = quantize(padded(-3.0, -1.0, -0.4), specs)
        assert torch.equal(neg, -pos)


class TestProperties:
    def test_round_trip_idempotence(self, specs):
        """Quantizing twice is the same as quantizing once."""
        torch.manual_seed(0)
        x = torch.randn(BLOCK_SIZE) * 3.0
        once = quantize_mx_op(x, specs, elem_format=ELEM_FORMAT, axes=0)
        twice = quantize_mx_op(once, specs, elem_format=ELEM_FORMAT, axes=0)
        assert torch.equal(once, twice)

    def test_monotonicity_within_group(self, specs):
        """a <= b implies Q(a) <= Q(b) within a group."""
        xs = sorted([0.1, 0.4, 0.6, 0.9, 1.2, 1.7, 2.2, 2.8, 3.5, 4.5])
        out = quantize(padded(*xs), specs)
        ordered = out[: len(xs)]
        assert torch.all(ordered[1:] >= ordered[:-1])

    @pytest.mark.parametrize("round_mode", ["nearest", "floor", "even"])
    def test_zero_maps_to_zero_under_every_rounding_mode(self, specs, round_mode):
        out = quantize(padded(0.0, 3.0, 0.0, 1.5, 0.0), specs, round=round_mode)
        assert out[0] == 0.0
        assert out[2] == 0.0
        assert out[4] == 0.0

    def test_scale_sanity_group_max_lands_on_largest_representable(self, specs):
        """A power-of-two group max is reproduced exactly."""
        out = quantize(padded(4.0), specs, round="nearest")
        assert out[0] == 4.0  # exactly representable at any scale choice

    def test_exhaustive_exponent_delta_coverage(self, specs):
        """Sweep all 256 exponent deltas below the block max: the probe stays
        finite, never grows, and eventually flushes to zero."""
        prev = float("inf")
        zero_seen = False
        for delta in range(256):
            probe_value = 2.0 ** (-delta)
            out = quantize(padded(1.0, probe_value), specs)
            probe_out = out[1].item()

            assert torch.isfinite(out[1]), f"delta={delta} produced non-finite output"
            assert probe_out <= prev + 1e-30, f"delta={delta} increased magnitude"
            prev = probe_out
            if probe_out == 0.0:
                zero_seen = True
            if zero_seen:
                assert probe_out == 0.0, f"delta={delta} un-flushed after reaching zero"
