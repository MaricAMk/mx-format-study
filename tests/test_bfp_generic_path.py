import pytest
import torch
from mx.mx_ops import quantize_mx_op
from mx.specs import MxSpecs

from mx_format_study.formats import (
    E8M0,
    MANTISSA2,
    MANTISSA4,
    MANTISSA8,
    QuantConfig,
    quantize_dequantize,
)

GROUP_SIZES = [8, 16, 32, 64, 128]
BFP_FORMATS = [(MANTISSA8, "int8"), (MANTISSA4, "int4"), (MANTISSA2, "int2")]


def upstream_quantize(
    x: torch.Tensor, group_size: int, elem_format: str, axis: int
) -> torch.Tensor:
    s = MxSpecs()
    s["block_size"] = group_size
    s["scale_bits"] = 8
    s["shared_exp_method"] = "max"
    return quantize_mx_op(x.clone(), s, elem_format=elem_format, axes=axis, round="even")


class TestUpstreamAgreement:
    @pytest.mark.parametrize("element_format,upstream_name", BFP_FORMATS)
    @pytest.mark.parametrize("group_size", GROUP_SIZES)
    def test_random_tensors_last_axis(self, element_format, upstream_name, group_size):
        config = QuantConfig(
            group_size=group_size, scale_format=E8M0, element_format=element_format
        )
        for seed in range(5):
            torch.manual_seed(seed)
            # Per-row log-normal magnitudes so rows land on many different
            # shared exponents, not just the one randn's range picks.
            x = torch.randn(4, group_size) * torch.exp(torch.randn(4, 1) * 3)
            ours = quantize_dequantize(x, config).output
            assert torch.equal(ours, upstream_quantize(x, group_size, upstream_name, axis=-1))

    @pytest.mark.parametrize("element_format,upstream_name", BFP_FORMATS)
    def test_random_tensor_transposed_axis(self, element_format, upstream_name):
        # Grouping along dim 0 must agree too.
        config = QuantConfig(group_size=8, scale_format=E8M0, element_format=element_format, axis=0)
        torch.manual_seed(0)
        x = torch.randn(16, 6)
        ours = quantize_dequantize(x, config).output
        assert torch.equal(ours, upstream_quantize(x, 8, upstream_name, axis=0))

    def test_rounding_ties_go_to_even(self):
        # Block max 4.0 puts BFP4's scale at 1.0, so 2.5 and 1.5 are exact
        # ties; both round to 2.0 under half-to-even (half-away-from-zero
        # would give 3.0 and 2.0).
        config = QuantConfig(group_size=8, scale_format=E8M0, element_format=MANTISSA4)
        x = torch.tensor([4.0, 2.5, 1.5, -2.5, 0.5, 0.0, 0.0, 0.0])
        ours = quantize_dequantize(x, config).output
        assert torch.equal(ours, torch.tensor([4.0, 2.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0]))
        assert torch.equal(ours, upstream_quantize(x, 8, "int4", axis=-1))


class TestPureMantissaGrid:
    @pytest.mark.parametrize("element_format", [MANTISSA8, MANTISSA4, MANTISSA2])
    def test_outputs_are_evenly_spaced_multiples_of_the_scale(self, element_format):
        # BFP: every output is an integer multiple of its group's scale.
        config = QuantConfig(group_size=32, scale_format=E8M0, element_format=element_format)
        torch.manual_seed(0)
        x = torch.randn(8, 32)
        result = quantize_dequantize(x, config)
        max_code = (1 << (element_format.bits - 1)) - 1
        for row, diag in zip(result.output, result.diagnostics):
            codes = row / diag.scale
            assert torch.equal(codes, codes.round())
            assert codes.abs().max() <= max_code

    def test_block_max_saturates_at_top_code(self):
        # 7.9 sits in [4, 8): scale 2**0 for BFP4, and 7.9 rounds to 8, one
        # past the top code -- it must saturate to 7, and be counted clipped.
        config = QuantConfig(group_size=8, scale_format=E8M0, element_format=MANTISSA4)
        x = torch.tensor([7.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = quantize_dequantize(x, config)
        assert result.output[0].item() == 7.0
        assert result.diagnostics[0].clipped == 1


class TestGroupSize:
    @pytest.mark.parametrize("group_size", GROUP_SIZES)
    def test_one_diagnostic_per_group(self, group_size):
        config = QuantConfig(group_size=group_size, scale_format=E8M0, element_format=MANTISSA4)
        x = torch.randn(3, 256)
        result = quantize_dequantize(x, config)
        assert len(result.diagnostics) == 3 * 256 // group_size

    def test_groups_are_independent(self):
        # An outlier in one group must not move any other group's scale.
        config = QuantConfig(group_size=8, scale_format=E8M0, element_format=MANTISSA4)
        torch.manual_seed(0)
        x = torch.randn(32)
        base = quantize_dequantize(x, config).output
        x_outlier = x.clone()
        x_outlier[0] = 1000.0
        with_outlier = quantize_dequantize(x_outlier, config).output
        assert torch.equal(base[8:], with_outlier[8:])

    def test_rejects_indivisible_axis(self):
        config = QuantConfig(group_size=32, scale_format=E8M0, element_format=MANTISSA4)
        with pytest.raises(ValueError):
            quantize_dequantize(torch.randn(48), config)
