import torch
from torch import nn

from mx_format_study.eval.quantize_model import (
    ActivationQuantizer,
    quantize_weights,
    target_linears,
)
from mx_format_study.formats import (
    E2M1,
    E8M0,
    QuantConfig,
    quantize_dequantize_fast,
    quantize_dequantize_prefix_fast,
)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
        self.lm_head = nn.Linear(32, 16)

    def forward(self, x):
        return self.lm_head(self.block(x))


CONFIG = QuantConfig(group_size=16, scale_format=E8M0, element_format=E2M1)


def test_lm_head_is_not_a_target():
    assert set(target_linears(Toy())) == {"block.0", "block.2"}


def test_quantize_weights_matches_fast_path_and_skips_lm_head():
    torch.manual_seed(0)
    model = Toy()
    before = {n: p.clone() for n, p in model.named_parameters()}
    stats = quantize_weights(model, CONFIG)

    for name in ("block.0", "block.2"):
        expected = quantize_dequantize_fast(before[f"{name}.weight"], CONFIG).output
        assert torch.equal(dict(model.named_parameters())[f"{name}.weight"], expected)
    assert torch.equal(model.lm_head.weight, before["lm_head.weight"])
    assert stats.elements == 32 * 64 + 64 * 32
    assert 0.0 < stats.relative_mse < 1.0


def test_grouping_axis_changes_weights():
    torch.manual_seed(0)
    a, b = Toy(), Toy()
    b.load_state_dict(a.state_dict())
    quantize_weights(a, CONFIG)
    quantize_weights(b, QuantConfig(**{**CONFIG.__dict__, "axis": 0}))
    assert not torch.equal(a.block[0].weight, b.block[0].weight)


def test_activation_hooks_quantize_inputs_and_are_removed():
    torch.manual_seed(0)
    model = Toy()
    x = torch.randn(2, 8, 32)
    clean = model(x)

    captured = {}
    h = model.block[0].register_forward_hook(lambda m, i, o: captured.setdefault("out", o))
    with ActivationQuantizer(model, CONFIG):
        assert not torch.equal(model(x), clean)
    h.remove()

    # The first linear saw exactly the fast path's quantized input.
    qx = quantize_dequantize_fast(x, CONFIG).output
    torch.testing.assert_close(captured["out"], model.block[0](qx), rtol=0, atol=0)

    assert torch.equal(model(x), clean)


def test_causal_activation_hooks_use_prefix_quantizer():
    torch.manual_seed(0)
    model = Toy()
    x = torch.randn(1, 16, 32)
    config = QuantConfig(group_size=8, scale_format=E8M0, element_format=E2M1, axis=-2)

    captured = {}
    h = model.block[0].register_forward_hook(lambda m, i, o: captured.setdefault("out", o))
    with ActivationQuantizer(model, config, causal=True):
        model(x)
    h.remove()

    qx = quantize_dequantize_prefix_fast(x, config)
    torch.testing.assert_close(captured["out"], model.block[0](qx), rtol=0, atol=0)
