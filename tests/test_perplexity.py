from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mx_format_study.eval.perplexity import (
    last_in_group,
    perplexity,
    perplexity_of,
    position_losses,
)
from mx_format_study.eval.quantize_model import ActivationQuantizer
from mx_format_study.formats import E2M1, E8M0, QuantConfig

VOCAB, DIM, SEQ, K = 50, 32, 32, 8


class ToyCausalLM(nn.Module):
    """Mixes positions only through a causal running mean, so position t
    sees positions <= t, like attention under a causal mask."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.up = nn.Linear(DIM, DIM)
        self.down = nn.Linear(DIM, DIM)
        self.lm_head = nn.Linear(DIM, VOCAB)

    def forward(self, ids):
        h = self.up(self.embed(ids))
        counts = torch.arange(1, ids.shape[1] + 1, dtype=h.dtype).view(1, -1, 1)
        h = h.cumsum(dim=1) / counts
        return SimpleNamespace(logits=self.lm_head(self.down(torch.relu(h))))


TOKEN_GROUPS = QuantConfig(group_size=K, scale_format=E8M0, element_format=E2M1, axis=-2)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ToyCausalLM().eval()


def test_perplexity_is_exp_mean_of_position_losses(model):
    ids = torch.randint(0, VOCAB, (3 * SEQ + 5,))
    losses = position_losses(model, ids, SEQ)
    assert losses.shape == (3, SEQ - 1)
    assert perplexity(model, ids, SEQ) == perplexity_of(losses)
    assert position_losses(model, ids, SEQ, max_windows=2).shape == (2, SEQ - 1)


def test_last_in_group_selects_group_ends():
    losses = torch.arange(2 * (SEQ - 1), dtype=torch.float64).view(2, SEQ - 1)
    picked = last_in_group(losses, K)
    # Positions 7, 15, 23; position 31 is the window's last and has no target.
    assert picked[0].tolist() == [7.0, 15.0, 23.0]
    assert picked[1].tolist() == [SEQ - 1 + 7.0, SEQ - 1 + 15.0, SEQ - 1 + 23.0]


@torch.no_grad()
def test_token_grouping_leaks_only_inside_unfinished_groups(model):
    """The premise of the control: change every token after position t = 19.
    With token-direction groups of 8, the logits at the group ends 7 and 15
    must not change, while some position in the unfinished group 16..23
    (positions 16..19) must, through its group scale."""
    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB, (1, SEQ))
    changed = ids.clone()
    changed[0, 20:] = (ids[0, 20:] + 1) % VOCAB

    with ActivationQuantizer(model, TOKEN_GROUPS):
        a = model(ids).logits
        b = model(changed).logits
    for t in (7, 15):
        assert torch.equal(a[0, t], b[0, t])
    assert any(not torch.equal(a[0, t], b[0, t]) for t in range(16, 20))

    # Unquantized, and with hidden-direction groups, nothing at or before
    # position 19 changes.
    hidden_groups = QuantConfig(group_size=K, scale_format=E8M0, element_format=E2M1)
    for ctx in (ActivationQuantizer(model, hidden_groups), torch.no_grad()):
        with ctx:
            assert torch.equal(model(ids).logits[0, :20], model(changed).logits[0, :20])
