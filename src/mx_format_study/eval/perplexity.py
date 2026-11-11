from __future__ import annotations

import math

import torch
from torch import nn


def tokenize(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt").input_ids[0]


@torch.no_grad()
def position_losses(
    model: nn.Module, token_ids: torch.Tensor, seq_len: int, max_windows: int | None = None
) -> torch.Tensor:
    """NLL of every predicted token, [n_windows, seq_len - 1], float64.
    Column t is the loss of the prediction made at position t."""
    n_windows = token_ids.numel() // seq_len
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    device = next(model.parameters()).device
    rows = []
    for i in range(n_windows):
        window = token_ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)
        logits = model(window).logits.float()
        nll = nn.functional.cross_entropy(logits[0, :-1], window[0, 1:], reduction="none")
        rows.append(nll.double().cpu())
    return torch.stack(rows)


def perplexity_of(losses: torch.Tensor) -> float:
    return math.exp(float(losses.mean()))


def last_in_group(losses: torch.Tensor, group_size: int) -> torch.Tensor:
    """Losses at positions k - 1, 2k - 1, ...: the last token of each token
    group, whose prediction can't depend on later tokens even with
    token-direction grouping."""
    return losses[:, group_size - 1 :: group_size]


def perplexity(
    model: nn.Module, token_ids: torch.Tensor, seq_len: int, max_windows: int | None = None
) -> float:
    return perplexity_of(position_losses(model, token_ids, seq_len, max_windows))
