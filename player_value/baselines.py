"""Context-only and additive one-hot points baselines with a shared model interface."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from player_value.constants import N_POINTS_CLASSES, UNK_INDEX


def marginal_logits(counts: Tensor) -> Tensor:
    """Logits whose softmax equals the empirical class frequencies.

    ``softmax(log(p)) == p`` for any probability vector ``p``, so returning
    ``log(counts / counts.sum())`` recovers the marginal distribution exactly.
    A tiny floor guards against ``log(0)`` for empty classes.
    """
    counts = counts.to(torch.float32)
    total = counts.sum()
    probs = counts / total
    return torch.log(probs.clamp_min(torch.finfo(torch.float32).tiny))


class ContextNull(nn.Module):
    """Linear on standardized ctx -> 5 points-class logits; ignores players.

    The bias of the linear layer is initialized from the train-marginal class
    counts (via :func:`marginal_logits`) so that at init, on zero-ctx, the model
    predicts the class marginals.
    """

    def __init__(self, n_ctx: int, class_counts: Tensor):
        super().__init__()
        self.lin = nn.Linear(n_ctx, N_POINTS_CLASSES)
        with torch.no_grad():
            self.lin.bias.copy_(marginal_logits(class_counts))

    def points_logits(
        self,
        off_idx: Tensor,
        def_idx: Tensor,
        ctx: Tensor,
        season_idx: Tensor | None = None,
    ) -> Tensor:
        # off_idx / def_idx / season_idx are ignored by construction.
        return self.lin(ctx)

    def freeze_unk(self) -> None:
        # No entity table -> nothing to zero. Provided for interface uniformity.
        return None


class OneHotPoints(nn.Module):
    """±1 player one-hots + linear ctx -> 5 points-class logits.

    Each player-season vocab entity has a 5-dim row in an ``EmbeddingBag``
    (``mode="sum"``).  The five offensive entities' rows are summed and ADDED;
    the five defensive entities' rows are summed and SUBTRACTED — the sign
    convention encoding "offense creates points, defense suppresses them".  A
    linear layer on standardized ctx and a per-class ``bias`` (initialized from
    the train marginals) complete the logits.
    """

    def __init__(self, vocab_size: int, n_ctx: int, class_counts: Tensor):
        super().__init__()
        self.table = nn.EmbeddingBag(vocab_size, N_POINTS_CLASSES, mode="sum")
        self.ctx = nn.Linear(n_ctx, N_POINTS_CLASSES)
        self.bias = nn.Parameter(marginal_logits(class_counts).clone())
        # ctx bias is redundant with self.bias; the shared logit offset lives in
        # self.bias (marginal init). Zero the linear's own bias to keep the two
        # from double-counting at init.
        with torch.no_grad():
            self.ctx.bias.zero_()
        self.freeze_unk()

    def points_logits(
        self,
        off_idx: Tensor,
        def_idx: Tensor,
        ctx: Tensor,
        season_idx: Tensor | None = None,
    ) -> Tensor:
        # season_idx ignored. EmbeddingBag over the 5 entities per row (dim=1).
        off = self.table(off_idx)   # [B, 5] summed offensive rows
        deff = self.table(def_idx)  # [B, 5] summed defensive rows
        return off - deff + self.ctx(ctx) + self.bias

    def freeze_unk(self) -> None:
        """Zero the reserved UNK entity row (index 0) in place."""
        with torch.no_grad():
            self.table.weight[UNK_INDEX].zero_()
