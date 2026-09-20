"""Permutation-invariant PyTorch player-value model with six masked task heads."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from player_value.baselines import marginal_logits
from player_value.constants import (
    N_FT_CLASSES,
    N_POINTS_CLASSES,
    N_ZONES,
    UNK_INDEX,
)


@dataclass
class ModelConfig:
    """Hyperparameters for :class:`ThetaNN`.

    ``lambdas`` uses ``field(default_factory=...)`` because a mutable dict must
    never be a shared dataclass default.
    """

    d_embed: int = 16
    hidden: Tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    n_ctx: int = 4
    n_seasons: int = 7
    d_season: int = 4
    lambdas: Dict[str, float] = field(
        default_factory=lambda: {
            "turnover": 1.0,
            "zone": 1.0,
            "make": 1.0,
            "oreb": 1.0,
            "ft": 1.0,
        }
    )
    l2_embed: float = 1e-4
    l2_intercept: float = 1e-4
    l2_trunk: float = 0.0
    temporal_tie: float = 1e-4


@dataclass
class Heads:
    """Per-possession logits emitted by :meth:`ThetaNN.forward`.

    Shapes for a batch of ``B`` possessions: ``points`` [B,5], ``turnover`` [B],
    ``zone`` [B,6], ``make`` [B], ``oreb`` [B], ``ft`` [B,5].
    """

    points: Tensor
    turnover: Tensor
    zone: Tensor
    make: Tensor
    oreb: Tensor
    ft: Tensor


class ThetaNN(nn.Module):
    def __init__(self, vocab_size: int, cfg: ModelConfig, class_counts: Tensor):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size

        # Entity params as Embedding tables (uniform freeze / penalty by row).
        self.embed = nn.Embedding(vocab_size, cfg.d_embed)
        self.off_intercept = nn.Embedding(vocab_size, 1)
        self.def_intercept = nn.Embedding(vocab_size, 1)
        self.season_embed = nn.Embedding(cfg.n_seasons, cfg.d_season)

        # Small init for entity params so the intercept channel + embeds start
        # near zero (marginal-init points head then predicts class marginals).
        with torch.no_grad():
            nn.init.normal_(self.embed.weight, std=0.01)
            self.off_intercept.weight.zero_()
            self.def_intercept.weight.zero_()
            nn.init.normal_(self.season_embed.weight, std=0.01)

        # trunk input = 2*d_embed (pooled off/def) + 1 (intercept channel)
        #               + n_ctx + d_season
        trunk_in = 2 * cfg.d_embed + 1 + cfg.n_ctx + cfg.d_season
        layers = []
        in_dim = trunk_in
        for h in cfg.hidden:
            lin = nn.Linear(in_dim, h)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.constant_(lin.bias, 0.1)
            layers += [lin, nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        trunk_out = in_dim

        # Heads.
        self.points_head = nn.Linear(trunk_out, N_POINTS_CLASSES)
        self.turnover_head = nn.Linear(trunk_out, 1)
        self.zone_head = nn.Linear(trunk_out, N_ZONES)
        self.make_head = nn.Linear(trunk_out, 1)
        self.oreb_head = nn.Linear(trunk_out, 1)
        self.ft_head = nn.Linear(trunk_out, N_FT_CLASSES)
        with torch.no_grad():
            self.points_head.bias.copy_(marginal_logits(class_counts))

        self.freeze_unk()

    # -- core --------------------------------------------------------------

    def _trunk_features(
        self, off_idx: Tensor, def_idx: Tensor, ctx: Tensor, season_idx: Tensor
    ) -> Tensor:
        off_e = self.embed(off_idx).mean(dim=1)   # [B, d_embed]
        def_e = self.embed(def_idx).mean(dim=1)   # [B, d_embed]
        # intercept_channel: Σ off_intercepts − Σ def_intercepts, ONE scalar.
        off_i = self.off_intercept(off_idx).sum(dim=1)   # [B, 1]
        def_i = self.def_intercept(def_idx).sum(dim=1)   # [B, 1]
        intercept_channel = off_i - def_i                # [B, 1]
        season_e = self.season_embed(season_idx)         # [B, d_season]
        return torch.cat([off_e, def_e, intercept_channel, ctx, season_e], dim=1)

    def forward(
        self,
        off_idx: Tensor,
        def_idx: Tensor,
        ctx: Tensor,
        season_idx: Tensor,
    ) -> Heads:
        feats = self._trunk_features(off_idx, def_idx, ctx, season_idx)
        z = self.trunk(feats)
        return Heads(
            points=self.points_head(z),
            turnover=self.turnover_head(z).squeeze(-1),
            zone=self.zone_head(z),
            make=self.make_head(z).squeeze(-1),
            oreb=self.oreb_head(z).squeeze(-1),
            ft=self.ft_head(z),
        )

    # -- PointsModel interface --------------------------------------------

    def points_logits(
        self,
        off_idx: Tensor,
        def_idx: Tensor,
        ctx: Tensor,
        season_idx: Tensor | None = None,
    ) -> Tensor:
        """[B,5] points-class logits.

        ``season_idx=None`` is treated as zeros — a baselines-compat convenience
        so callers that never carry a season (the baselines ignore it entirely)
        can invoke the θ-NN through the identical signature.
        """
        if season_idx is None:
            season_idx = torch.zeros(off_idx.shape[0], dtype=torch.long,
                                     device=off_idx.device)
        return self.forward(off_idx, def_idx, ctx, season_idx).points

    def freeze_unk(self) -> None:
        """Zero row 0 (UNK) of the embed + both intercept tables, in place.

        CONTRACT: the training loop must call this after EVERY optimizer step
        so gradients never let the reserved UNK row carry signal.
        """
        with torch.no_grad():
            self.embed.weight[UNK_INDEX].zero_()
            self.off_intercept.weight[UNK_INDEX].zero_()
            self.def_intercept.weight[UNK_INDEX].zero_()

    # -- regularization ----------------------------------------------------

    def penalties(self, prev: "ThetaNN | None", minutes_w: Tensor) -> Tensor:
        """Minutes-weighted L2 + temporal-tie + trunk-L2 regularizer.

        Terms (all summed into one scalar):

        * ``l2_embed · Σ_i w_i ‖embed_i‖²``
        * ``l2_intercept · Σ_i w_i (off_i² + def_i²)``
        * if ``prev`` is not None: ``temporal_tie · Σ_i w_i (‖embed_i −
          prev.embed_i‖² + (off_i − prev.off_i)² + (def_i − prev.def_i)²)``
          — ``prev`` params are DETACHED (temporal tie shrinks *this* season's
          params toward last season's frozen values, not the reverse).
        * if ``l2_trunk`` != 0: ``l2_trunk · Σ ‖trunk weights‖²``

        ``minutes_w`` is a ``[V]`` per-entity weight (possession-count-derived);
        ``w_i`` MULTIPLIES the penalty for entity ``i``.  This function applies
        the weights exactly as given — any "more minutes → weaker relative
        shrinkage" policy is the CALLER's responsibility in choosing ``w``.  The
        UNK row-0 weight is irrelevant because that row is frozen to zero.
        """
        cfg = self.cfg
        w = minutes_w.to(self.embed.weight.dtype)          # [V]

        embed_sq = (self.embed.weight ** 2).sum(dim=1)      # [V]
        off_sq = (self.off_intercept.weight ** 2).sum(dim=1)  # [V]
        def_sq = (self.def_intercept.weight ** 2).sum(dim=1)  # [V]

        total = cfg.l2_embed * (w * embed_sq).sum()
        total = total + cfg.l2_intercept * (w * (off_sq + def_sq)).sum()

        if prev is not None:
            pe = prev.embed.weight.detach()
            po = prev.off_intercept.weight.detach()
            pd = prev.def_intercept.weight.detach()
            d_embed = ((self.embed.weight - pe) ** 2).sum(dim=1)
            d_off = ((self.off_intercept.weight - po) ** 2).sum(dim=1)
            d_def = ((self.def_intercept.weight - pd) ** 2).sum(dim=1)
            total = total + cfg.temporal_tie * (
                w * (d_embed + d_off + d_def)
            ).sum()

        if cfg.l2_trunk != 0.0:
            trunk_sq = sum(
                (m.weight ** 2).sum()
                for m in self.trunk
                if isinstance(m, nn.Linear)
            )
            total = total + cfg.l2_trunk * trunk_sq

        return total


# -- multi-task loss -------------------------------------------------------


def _weighted_ce(logits: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    """Σ(w·CE)/Σw over the given rows; 0.0 (differentiable) if empty."""
    if target.numel() == 0:
        return (logits.sum() * 0.0)
    per = F.cross_entropy(logits, target, reduction="none")
    return (weight * per).sum() / weight.sum()


def _weighted_bce(logits: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    """Σ(w·BCEWithLogits)/Σw over the given rows; 0.0 (diff) if empty."""
    if target.numel() == 0:
        return (logits.sum() * 0.0)
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (weight * per).sum() / weight.sum()


def compute_loss(
    heads: Heads, batch: Dict[str, Tensor], lambdas: Dict[str, float]
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Masked, weight-normalized multi-task loss.

    All losses use ``reduction="none"`` then manual weight-normalization
    (``Σ w·loss / Σ w``) — NEVER post-softmax losses.  Aux event heads index
    possession-level logits via ``batch["fga_pos"]`` / ``["reb_pos"]`` /
    ``["ft_pos"]``; empty event tensors contribute exactly 0.0 while keeping a
    differentiable path.  ``total = parts["points"] + Σ_h lambdas[h]·parts[h]``.
    """
    weight = batch["weight"]

    # Points: CE over all possessions, per-possession weighted.
    points = _weighted_ce(heads.points, batch["points_class"], weight)

    # Turnover: BCE over all possessions, per-possession weighted.
    turnover = _weighted_bce(heads.turnover, batch["turnover"], weight)

    # Zone: CE on FGA possessions only.
    fga_pos = batch["fga_pos"]
    zone = _weighted_ce(heads.zone[fga_pos], batch["fga_zone"], weight[fga_pos])

    # Make: BCE on FGA possessions only.
    make = _weighted_bce(heads.make[fga_pos], batch["fga_made"], weight[fga_pos])

    # Oreb: BCE on rebound possessions only.
    reb_pos = batch["reb_pos"]
    oreb = _weighted_bce(heads.oreb[reb_pos], batch["reb_oreb"], weight[reb_pos])

    # FT: CE on FT-trip possessions only.
    ft_pos = batch["ft_pos"]
    ft = _weighted_ce(heads.ft[ft_pos], batch["ft_class"], weight[ft_pos])

    parts = {
        "points": points,
        "turnover": turnover,
        "zone": zone,
        "make": make,
        "oreb": oreb,
        "ft": ft,
    }
    total = parts["points"]
    for h in ("turnover", "zone", "make", "oreb", "ft"):
        total = total + lambdas[h] * parts[h]
    return total, parts


def expected_points(points_log_probs: Tensor, v4plus: float) -> Tensor:
    """Σ v(k)·p(k) with class values ``(0, 1, 2, 3, v4plus)``.

    ``points_log_probs`` is ``[B, 5]`` log-probabilities (log-softmax output);
    exponentiating recovers ``p(k)``.
    """
    values = torch.tensor(
        [0.0, 1.0, 2.0, 3.0, float(v4plus)],
        dtype=points_log_probs.dtype,
        device=points_log_probs.device,
    )
    probs = points_log_probs.exp()
    return (probs * values).sum(dim=-1)
