"""CPU training, game-level holdout comparison, and reference replacement scores."""
from copy import deepcopy
import math
import random
import torch
import torch.nn.functional as F
from .model import ThetaNN, compute_loss, expected_points


def logits(model, batch):
    return model.points_logits(batch["off_idx"], batch["def_idx"], batch["ctx"], batch["season_idx"])


def fit(model, train, validation, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    best_nll, best_epoch, best_state, history = math.inf, 0, None, []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if isinstance(model, ThetaNN):
            heads = model(train["off_idx"], train["def_idx"], train["ctx"], train["season_idx"])
            loss, _ = compute_loss(heads, train, model.cfg.lambdas)
            loss = loss + model.penalties(None, torch.ones(model.vocab_size))
        else:
            loss = F.cross_entropy(logits(model, train), train["points_class"])
        loss.backward()
        optimizer.step()
        model.freeze_unk()
        model.eval()
        with torch.no_grad():
            val_nll = F.cross_entropy(logits(model, validation), validation["points_class"]).item()
        history.append({"epoch": epoch + 1, "train_objective": loss.item(), "validation_nll": val_nll})
        if val_nll < best_nll:
            best_nll, best_epoch = val_nll, epoch + 1
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return {"selected_epoch": best_epoch, "validation_nll": best_nll, "history": history}


@torch.no_grad()
def evaluate(model, batch, rows, v4plus):
    prediction = logits(model, batch)
    per_row = F.cross_entropy(prediction, batch["points_class"], reduction="none").tolist()
    by_game = {}
    for r, loss in zip(rows, per_row):
        by_game.setdefault(r["game_id"], []).append(loss)
    means = {g: sum(losses) / len(losses) for g, losses in by_game.items()}
    ep = expected_points(prediction.log_softmax(-1), v4plus)
    truth = torch.tensor([r["points"] for r in rows], dtype=torch.float32)
    return {"possession_nll": sum(per_row) / len(per_row),
            "game_mean_nll": sum(means.values()) / len(means),
            "expected_points_mse": ((ep - truth) ** 2).mean().item(),
            "game_nll": means}


def paired_bootstrap(model_losses, baseline_losses, seed=91, replicates=2000):
    games = sorted(model_losses)
    if set(games) != set(baseline_losses) or len(games) < 2:
        raise ValueError("Paired bootstrap requires the same two or more games")
    deltas = [model_losses[g] - baseline_losses[g] for g in games]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(replicates))
    low, high = means[int(0.05 * replicates)], means[int(0.95 * replicates) - 1]
    return {"mean_delta_nll": sum(deltas) / len(deltas), "ci90_low": low, "ci90_high": high,
            "games": len(games), "bootstrap_replicates": replicates,
            "passes_lower_nll_gate": high < 0}


@torch.no_grad()
def contribution_scores(model, transform, batch, rows, min_support=50):
    """Replace a focal player with eligible peers in their observed holdout contexts.

    Offense: actual expected points minus replacement expected points.
    Defense: replacement expected points minus actual expected points.
    Positive is favorable on both sides. Estimates are descriptive, not causal.
    """
    scores = []
    for (player, season), entity in transform.vocabulary.items():
        peers = [e for (p, s), e in transform.vocabulary.items() if s == season and p != player]
        values = {}
        for side in ("off", "def"):
            matrix = batch[side + "_idx"]
            selected = (matrix == entity).any(1).nonzero().flatten()
            deltas = []
            for index in selected.tolist():
                off, defense = batch["off_idx"][index:index + 1], batch["def_idx"][index:index + 1]
                occupied = set(off[0].tolist() + defense[0].tolist())
                alternatives = [p for p in peers if p not in occupied]
                if not alternatives:
                    continue
                ctx, sea = batch["ctx"][index:index + 1], batch["season_idx"][index:index + 1]
                actual = expected_points(model.points_logits(off, defense, ctx, sea).log_softmax(-1), transform.v4plus)[0]
                off_new, def_new = off.repeat(len(alternatives), 1), defense.repeat(len(alternatives), 1)
                replacement = off_new if side == "off" else def_new
                slot = (matrix[index] == entity).nonzero().item()
                replacement[:, slot] = torch.tensor(alternatives)
                counterfactual = expected_points(model.points_logits(off_new, def_new, ctx.repeat(len(alternatives), 1),
                                               sea.repeat(len(alternatives))).log_softmax(-1), transform.v4plus).mean()
                delta = actual - counterfactual if side == "off" else counterfactual - actual
                deltas.append(delta.item() * 100)
            values[side + "_per100"] = sum(deltas) / len(deltas) if deltas else None
            values[side + "_possessions"] = len(deltas)
        supported = min(values["off_possessions"], values["def_possessions"])
        scores.append({"player_id": f"P{player:02d}", "season": season, **values,
                       "support": "no_support" if supported == 0 else "low_sample" if supported < min_support else "ok"})
    for side in ("off", "def"):
        count = sum(r[side + "_possessions"] for r in scores)
        center = sum((r[side + "_per100"] or 0) * r[side + "_possessions"] for r in scores) / max(1, count)
        for row in scores:
            if row[side + "_per100"] is not None:
                row[side + "_per100"] -= center
    for row in scores:
        row["total_per100"] = (row["off_per100"] + row["def_per100"]
                               if row["off_per100"] is not None and row["def_per100"] is not None else None)
    return sorted(scores, key=lambda r: (r["total_per100"] is not None, r["total_per100"] or 0), reverse=True)
