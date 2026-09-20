"""Validate normalized possessions and fit transformations on training games only."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import torch

CONTEXT = ("home_off", "margin_bucket", "period", "clock_band")
SPLITS = ("train", "validation", "test")


def validate(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("At least one possession is required")
    seen, game_splits, game_seasons = set(), {}, {}
    for r in rows:
        key = (r["game_id"], r["possession_id"])
        if key in seen:
            raise ValueError(f"Duplicate possession: {key}")
        seen.add(key)
        if not isinstance(r["game_id"], str) or not isinstance(r["season"], str):
            raise ValueError("game_id and season must be strings")
        if type(r["possession_id"]) is not int or r["possession_id"] < 0:
            raise ValueError("possession_id must be a nonnegative integer")
        if r["split"] not in SPLITS:
            raise ValueError("Unknown split")
        prior = game_splits.setdefault(r["game_id"], r["split"])
        if prior != r["split"]:
            raise ValueError("A game cannot cross splits")
        if game_seasons.setdefault(r["game_id"], r["season"]) != r["season"]:
            raise ValueError("A game cannot cross seasons")
        lineup = r["off_players"] + r["def_players"]
        if (len(r["off_players"]) != 5 or len(r["def_players"]) != 5
                or len(set(lineup)) != 10 or any(type(p) is not int or p < 1 for p in lineup)):
            raise ValueError("Exactly ten distinct positive integer player IDs required")
        if type(r["points"]) is not int or not 0 <= r["points"] <= 10:
            raise ValueError("points must be an integer from zero to ten")
        if type(r["period"]) is not int or r["period"] not in (1, 2, 3, 4):
            raise ValueError("Only regulation periods are supported")
        if type(r["home_off"]) is not int or r["home_off"] not in (0, 1):
            raise ValueError("home_off must be zero or one")
        if (type(r["start_seconds_remaining"]) is not int
                or not 0 <= r["start_seconds_remaining"] <= 720):
            raise ValueError("Invalid regulation clock")
        if (not isinstance(r["start_margin"], (int, float))
                or not math.isfinite(r["start_margin"])):
            raise ValueError("start_margin must be finite")
        if type(r["turnover"]) is not int or r["turnover"] not in (0, 1):
            raise ValueError("turnover must be zero or one")
        fga = r["fga"]
        if fga is not None and (type(fga["zone"]) is not int or fga["zone"] not in range(6)
                                or type(fga["made"]) is not int or fga["made"] not in (0, 1)):
            raise ValueError("Invalid field-goal event")
        ft = r["ft_points"]
        if ft is not None and (type(ft) is not int or not 0 <= ft <= 10):
            raise ValueError("Invalid free-throw trip")
        reb = r["offensive_rebound"]
        if reb is not None and (type(reb) is not int or reb not in (0, 1)):
            raise ValueError("Invalid rebound outcome")
        scored = (0 if fga is None else fga["made"] * (3 if fga["zone"] >= 3 else 2)) + (ft or 0)
        if scored != r["points"]:
            raise ValueError("Event scoring does not reconcile with possession points")
    if set(game_splits.values()) != set(SPLITS):
        raise ValueError("Nonempty train, validation, and test games are required")


def canonical_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)


def fingerprint(rows: list[dict]) -> str:
    return hashlib.sha256(canonical_jsonl(rows).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate(rows)
    return rows


def raw_context(r: dict) -> list[float]:
    return [r["home_off"], int(max(-10, min(10, r["start_margin"])) / 5),
            r["period"], r["start_seconds_remaining"] // 180]


@dataclass
class Transform:
    vocabulary: dict[tuple[int, str], int]
    seasons: dict[str, int]
    mean: torch.Tensor
    std: torch.Tensor
    v4plus: float

    @classmethod
    def fit(cls, rows: list[dict]):
        train = [r for r in rows if r["split"] == "train"]
        if not train:
            raise ValueError("Cannot fit without training rows")
        entities = sorted({(p, r["season"]) for r in train for p in r["off_players"] + r["def_players"]})
        context = torch.tensor([raw_context(r) for r in train], dtype=torch.float32)
        upper = [r["points"] for r in train if r["points"] >= 4]
        return cls({p: i + 1 for i, p in enumerate(entities)},
                   {s: i + 1 for i, s in enumerate(sorted({r["season"] for r in train}))},
                   context.mean(0), context.std(0, correction=0).clamp_min(1e-6),
                   sum(upper) / len(upper) if upper else 4.0)

    def batch(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        def lineup(name):
            return torch.tensor([[self.vocabulary.get((p, r["season"]), 0) for p in r[name]] for r in rows])
        b = {
            "off_idx": lineup("off_players"), "def_idx": lineup("def_players"),
            "ctx": (torch.tensor([raw_context(r) for r in rows], dtype=torch.float32) - self.mean) / self.std,
            "season_idx": torch.tensor([self.seasons.get(r["season"], 0) for r in rows]),
            "points_class": torch.tensor([min(4, r["points"]) for r in rows]),
            "turnover": torch.tensor([r["turnover"] for r in rows], dtype=torch.float32),
            "weight": torch.ones(len(rows)),
        }
        for prefix, key in (("fga", "fga"), ("reb", "offensive_rebound"), ("ft", "ft_points")):
            b[prefix + "_pos"] = torch.tensor([i for i, r in enumerate(rows) if r[key] is not None], dtype=torch.long)
        b["fga_zone"] = torch.tensor([r["fga"]["zone"] for r in rows if r["fga"] is not None], dtype=torch.long)
        b["fga_made"] = torch.tensor([r["fga"]["made"] for r in rows if r["fga"] is not None], dtype=torch.float32)
        b["reb_oreb"] = torch.tensor([r["offensive_rebound"] for r in rows if r["offensive_rebound"] is not None], dtype=torch.float32)
        b["ft_class"] = torch.tensor([min(4, r["ft_points"]) for r in rows if r["ft_points"] is not None], dtype=torch.long)
        return b
