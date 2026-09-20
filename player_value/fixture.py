"""Invented players, lineups and outcomes. No external sports data is used."""
import math
import random


def generate(seed: int = 17, games: int = 64, possessions_per_game: int = 48) -> list[dict]:
    if games < 10 or possessions_per_game < 1:
        raise ValueError("Fixture requires at least ten games and one possession per game")
    rng = random.Random(seed)
    off_skill = {p: rng.gauss(0, 0.55) for p in range(1, 25)}
    def_skill = {p: rng.gauss(0, 0.45) for p in range(1, 25)}
    rows = []
    train_end, val_end = int(games * 0.625), int(games * 0.8125)
    for game in range(games):
        split = "train" if game < train_end else "validation" if game < val_end else "test"
        for pos in range(possessions_per_game):
            lineup = rng.sample(range(1, 25), 10)
            offense, defense = lineup[:5], lineup[5:]
            home = pos % 2
            signal = (sum(off_skill[p] for p in offense) - sum(def_skill[p] for p in defense)) / 2
            signal += 0.15 * (home - 0.5)
            logits = [0.8 - signal, -1.6, 0.4 + signal, -0.7 + 0.65 * signal, -3.5 + signal]
            points = rng.choices(range(5), weights=[math.exp(x) for x in logits])[0]
            turnover = int(points == 0 and rng.random() < 0.28)
            ft, reb, fga = None, None, None
            if points == 0 and not turnover:
                fga = {"zone": rng.randrange(6), "made": 0}
                reb = 0  # A single missed attempt ends this simplified possession.
            elif points == 1:
                ft = 1
            elif points in (2, 3):
                fga = {"zone": rng.randrange(3) + (3 if points == 3 else 0), "made": 1}
            elif points == 4:
                fga, ft = {"zone": rng.randrange(3, 6), "made": 1}, 1
            rows.append({"game_id": f"SYN-{game:03d}", "possession_id": pos, "season": "synthetic-1",
                         "split": split, "off_players": offense, "def_players": defense,
                         "home_off": home, "start_margin": rng.randint(-18, 18),
                         "period": 1 + pos % 4, "start_seconds_remaining": rng.randrange(721),
                         "points": points, "turnover": turnover, "fga": fga,
                         "ft_points": ft, "offensive_rebound": reb})
    return rows
