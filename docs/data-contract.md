# Normalized possession contract

The public boundary is one JSON object per possession. `player_value.data.validate` rejects duplicates, split leakage, inconsistent scoring, invalid lineups, nonfinite context, and missing splits. The fixture generator creates this exact contract. Use `load_jsonl(Path(...))` to validate a generated file.

| Field | Type / rule | Role |
|---|---|---|
| `game_id`, `possession_id` | String and nonnegative integer; pair is unique | Grouping and deduplication |
| `season` | String; constant within game | Player-season and season vocabularies |
| `split` | `train`, `validation`, or `test`; constant within game | Explicit whole-game split |
| `off_players`, `def_players` | Five distinct positive integer IDs each; no shared IDs | The ten players at possession start |
| `home_off` | Integer 0 or 1 | Pre-possession context |
| `start_margin` | Finite number | Pre-possession score margin |
| `period` | Integer 1–4 | Regulation period; overtime rejected |
| `start_seconds_remaining` | Integer 0–720 | Clock at possession start |
| `points` | Integer 0–10 | Raw scoring outcome, capped at class 4 for classification |
| `turnover` | Integer 0 or 1 | Auxiliary label |
| `fga` | `null` or `{zone: 0..5, made: 0 or 1}` | At most one shot event; zones 0–2 score 2 and 3–5 score 3 |
| `ft_points` | `null` or integer 0–10 | At most one free-throw trip, capped at class 4 |
| `offensive_rebound` | `null` or integer 0 or 1 | Rebound outcome if observed |

The released event contract is deliberately smaller than a full play-by-play schema. It represents one shot attempt and one free-throw trip per possession; a real ingestion adapter would need event lists and reliable possession/lineup reconstruction. The validator checks event scoring against raw possession points. It does not claim to resolve all basketball event-sequence semantics.

Transformations preserve a fixed context order: home offense, `int(clip(start_margin, -10, 10) / 5)`, raw period, and `start_seconds_remaining // 180`. Means and population standard deviations are fitted on training rows only. Standard deviations are floored at `1e-6`.

Player-season vocabulary entries come from training lineups only. Unknown entities map to index zero, whose model parameters are kept zero. The season vocabulary is also training-only, with zero reserved for unknown. The demo has one invented season; cross-season forecasting and unknown-season predictions are not validated by it.

Raw points are capped into five classes: 0, 1, 2, 3, and 4+. Expected points use the training-only mean of raw outcomes of at least four points for the last class, or 4.0 if training contains none. This is recorded as `v4plus` in every evaluation report.

The generator uses 64 games and 48 possessions per game. Games 0–39 train the model (1,920 rows), 40–51 select the checkpoint (576 rows), and 52–63 evaluate it (576 rows). These are game-grouped partitions of an invented single-season process, not forward NBA seasons. No outcome label is fed back as an input feature.
