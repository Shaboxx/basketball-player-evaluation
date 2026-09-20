"""Contract, leakage, model-structure and end-to-end regression checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from player_value.data import Transform, fingerprint, validate
from player_value.demo import run
from player_value.evaluation import contribution_scores, paired_bootstrap
from player_value.fixture import generate
from player_value.model import ModelConfig, ThetaNN, compute_loss, expected_points


class Contracts(unittest.TestCase):
    def setUp(self):
        self.rows = generate(games=16, possessions_per_game=12)

    def test_fixture_reconciles_and_is_repeatable(self):
        validate(self.rows)
        self.assertEqual(fingerprint(self.rows), fingerprint(generate(games=16, possessions_per_game=12)))
        self.assertNotEqual(fingerprint(self.rows), fingerprint(generate(seed=18, games=16, possessions_per_game=12)))

    def test_duplicate_and_cross_split_game_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate(self.rows + [self.rows[0]])
        self.rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "cross splits"):
            validate(self.rows)

    def test_invalid_lineup_and_scoring_rejected(self):
        self.rows[0]["off_players"][0] = self.rows[0]["def_players"][0]
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate(self.rows)
        self.rows = generate(games=16, possessions_per_game=12)
        self.rows[0]["points"] += 1
        with self.assertRaisesRegex(ValueError, "reconcile"):
            validate(self.rows)

    def test_holdout_does_not_fit_statistics_or_vocabulary(self):
        original = Transform.fit(self.rows)
        edited = deepcopy(self.rows)
        for row in edited:
            if row["split"] != "train":
                row["start_margin"] = 1000
                row["points"] = 10
                row["off_players"][0] = 999
        new = Transform.fit(edited)
        self.assertTrue(torch.equal(original.mean, new.mean))
        self.assertTrue(torch.equal(original.std, new.std))
        self.assertEqual(original.vocabulary, new.vocabulary)
        self.assertEqual(original.v4plus, new.v4plus)
        held = new.batch([r for r in edited if r["split"] == "test"])
        self.assertTrue((held["off_idx"][:, 0] == 0).all())


class ModelInvariants(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)
        self.rows = generate(games=16, possessions_per_game=12)
        self.transform = Transform.fit(self.rows)
        self.batch = self.transform.batch(self.rows[:12])
        cfg = ModelConfig(d_embed=4, hidden=(12,), dropout=0, n_seasons=2)
        self.model = ThetaNN(25, cfg, torch.ones(5)).eval()

    def test_lineup_permutation_invariance(self):
        b = self.batch
        a = self.model.points_logits(b["off_idx"], b["def_idx"], b["ctx"], b["season_idx"])
        z = self.model.points_logits(b["off_idx"].flip(1), b["def_idx"].flip(1), b["ctx"], b["season_idx"])
        self.assertTrue(torch.allclose(a, z, atol=1e-6))

    def test_intercepts_affect_output_and_unk_is_zero(self):
        b = self.batch
        before = self.model.points_logits(b["off_idx"], b["def_idx"], b["ctx"], b["season_idx"]).detach()
        with torch.no_grad():
            self.model.off_intercept.weight[1:].add_(1)
            self.model.embed.weight[0].fill_(1)
        self.model.freeze_unk()
        after = self.model.points_logits(b["off_idx"], b["def_idx"], b["ctx"], b["season_idx"])
        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(self.model.embed.weight[0].abs().sum().item(), 0)

    def test_empty_events_have_zero_finite_auxiliary_loss(self):
        b = self.batch
        for key in ("fga_pos", "fga_zone", "fga_made", "reb_pos", "reb_oreb", "ft_pos", "ft_class"):
            b[key] = b[key][:0]
        heads = self.model(b["off_idx"], b["def_idx"], b["ctx"], b["season_idx"])
        total, parts = compute_loss(heads, b, self.model.cfg.lambdas)
        self.assertTrue(torch.isfinite(total))
        for head in ("zone", "make", "oreb", "ft"):
            self.assertEqual(parts[head].item(), 0)
        total.backward()

    def test_expected_points_uses_four_plus_value(self):
        logp = torch.tensor([[0, 0, 0, 0, 1.0]]).log()
        self.assertAlmostEqual(expected_points(logp, 4.5).item(), 4.5)

    def test_bootstrap_paired_sign_and_mismatch(self):
        a = {str(i): 0.6 for i in range(8)}
        b = {str(i): 0.7 for i in range(8)}
        result = paired_bootstrap(a, b)
        self.assertTrue(result["passes_lower_nll_gate"])
        self.assertAlmostEqual(result["mean_delta_nll"], -0.1)
        with self.assertRaises(ValueError):
            paired_bootstrap(a, {"other": 0.2})

    def test_replacement_signs_and_appearance_weighted_centering(self):
        class KnownEffect:
            def points_logits(self, off, defense, ctx, season):
                probability = 0.4 + 0.2 * (off == 1).any(1) - 0.15 * (defense == 2).any(1)
                zero = torch.zeros_like(probability)
                return torch.stack([1 - probability, zero, probability, zero, zero], 1).log()

        batch = self.transform.batch(self.rows)
        scores = contribution_scores(KnownEffect(), self.transform, batch, self.rows, min_support=1)
        by_player = {row["player_id"]: row for row in scores}
        self.assertGreater(by_player["P01"]["off_per100"], 30)
        self.assertGreater(by_player["P02"]["def_per100"], 20)
        for side in ("off", "def"):
            centered = sum(row[side + "_per100"] * row[side + "_possessions"] for row in scores)
            self.assertAlmostEqual(centered, 0, places=6)


class Pipeline(unittest.TestCase):
    def test_demo_writes_real_finite_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run(Path(tmp), epochs=4)
            self.assertEqual(report["corpus"], {"games": 64, "possessions": 3072})
            self.assertEqual(report["splits"]["test"]["games"], 12)
            loaded = json.loads((Path(tmp) / "evaluation.json").read_text())
            self.assertEqual(report["fixture_sha256"], loaded["fixture_sha256"])
            self.assertIn("one_hot", report["paired_comparisons"])
            self.assertEqual(len((Path(tmp) / "player_scores.csv").read_text().splitlines()), 25)
            self.assertTrue((Path(tmp) / "evaluation.md").is_file())


if __name__ == "__main__":
    unittest.main()
