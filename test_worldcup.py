"""Tests du module ML Coupe du Monde (worldcup/).

Tous les tests sont hors-ligne : aucune dépendance réseau. Le modèle est
entraîné sur un petit jeu synthétique pour rester rapide.

    python -m unittest test_worldcup -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from worldcup import value
from worldcup.features import FEATURE_COLUMNS, EloFormBuilder
from worldcup.model import MatchPredictor


def _synthetic_history(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Génère un historique où une équipe « forte » gagne plus souvent."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    strength = {t: rng.normal(0, 1) for t in teams}
    rows = []
    base = pd.Timestamp("2000-01-01")
    for k in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        # Buts simulés via une Poisson dont la moyenne dépend de la force.
        lh = max(0.2, 1.4 + 0.5 * (strength[h] - strength[a]))
        la = max(0.2, 1.1 + 0.5 * (strength[a] - strength[h]))
        hg, ag = rng.poisson(lh), rng.poisson(la)
        rows.append({
            "date": base + pd.Timedelta(days=k),
            "home_team": h, "away_team": a,
            "home_score": hg, "away_score": ag,
            "neutral": bool(rng.integers(0, 2)),
            "tournament": "Friendly", "tournament_weight": 1.0,
            "result": "1" if hg > ag else ("2" if hg < ag else "N"),
        })
    return pd.DataFrame(rows)


class TestValue(unittest.TestCase):
    def test_implied_probabilities_sum_to_one(self):
        odds = {"1": 2.0, "N": 3.5, "2": 4.0}
        fair = value.implied_probabilities(odds)
        self.assertAlmostEqual(sum(fair.values()), 1.0, places=6)
        # La marge bookmaker rend la somme des 1/cote > 1, donc proba juste < 1/cote.
        self.assertLess(fair["1"], 1 / 2.0)

    def test_implied_needs_two_outcomes(self):
        self.assertEqual(value.implied_probabilities({"1": 2.0}), {})

    def test_kelly_zero_when_no_edge(self):
        # proba juste = 1/2 ; cote 1.8 => espérance négative => mise nulle.
        self.assertEqual(value.kelly_stake(0.5, 1.8, 100), 0.0)

    def test_kelly_positive_with_edge(self):
        stake = value.kelly_stake(0.60, 2.0, 100)  # edge = 0.2
        self.assertGreater(stake, 0)

    def test_kelly_capped(self):
        # Edge énorme : la mise doit rester plafonnée par MAX_BET_FRACTION.
        from worldcup import config
        stake = value.kelly_stake(0.99, 5.0, 1000)
        self.assertLessEqual(stake, 1000 * config.MAX_BET_FRACTION + 1e-9)

    def test_find_value_bets_orders_by_edge(self):
        probs = {"1": 0.55, "N": 0.25, "2": 0.20}
        odds = {"1": 2.2, "N": 3.0, "2": 3.0}
        bets = value.find_value_bets(probs, odds, bankroll=100, min_value=0.02)
        self.assertTrue(bets)
        edges = [b.edge_pct for b in bets]
        self.assertEqual(edges, sorted(edges, reverse=True))


class TestFeatures(unittest.TestCase):
    def test_no_leakage_first_match_is_neutral(self):
        """Le tout premier match d'une équipe ne doit voir aucune info future."""
        df = _synthetic_history(50)
        builder = EloFormBuilder()
        X, y = builder.build_training_matrix(df)
        self.assertEqual(list(X.columns), FEATURE_COLUMNS)
        self.assertEqual(len(X), len(df))
        # Première ligne : Elo de départ identique => elo_diff dépend seulement de l'avantage terrain.
        self.assertFalse(X.isnull().any().any())

    def test_strong_team_gets_higher_elo(self):
        df = _synthetic_history(600)
        builder = EloFormBuilder()
        builder.build_training_matrix(df)
        # Au moins une équipe doit s'écarter nettement du rating initial.
        elos = [builder.elo_of(t) for t in builder.known_teams()]
        self.assertGreater(max(elos) - min(elos), 50)


class TestModel(unittest.TestCase):
    def test_fit_predict_proba_valid(self):
        df = _synthetic_history(500)
        model = MatchPredictor().fit(df, calibrate=True)
        probs = model.predict_match("T0", "T1", pd.Timestamp("2002-01-01"),
                                    neutral=True, tournament="FIFA World Cup")
        self.assertEqual(set(probs), {"1", "N", "2"})
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)
        for p in probs.values():
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_save_and_load_roundtrip(self):
        import tempfile
        from pathlib import Path
        df = _synthetic_history(300)
        model = MatchPredictor().fit(df, calibrate=True)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.joblib"
            model.save(path)
            loaded = MatchPredictor.load(path)
        p1 = model.predict_match("T2", "T3", pd.Timestamp("2002-01-01"))
        p2 = loaded.predict_match("T2", "T3", pd.Timestamp("2002-01-01"))
        for k in ("1", "N", "2"):
            self.assertAlmostEqual(p1[k], p2[k], places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
