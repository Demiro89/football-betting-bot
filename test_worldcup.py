"""Tests du module ML Coupe du Monde (worldcup/).

Tous les tests sont hors-ligne : aucune dépendance réseau. Le modèle est
entraîné sur un petit jeu synthétique pour rester rapide.

    python -m unittest test_worldcup -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from worldcup import live_odds, tickets, tracking, value
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

    def test_predict_with_tz_aware_date(self):
        """Les cotes en direct fournissent des dates avec fuseau (tz-aware) :
        la prédiction ne doit pas planter (régression mode live)."""
        df = _synthetic_history(300)
        model = MatchPredictor().fit(df, calibrate=True)
        d = pd.Timestamp("2002-06-14T18:00:00+00:00")  # tz-aware (UTC)
        probs = model.predict_match("T0", "T1", d, neutral=True)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)

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


def _fake_event() -> dict:
    """Événement The Odds API simulé avec 2 books aux cotes différentes."""
    return {
        "home_team": "Brazil", "away_team": "Serbia", "_sport_key": "soccer_fifa_world_cup",
        "commence_time": "2026-06-14T18:00:00Z",
        "bookmakers": [
            {"key": "pinnacle", "title": "Pinnacle", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Brazil", "price": 1.50}, {"name": "Draw", "price": 4.20},
                    {"name": "Serbia", "price": 6.50}]}]},
            {"key": "softbook", "title": "SoftBook", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Brazil", "price": 1.55}, {"name": "Draw", "price": 4.50},
                    {"name": "Serbia", "price": 7.00}]}]},
        ],
    }


class TestLiveOdds(unittest.TestCase):
    def test_best_odds_line_shopping(self):
        best = live_odds.best_odds(_fake_event())
        # On doit retenir la meilleure cote de chaque issue (SoftBook ici).
        self.assertEqual(best["1"]["odds"], 1.55)
        self.assertEqual(best["2"]["odds"], 7.00)
        self.assertEqual(best["1"]["book"], "SoftBook")

    def test_consensus_sums_to_one(self):
        cons = live_odds.consensus_probabilities(_fake_event())
        self.assertAlmostEqual(sum(cons.values()), 1.0, places=6)
        self.assertGreater(cons["1"], cons["2"])  # Brésil favori

    def test_resolve_team_alias_and_fuzzy(self):
        known = ["United States", "Korea Republic", "Brazil", "Germany"]
        self.assertEqual(live_odds.resolve_team("USA", known), "United States")
        self.assertEqual(live_odds.resolve_team("South Korea", known), "Korea Republic")
        self.assertEqual(live_odds.resolve_team("Brazil", known), "Brazil")

    def test_event_to_fixture(self):
        row = live_odds.event_to_fixture(_fake_event(), ["Brazil", "Serbia"])
        self.assertEqual(row["home_team"], "Brazil")
        self.assertTrue(row["neutral"])           # match de Coupe du Monde
        self.assertEqual(row["odd_2"], 7.00)
        self.assertEqual(row["n_books"], 2)


class TestEnsemble(unittest.TestCase):
    def test_blend_renormalizes(self):
        model = {"1": 0.6, "N": 0.25, "2": 0.15}
        cons = {"1": 0.4, "N": 0.3, "2": 0.3}
        blended = value.ensemble_probabilities(model, cons, market_weight=0.5)
        self.assertAlmostEqual(sum(blended.values()), 1.0, places=6)
        self.assertTrue(0.4 < blended["1"] < 0.6)

    def test_blend_without_consensus_returns_model(self):
        model = {"1": 0.6, "N": 0.25, "2": 0.15}
        self.assertEqual(value.ensemble_probabilities(model, None), model)


class TestTickets(unittest.TestCase):
    def _leg(self, match, sel="1", odds=2.0, mp=0.60, kp=0.50):
        return tickets.TicketLeg(match=match, selection=sel, label="Victoire domicile",
                                 odds=odds, model_prob=mp, market_prob=kp)

    def test_build_ticket_multiplies_odds_and_probs(self):
        legs = [self._leg("A vs B", odds=2.0, mp=0.6, kp=0.5),
                self._leg("C vs D", odds=3.0, mp=0.5, kp=0.4)]
        t = tickets.build_ticket(legs, bankroll=100)
        self.assertAlmostEqual(t.combined_odds, 6.0, places=6)
        self.assertAlmostEqual(t.model_prob, 0.30, places=6)   # 0.6 * 0.5
        self.assertEqual(t.kind, "combiné")
        # edge = 0.30 * 6 - 1 = 0.8 -> +80%
        self.assertAlmostEqual(t.edge_pct, 80.0, places=1)

    def test_single_ticket_kind(self):
        t = tickets.build_ticket([self._leg("A vs B")], bankroll=100)
        self.assertEqual(t.kind, "simple")
        self.assertGreaterEqual(t.potential_return, 0.0)

    def test_propose_avoids_same_match_in_combo(self):
        # Deux jambes value sur le MÊME match ne doivent pas être combinées.
        legs = [self._leg("A vs B", sel="1", odds=2.0, mp=0.6, kp=0.5),
                self._leg("A vs B", sel="2", odds=2.5, mp=0.5, kp=0.45)]
        out = tickets.propose_tickets(legs, bankroll=100)
        self.assertEqual(len(out["singles"]), 2)
        self.assertIsNone(out["combo"])  # une seule jambe par match -> pas de combiné

    def test_propose_builds_combo_from_distinct_matches(self):
        legs = [self._leg("A vs B", odds=2.0, mp=0.6, kp=0.5),
                self._leg("C vs D", odds=2.0, mp=0.6, kp=0.5)]
        out = tickets.propose_tickets(legs, bankroll=100)
        self.assertIsNotNone(out["combo"])
        self.assertEqual(len(out["combo"].legs), 2)


class TestTracking(unittest.TestCase):
    def test_clv_positive_when_beating_close(self):
        # Cote prise 2.10, clôture 2.00 => on a pris mieux => CLV > 0.
        self.assertGreater(tracking.clv_percent(2.10, 2.00), 0)
        self.assertLess(tracking.clv_percent(1.90, 2.00), 0)

    def test_log_ticket_combo(self):
        import tempfile
        from pathlib import Path
        from worldcup import config
        original = config.BETS_LOG
        legs = [tickets.TicketLeg("A vs B", "1", "Victoire domicile", 2.0, 0.6, 0.5),
                tickets.TicketLeg("C vs D", "2", "Victoire extérieur", 3.0, 0.5, 0.4)]
        ticket = tickets.build_ticket(legs, bankroll=100)
        with tempfile.TemporaryDirectory() as d:
            config.BETS_LOG = Path(d) / "bets.csv"
            try:
                tracking.log_ticket(ticket)
                rows = tracking.load_bets()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["selection"], "combiné x2")
                self.assertAlmostEqual(float(rows[0]["odds_taken"]), 6.0, places=2)
                self.assertIn("A vs B", rows[0]["match"])
            finally:
                config.BETS_LOG = original

    def test_log_and_summary_roundtrip(self):
        import tempfile
        from pathlib import Path
        from worldcup import config
        original = config.BETS_LOG
        with tempfile.TemporaryDirectory() as d:
            config.BETS_LOG = Path(d) / "bets.csv"
            try:
                tracking.log_bet("Brazil vs Serbia", "Victoire domicile", 1.55,
                                 0.70, 0.66, 5.0, 5.0)
                bets = tracking.load_bets()
                self.assertEqual(len(bets), 1)
                self.assertEqual(bets[0]["selection"], "Victoire domicile")
                s = tracking.summary()
                self.assertEqual(s["n_total"], 1)
                self.assertEqual(s["n_settled"], 0)
            finally:
                config.BETS_LOG = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
