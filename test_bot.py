"""Suite de tests automatisée du bot de value betting.

Lancement : python -m unittest discover -v
Toutes les fonctions réseau sont simulées (mock) — aucun appel API réel.
"""

import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import requests

import backtest as bt
import main
import roi


# =============================================================================
# KELLY
# =============================================================================
class TestKelly(unittest.TestCase):
    def test_edge_positif_donne_mise(self):
        # prob 0.6 @ 2.0 : edge 0.2, quart de Kelly plafonné à 5 % => 50 €
        self.assertEqual(main.kelly_stake(0.6, 2.0, 1000), 50.0)

    def test_edge_negatif_donne_zero(self):
        self.assertEqual(main.kelly_stake(0.3, 2.0, 1000), 0.0)

    def test_cote_unite_donne_zero(self):
        self.assertEqual(main.kelly_stake(0.9, 1.0, 1000), 0.0)

    def test_mise_sous_minimum_donne_zero(self):
        # edge minuscule + petit bankroll => mise < MIN_BET => écartée
        self.assertEqual(main.kelly_stake(0.51, 2.0, 10), 0.0)

    def test_plafond_respecte(self):
        stake = main.kelly_stake(0.95, 3.0, 1000)
        self.assertLessEqual(stake, 1000 * main.MAX_BET_FRACTION)


# =============================================================================
# MODÈLE
# =============================================================================
class TestModel(unittest.TestCase):
    def test_score_matrix_normalisee(self):
        grid = main.score_matrix(1.5, 1.2)
        total = sum(p for row in grid for p in row)
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertTrue(all(p >= 0 for row in grid for p in row))

    def test_1x2_somme_un(self):
        m = main.market_probabilities(1.7, 1.1)
        self.assertAlmostEqual(m["1"] + m["N"] + m["2"], 1.0, places=6)

    def test_over_under_complementaires(self):
        m = main.market_probabilities(2.0, 1.5)
        self.assertAlmostEqual(m["O2.5"] + m["U2.5"], 1.0, places=6)

    def test_btts_complementaires(self):
        m = main.market_probabilities(1.4, 1.3)
        self.assertAlmostEqual(m["BTTS"] + m["NOBTTS"], 1.0, places=6)

    def test_domicile_favori_si_lambda_superieur(self):
        m = main.market_probabilities(2.5, 0.6)
        self.assertGreater(m["1"], m["2"])

    def test_lambda_eleve_augmente_over(self):
        faible = main.market_probabilities(0.5, 0.5)["O2.5"]
        fort = main.market_probabilities(2.5, 2.5)["O2.5"]
        self.assertGreater(fort, faible)

    def test_dixon_coles_tau_cas_speciaux(self):
        rho = main.DIXON_COLES_RHO
        self.assertAlmostEqual(main._dixon_coles_tau(0, 0, 1.5, 1.0, rho), 1 - 1.5 * 1.0 * rho)
        self.assertAlmostEqual(main._dixon_coles_tau(0, 1, 1.5, 1.0, rho), 1 + 1.5 * rho)
        self.assertAlmostEqual(main._dixon_coles_tau(1, 0, 1.5, 1.0, rho), 1 + 1.0 * rho)
        self.assertAlmostEqual(main._dixon_coles_tau(1, 1, 1.5, 1.0, rho), 1 - rho)
        self.assertEqual(main._dixon_coles_tau(2, 3, 1.5, 1.0, rho), 1.0)


# =============================================================================
# COTES
# =============================================================================
class TestOdds(unittest.TestCase):
    def _event(self, markets):
        return [{
            "home_team": "Arsenal", "away_team": "Chelsea",
            "bookmakers": [{"key": "unibet", "markets": markets}],
        }]

    def test_find_odds_h2h(self):
        events = self._event([{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": 1.8},
            {"name": "Draw", "price": 3.6},
            {"name": "Chelsea", "price": 4.5},
        ]}])
        self.assertEqual(main.find_odds(events, "Arsenal", "Chelsea"),
                         {"1": 1.8, "N": 3.6, "2": 4.5})

    def test_find_odds_totals_filtre_la_ligne(self):
        events = self._event([{"key": "totals", "outcomes": [
            {"name": "Over", "price": 1.9, "point": 2.5},
            {"name": "Under", "price": 1.95, "point": 2.5},
            {"name": "Over", "price": 1.4, "point": 1.5},
        ]}])
        odds = main.find_odds(events, "Arsenal", "Chelsea")
        self.assertEqual(odds, {"O2.5": 1.9, "U2.5": 1.95})

    def test_find_odds_aucun_match(self):
        events = self._event([{"key": "h2h", "outcomes": []}])
        self.assertEqual(main.find_odds(events, "Liverpool", "Everton"), {})

    def test_find_odds_mauvais_bookmaker(self):
        events = [{"home_team": "Arsenal", "away_team": "Chelsea",
                   "bookmakers": [{"key": "betclic", "markets": [
                       {"key": "h2h", "outcomes": [{"name": "Arsenal", "price": 2.0}]}]}]}]
        self.assertEqual(main.find_odds(events, "Arsenal", "Chelsea"), {})

    def test_devig_groupes_separes(self):
        fair = main.devigged_market({"1": 2.0, "N": 3.5, "2": 4.0,
                                     "O2.5": 1.9, "U2.5": 1.95})
        self.assertAlmostEqual(fair["1"] + fair["N"] + fair["2"], 1.0, places=6)
        self.assertAlmostEqual(fair["O2.5"] + fair["U2.5"], 1.0, places=6)

    def test_devig_groupe_incomplet_ignore(self):
        # un seul résultat 1X2 disponible => groupe ignoré
        self.assertEqual(main.devigged_market({"1": 2.0}), {})

    def test_names_match(self):
        self.assertTrue(main.names_match("Manchester City", "Man City"))
        self.assertTrue(main.names_match("Atlético Madrid", "Atletico Madrid"))
        self.assertTrue(main.names_match("Bayern Munich", "Bayern Munich"))
        self.assertFalse(main.names_match("Arsenal", "Chelsea"))
        self.assertFalse(main.names_match("", "Arsenal"))

    def test_normalize_name(self):
        self.assertEqual(main.normalize_name("Paris FC"), "paris")
        self.assertEqual(main.normalize_name("Atlético"), "atletico")


# =============================================================================
# FORCE DES ÉQUIPES
# =============================================================================
class TestStrength(unittest.TestCase):
    def setUp(self):
        main._CACHE.clear()

    def test_to_float(self):
        self.assertEqual(main._to_float("1.5"), 1.5)
        self.assertEqual(main._to_float(2), 2.0)
        self.assertIsNone(main._to_float(None))
        self.assertIsNone(main._to_float("abc"))

    def test_expected_goals_croise_attaque_defense(self):
        home = main.TeamStrength(2.0, 0.8, 1.3, 1.2)
        away = main.TeamStrength(1.5, 1.0, 1.0, 1.6)
        lh, la = main.expected_goals(home, away)
        self.assertAlmostEqual(lh, (2.0 + 1.6) / 2)
        self.assertAlmostEqual(la, (1.0 + 0.8) / 2)

    @patch("main._request")
    def test_get_team_strength_parsing(self, mock_req):
        mock_req.return_value = {"response": {"goals": {
            "for": {"average": {"home": "2.1", "away": "1.4"}},
            "against": {"average": {"home": "0.9", "away": "1.7"}},
        }}}
        s = main.get_team_strength(39, 42, 2024)
        self.assertAlmostEqual(s.attack_home, 2.1)
        self.assertAlmostEqual(s.defense_away, 1.7)

    @patch("main._request")
    def test_get_team_strength_moyenne_zero_preservee(self, mock_req):
        # une moyenne légitime de 0.0 ne doit PAS être écrasée par le défaut
        mock_req.return_value = {"response": {"goals": {
            "for": {"average": {"home": "0.0", "away": "0.0"}},
            "against": {"average": {"home": "0.0", "away": "0.0"}},
        }}}
        s = main.get_team_strength(39, 99, 2024)
        self.assertEqual(s.attack_home, 0.0)

    @patch("main._request")
    def test_get_team_strength_fallback_si_vide(self, mock_req):
        mock_req.return_value = {"response": None}
        s = main.get_team_strength(39, 7, 2024)
        self.assertEqual(s, main.DEFAULT_STRENGTH)

    def test_key_players_out_equipe_non_suivie(self):
        # équipe absente de KEY_PLAYERS => 0, sans aucun appel API
        self.assertEqual(main.key_players_out(999999, 2024), 0)


# =============================================================================
# CONFIGURATION
# =============================================================================
class TestConfig(unittest.TestCase):
    def test_env_float(self):
        with patch.dict(os.environ, {"X_TEST": "3.5"}):
            self.assertEqual(main._env_float("X_TEST", 1.0), 3.5)
        with patch.dict(os.environ, {"X_TEST": ""}):
            self.assertEqual(main._env_float("X_TEST", 1.0), 1.0)
        with patch.dict(os.environ, {"X_TEST": "abc"}):
            self.assertEqual(main._env_float("X_TEST", 7.0), 7.0)

    def test_env_int(self):
        with patch.dict(os.environ, {"X_TEST": "4.9"}):
            self.assertEqual(main._env_int("X_TEST", 1), 4)

    def test_current_season(self):
        self.assertEqual(main.current_season(datetime(2026, 5, 18)), 2025)
        self.assertEqual(main.current_season(datetime(2025, 8, 1)), 2025)
        self.assertEqual(main.current_season(datetime(2025, 7, 31)), 2024)


# =============================================================================
# CACHE
# =============================================================================
class TestCache(unittest.TestCase):
    def setUp(self):
        main._CACHE.clear()

    def test_set_puis_get(self):
        main._cache_set("k", [1, 2, 3])
        self.assertEqual(main._cache_get("k"), [1, 2, 3])

    def test_get_absent(self):
        self.assertIsNone(main._cache_get("inexistant"))

    def test_entree_perimee(self):
        stale = (datetime.now(timezone.utc) - timedelta(hours=13)).isoformat()
        main._CACHE["vieux"] = {"ts": stale, "value": 42}
        self.assertIsNone(main._cache_get("vieux"))

    def test_ttl_personnalise(self):
        # une entrée de 13h est valide sous TTL 72h, périmée sous TTL 12h
        ts = (datetime.now(timezone.utc) - timedelta(hours=13)).isoformat()
        main._CACHE["k"] = {"ts": ts, "value": 7}
        self.assertEqual(main._cache_get("k", timedelta(hours=72)), 7)
        self.assertIsNone(main._cache_get("k", timedelta(hours=12)))


# =============================================================================
# COUCHE HTTP
# =============================================================================
class TestHttp(unittest.TestCase):
    def _resp(self, status, payload=None):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = payload or {}
        return r

    @patch("main.requests.get")
    def test_succes(self, mock_get):
        mock_get.return_value = self._resp(200, {"ok": True})
        self.assertEqual(main._request("http://x"), {"ok": True})

    @patch("main.requests.get")
    def test_erreur_serveur_renvoie_none(self, mock_get):
        mock_get.return_value = self._resp(500)
        self.assertIsNone(main._request("http://x"))

    @patch("main.time.sleep")
    @patch("main.requests.get")
    def test_429_puis_succes(self, mock_get, _sleep):
        mock_get.side_effect = [self._resp(429), self._resp(200, {"ok": 1})]
        self.assertEqual(main._request("http://x"), {"ok": 1})

    @patch("main.time.sleep")
    @patch("main.requests.get")
    def test_exception_puis_succes(self, mock_get, _sleep):
        mock_get.side_effect = [requests.RequestException("boom"),
                                self._resp(200, {"ok": 1})]
        self.assertEqual(main._request("http://x"), {"ok": 1})

    @patch("main.time.sleep")
    @patch("main.requests.get")
    def test_echec_total_renvoie_none(self, mock_get, _sleep):
        mock_get.side_effect = requests.RequestException("boom")
        self.assertIsNone(main._request("http://x"))

    @patch("main._request")
    def test_api_football_renvoie_liste(self, mock_req):
        mock_req.return_value = {"response": [{"a": 1}]}
        self.assertEqual(main.api_football("/fixtures"), [{"a": 1}])
        mock_req.return_value = None
        self.assertEqual(main.api_football("/fixtures"), [])

    @patch("main._request")
    def test_get_odds_echec_renvoie_none(self, mock_req):
        # échec de requête => None (distinct de "pas de match")
        mock_req.return_value = None
        self.assertIsNone(main.get_odds("soccer_epl"))

    @patch("main._request")
    def test_get_odds_succes_vide_renvoie_liste(self, mock_req):
        # requête réussie mais aucun match (hors-saison) => liste vide, pas None
        mock_req.return_value = []
        self.assertEqual(main.get_odds("soccer_epl"), [])


# =============================================================================
# TELEGRAM & MESSAGE
# =============================================================================
class TestTelegram(unittest.TestCase):
    def test_message_vide(self):
        msg = main.build_message([], 2025)
        self.assertIn("Aucun value bet", msg)

    def test_message_avec_paris(self):
        bets = [{"match": "A vs B", "pari": "1", "cote": 2.1, "proba": 55.0,
                 "market_prob": 48.0, "value_pct": 7.5, "stake": 25.0}]
        msg = main.build_message(bets, 2025)
        self.assertIn("1 VALUE BET", msg)
        self.assertIn("A vs B", msg)

    def test_message_echappe_html(self):
        bets = [{"match": "A & B <C>", "pari": "1", "cote": 2.0, "proba": 50.0,
                 "market_prob": 49.0, "value_pct": 6.0, "stake": 10.0}]
        msg = main.build_message(bets, 2025)
        self.assertIn("&amp;", msg)
        self.assertNotIn("<C>", msg)

    @patch("main.requests.post")
    def test_telegram_non_configure_n_envoie_rien(self, mock_post):
        with patch.object(main, "BOT_TOKEN", None), patch.object(main, "CHAT_ID", None):
            main.send_telegram("test")
        mock_post.assert_not_called()


# =============================================================================
# ANALYSE D'UN MATCH (intégration)
# =============================================================================
class TestAnalyseFixture(unittest.TestCase):
    def _fixture(self):
        return {"teams": {"home": {"id": 1, "name": "Arsenal"},
                          "away": {"id": 2, "name": "Chelsea"}}}

    def _odds_events(self, odd_home):
        return [{"home_team": "Arsenal", "away_team": "Chelsea",
                 "bookmakers": [{"key": "unibet", "markets": [
                     {"key": "h2h", "outcomes": [
                         {"name": "Arsenal", "price": odd_home},
                         {"name": "Draw", "price": 3.5},
                         {"name": "Chelsea", "price": 4.0}]}]}]}]

    @patch("main.key_players_out", return_value=0)
    @patch("main.get_team_strength")
    def test_value_detectee_sur_grosse_cote(self, mock_str, _out):
        mock_str.return_value = main.TeamStrength(2.6, 0.6, 2.4, 0.7)
        bets = main.analyse_fixture(self._fixture(), 39, 2024, self._odds_events(5.0))
        self.assertTrue(any(b["pari"] == "1" for b in bets))
        for b in bets:
            self.assertGreater(b["value_pct"], 0)
            self.assertGreater(b["stake"], 0)

    @patch("main.key_players_out", return_value=0)
    @patch("main.get_team_strength")
    def test_aucune_value_sur_cote_juste(self, mock_str, _out):
        mock_str.return_value = main.DEFAULT_STRENGTH
        # lambdas issus de DEFAULT des deux côtés : (1.5+1.5)/2 et (1.1+1.1)/2
        model = main.market_probabilities(1.5, 1.1)
        # cotes exactement justes (1/proba) => value nulle => aucun pari
        events = [{"home_team": "Arsenal", "away_team": "Chelsea",
                   "bookmakers": [{"key": "unibet", "markets": [
                       {"key": "h2h", "outcomes": [
                           {"name": "Arsenal", "price": 1 / model["1"]},
                           {"name": "Draw", "price": 1 / model["N"]},
                           {"name": "Chelsea", "price": 1 / model["2"]}]}]}]}]
        bets = main.analyse_fixture(self._fixture(), 39, 2024, events)
        self.assertEqual(bets, [])

    @patch("main.key_players_out", return_value=0)
    @patch("main.get_team_strength")
    def test_match_sans_cote_ignore(self, mock_str, _out):
        mock_str.return_value = main.DEFAULT_STRENGTH
        self.assertEqual(main.analyse_fixture(self._fixture(), 39, 2024, []), [])


# =============================================================================
# BACKTEST
# =============================================================================
class TestBacktest(unittest.TestCase):
    def test_clip(self):
        self.assertGreater(bt._clip(0.0), 0.0)
        self.assertLess(bt._clip(1.0), 1.0)
        self.assertEqual(bt._clip(0.5), 0.5)

    def test_score_ternary_bonne_prediction(self):
        brier, logloss, correct = bt.score_ternary((0.6, 0.25, 0.15), 0)
        self.assertAlmostEqual(brier, 0.16 + 0.0625 + 0.0225, places=6)
        self.assertEqual(correct, 1)

    def test_score_ternary_mauvaise_prediction(self):
        _, _, correct = bt.score_ternary((0.6, 0.25, 0.15), 2)
        self.assertEqual(correct, 0)

    def test_score_binary(self):
        brier, _, correct = bt.score_binary(0.7, False)
        self.assertAlmostEqual(brier, 0.49, places=6)
        self.assertEqual(correct, 0)
        _, _, correct = bt.score_binary(0.7, True)
        self.assertEqual(correct, 1)

    def test_rolling_strength_prior(self):
        rs = bt.RollingStrength()
        self.assertEqual(rs.strength(1).attack_home, main.DEFAULT_STRENGTH.attack_home)

    def test_rolling_strength_mise_a_jour(self):
        rs = bt.RollingStrength()
        for _ in range(2):
            rs.update({"home": 1, "away": 2, "gh": 3, "ga": 0})
        # (3+3 + défaut*4) / (2+4)
        attendu = (6 + main.DEFAULT_STRENGTH.attack_home * bt.PRIOR_MATCHES) / (2 + bt.PRIOR_MATCHES)
        self.assertAlmostEqual(rs.strength(1).attack_home, attendu)
        self.assertEqual(rs.n_home[1], 2)
        self.assertEqual(rs.n_away[2], 2)

    @patch("main.api_football")
    def test_fetch_finished_filtre_et_trie(self, mock_api):
        mock_api.return_value = [
            {"fixture": {"date": "2024-09-02", "status": {"short": "FT"}},
             "teams": {"home": {"id": 1}, "away": {"id": 2}}, "goals": {"home": 2, "away": 1}},
            {"fixture": {"date": "2024-09-01", "status": {"short": "FT"}},
             "teams": {"home": {"id": 3}, "away": {"id": 4}}, "goals": {"home": 0, "away": 0}},
            {"fixture": {"date": "2024-09-03", "status": {"short": "NS"}},
             "teams": {"home": {"id": 5}, "away": {"id": 6}}, "goals": {"home": None, "away": None}},
        ]
        games = bt.fetch_finished(39, 2024)
        self.assertEqual(len(games), 2)
        self.assertEqual(games[0]["date"], "2024-09-01")
        self.assertEqual(games[1]["gh"], 2)


# =============================================================================
# DÉDUPLICATION & HEARTBEAT (fonctionnement 24/7)
# =============================================================================
class TestDedup(unittest.TestCase):
    def setUp(self):
        main._CACHE.clear()

    def _bet(self, value=8.0):
        return {"match": "A vs B", "pari": "1", "value_pct": value}

    def test_pari_inedit_est_nouveau(self):
        self.assertTrue(main.is_new_bet(self._bet()))

    def test_pari_deja_notifie_ignore(self):
        bet = self._bet()
        main.mark_notified(bet)
        self.assertFalse(main.is_new_bet(bet))

    def test_revalue_significative_re_notifie(self):
        main.mark_notified(self._bet(8.0))
        self.assertTrue(main.is_new_bet(self._bet(8.0 + main.REVALUE_MARGIN + 1)))

    def test_revalue_faible_reste_ignore(self):
        main.mark_notified(self._bet(8.0))
        self.assertFalse(main.is_new_bet(self._bet(8.5)))

    def test_notification_perimee_est_nouveau(self):
        old = (datetime.now(timezone.utc) - main.DEDUP_TTL - timedelta(hours=1)).isoformat()
        main._CACHE["notified:A vs B|1"] = {"ts": old, "value_pct": 8.0}
        self.assertTrue(main.is_new_bet(self._bet()))

    def test_heartbeat_une_fois_par_jour(self):
        self.assertTrue(main._maybe_heartbeat())
        self.assertFalse(main._maybe_heartbeat())

    def test_prune_supprime_entrees_perimees(self):
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        main._CACHE["notified:vieux|1"] = {"ts": old, "value_pct": 5.0}
        main._cache_set("frais", 1)
        main._maybe_heartbeat()
        main._prune_cache()
        self.assertNotIn("notified:vieux|1", main._CACHE)
        self.assertIn("frais", main._CACHE)
        self.assertIn("heartbeat_date", main._CACHE)  # entrée non datée préservée


# =============================================================================
# SUIVI DU ROI
# =============================================================================
class TestRoi(unittest.TestCase):
    def test_pari_gagnant(self):
        s = roi.summarise([{"cote": "2.0", "mise": "5", "resultat": "G"}])
        self.assertEqual(s["n"], 1)
        self.assertAlmostEqual(s["total_retour"], 10.0)
        self.assertAlmostEqual(s["profit"], 5.0)
        self.assertAlmostEqual(s["roi"], 100.0)

    def test_pari_perdant(self):
        s = roi.summarise([{"cote": "2.0", "mise": "5", "resultat": "P"}])
        self.assertAlmostEqual(s["profit"], -5.0)
        self.assertAlmostEqual(s["roi"], -100.0)

    def test_pari_annule_rembourse(self):
        s = roi.summarise([{"cote": "2.0", "mise": "5", "resultat": "A"}])
        self.assertAlmostEqual(s["profit"], 0.0)
        self.assertEqual(s["decisifs"], 0)

    def test_pari_en_attente(self):
        s = roi.summarise([{"cote": "2.0", "mise": "5", "resultat": ""}])
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["pending"], 1)

    def test_ligne_invalide(self):
        rows = [{"cote": "abc", "mise": "5", "resultat": "G"},
                {"cote": "2.0", "mise": "5", "resultat": "X"},
                {"cote": "0", "mise": "5", "resultat": "G"}]
        s = roi.summarise(rows)
        self.assertEqual(s["invalid"], 3)
        self.assertEqual(s["n"], 0)

    def test_bilan_mixte(self):
        rows = [{"cote": "2.0", "mise": "5", "resultat": "G"},
                {"cote": "3.0", "mise": "5", "resultat": "P"},
                {"cote": "2.0", "mise": "5", "resultat": "A"}]
        s = roi.summarise(rows)
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["total_mise"], 15.0)
        self.assertAlmostEqual(s["profit"], 0.0)
        self.assertAlmostEqual(s["taux"], 50.0)

    def test_aucune_donnee(self):
        s = roi.summarise([])
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["roi"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
