"""
Backtest / évaluation du modèle (sans pari réel).

Rejoue une ou plusieurs saisons en walk-forward : chaque match est prédit
uniquement à partir des matchs antérieurs (aucune fuite de données), puis
comparé au résultat réel. Mesure la qualité prédictive du modèle (score de
Brier, log-loss, calibration) face à une baseline de fréquences de base.

Un modèle qui ne bat pas la baseline ne peut pas être rentable.

Usage : python backtest.py [saison] [league_id ...]
  - saison    : année de début (défaut : saison précédente)
  - league_id : ids API-Football (défaut : tous ceux de main.LEAGUES)
"""

import sys
from collections import defaultdict
from datetime import datetime, timezone
from math import log

import main as m

PRIOR_MATCHES = 4    # poids du prior de force (en nombre de matchs équivalents)
MIN_HISTORY = 3      # matchs requis (domicile / extérieur) avant d'évaluer un match
EPS = 1e-9
FINISHED = {"FT", "AET", "PEN"}


def fetch_finished(league_id: int, season: int) -> list[dict]:
    """Matchs terminés d'un championnat, triés par date."""
    fixtures = m.api_football("/fixtures", {"league": league_id, "season": season})
    games = []
    for f in fixtures:
        if f.get("fixture", {}).get("status", {}).get("short") not in FINISHED:
            continue
        gh = f.get("goals", {}).get("home")
        ga = f.get("goals", {}).get("away")
        if gh is None or ga is None:
            continue
        games.append({
            "date": f.get("fixture", {}).get("date", ""),
            "home": f["teams"]["home"]["id"],
            "away": f["teams"]["away"]["id"],
            "gh": int(gh),
            "ga": int(ga),
        })
    games.sort(key=lambda g: g["date"])
    return games


class RollingStrength:
    """Force des équipes calculée au fil de l'eau (matchs antérieurs uniquement)."""

    def __init__(self):
        self.for_home = defaultdict(float)
        self.against_home = defaultdict(float)
        self.n_home = defaultdict(int)
        self.for_away = defaultdict(float)
        self.against_away = defaultdict(float)
        self.n_away = defaultdict(int)

    def strength(self, team: int) -> m.TeamStrength:
        d = m.DEFAULT_STRENGTH

        def shrink(total: float, n: int, prior: float) -> float:
            return (total + prior * PRIOR_MATCHES) / (n + PRIOR_MATCHES)

        return m.TeamStrength(
            attack_home=shrink(self.for_home[team], self.n_home[team], d.attack_home),
            defense_home=shrink(self.against_home[team], self.n_home[team], d.defense_home),
            attack_away=shrink(self.for_away[team], self.n_away[team], d.attack_away),
            defense_away=shrink(self.against_away[team], self.n_away[team], d.defense_away),
        )

    def update(self, g: dict) -> None:
        h, a = g["home"], g["away"]
        self.for_home[h] += g["gh"]
        self.against_home[h] += g["ga"]
        self.n_home[h] += 1
        self.for_away[a] += g["ga"]
        self.against_away[a] += g["gh"]
        self.n_away[a] += 1


def _clip(p: float) -> float:
    return min(max(p, EPS), 1 - EPS)


def score_ternary(probs: tuple, outcome: int) -> tuple[float, float, int]:
    """Brier, log-loss et bonne prédiction (argmax) pour un marché à 3 issues."""
    brier = sum((p - (k == outcome)) ** 2 for k, p in enumerate(probs))
    logloss = -log(_clip(probs[outcome]))
    correct = int(max(range(3), key=lambda k: probs[k]) == outcome)
    return brier, logloss, correct


def score_binary(p_yes: float, happened: bool) -> tuple[float, float, int]:
    """Brier, log-loss et bonne prédiction pour un marché binaire."""
    p, y = _clip(p_yes), int(happened)
    brier = (p_yes - y) ** 2
    logloss = -(y * log(p) + (1 - y) * log(1 - p))
    correct = int((p_yes >= 0.5) == bool(y))
    return brier, logloss, correct


def _mean(rows: list, idx: int) -> float:
    return sum(r[idx] for r in rows) / len(rows) if rows else 0.0


def report(season: int, model: dict, base: dict, calib: list) -> None:
    n = len(model["1x2"])
    print(f"\n=== BACKTEST saison {season} — {n} matchs évalués ===\n")
    if n == 0:
        print("Aucun match évaluable (historique insuffisant).")
        return

    for label, key in (("1X2", "1x2"), ("Over/Under 2.5", "ou"), ("BTTS", "btts")):
        mb, ml, mc = (_mean(model[key], i) for i in range(3))
        bb, bl, bc = (_mean(base[key], i) for i in range(3))
        verdict = "modèle > baseline" if ml < bl else "modèle <= baseline (sans valeur)"
        print(f"Marché {label} :")
        print(f"  Brier     modèle {mb:.4f} | baseline {bb:.4f}")
        print(f"  Log-loss  modèle {ml:.4f} | baseline {bl:.4f}   -> {verdict}")
        print(f"  Précision modèle {mc * 100:.1f}% | baseline {bc * 100:.1f}%\n")

    print("Calibration P(victoire domicile) — prévu vs réel :")
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        sel = [(p, y) for p, y in calib if lo <= p < hi]
        if not sel:
            continue
        pred = sum(p for p, _ in sel) / len(sel)
        real = sum(1 for _, y in sel if y) / len(sel)
        print(f"  [{lo * 100:>3.0f}-{min(hi, 1) * 100:>3.0f}%] "
              f"prévu {pred * 100:5.1f}% | réel {real * 100:5.1f}%  (n={len(sel)})")

    print("\nNote : ROI non calculé (pas de cotes historiques gratuites). Ce backtest")
    print("mesure la qualité prédictive du modèle, pas la rentabilité. Le suivi du")
    print("ROI réel se fait via bets_log.csv une fois le bot en production.")


def backtest(season: int, league_ids: list[int]) -> None:
    all_games = []
    for lid in league_ids:
        games = fetch_finished(lid, season)
        print(f"  Ligue {lid:>4} : {len(games)} matchs terminés")
        all_games.append(games)

    total = sum(len(g) for g in all_games)
    if total == 0:
        print("\nAucun match terminé trouvé — saison ou ids de ligue invalides ?")
        return

    # Fréquences de base (baseline) sur l'ensemble des matchs.
    n_home = n_draw = n_away = n_over = n_btts = 0
    for games in all_games:
        for g in games:
            if g["gh"] > g["ga"]:
                n_home += 1
            elif g["gh"] == g["ga"]:
                n_draw += 1
            else:
                n_away += 1
            if g["gh"] + g["ga"] >= 3:
                n_over += 1
            if g["gh"] >= 1 and g["ga"] >= 1:
                n_btts += 1
    base_1x2 = (n_home / total, n_draw / total, n_away / total)
    base_over = n_over / total
    base_btts = n_btts / total

    model: dict = {"1x2": [], "ou": [], "btts": []}
    base: dict = {"1x2": [], "ou": [], "btts": []}
    calib: list = []

    for games in all_games:
        rs = RollingStrength()
        for g in games:
            ready = rs.n_home[g["home"]] >= MIN_HISTORY and rs.n_away[g["away"]] >= MIN_HISTORY
            if ready:
                lh, la = m.expected_goals(rs.strength(g["home"]), rs.strength(g["away"]))
                probs = m.market_probabilities(lh, la)

                outcome = 0 if g["gh"] > g["ga"] else (1 if g["gh"] == g["ga"] else 2)
                over = g["gh"] + g["ga"] >= 3
                btts = g["gh"] >= 1 and g["ga"] >= 1

                model["1x2"].append(score_ternary((probs["1"], probs["N"], probs["2"]), outcome))
                model["ou"].append(score_binary(probs["O2.5"], over))
                model["btts"].append(score_binary(probs["BTTS"], btts))

                base["1x2"].append(score_ternary(base_1x2, outcome))
                base["ou"].append(score_binary(base_over, over))
                base["btts"].append(score_binary(base_btts, btts))

                calib.append((probs["1"], outcome == 0))
            rs.update(g)

    report(season, model, base, calib)


if __name__ == "__main__":
    if not m.API_FOOTBALL_KEY:
        print("API_FOOTBALL_KEY manquante — renseignez-la dans .env")
        sys.exit(1)

    args = sys.argv[1:]
    default_season = m.current_season(datetime.now(timezone.utc)) - 1
    season = int(args[0]) if args else default_season
    leagues = [int(x) for x in args[1:]] if len(args) > 1 else list(m.LEAGUES.keys())

    print(f"Backtest saison {season} | ligues : {leagues}")
    backtest(season, leagues)
