"""Simulateur de tournoi « façon Opta » (Monte-Carlo).

Principe (identique à celui des supercalculateurs type Opta) : on rejoue le
tournoi des milliers de fois. À chaque simulation, chaque match est tiré au
sort selon la probabilité de victoire estimée par le modèle ML. En agrégeant
toutes les simulations, on obtient pour chaque nation sa probabilité d'aller
en demi-finale, en finale, et de remporter le titre.

Honnêteté : ce sont les forces issues de NOTRE modèle (Elo + forme sur les
matchs internationaux), pas les données privées d'Opta. La méthode est la même ;
la qualité dépend du modèle. Les matchs sont joués sur terrain neutre et le
tirage est aléatoire à chaque simulation (le bracket réel n'est pas encodé) :
les probabilités sont donc indicatives.
"""

from __future__ import annotations

import math
import random

import pandas as pd

# Date de référence neutre pour les features (matchs futurs hypothétiques).
_REF_DATE = pd.Timestamp("2026-06-15")


def knockout_win_prob(model, a: str, b: str, tournament: str = "FIFA World Cup",
                      date=None) -> float:
    """Probabilité que l'équipe `a` élimine `b` (terrain neutre).

    En élimination directe il n'y a pas de nul : la probabilité de match nul
    est répartie entre les deux équipes au prorata de leur force (tirs au but).
    """
    date = date or _REF_DATE
    p = model.predict_match(a, b, date, neutral=True, tournament=tournament)
    p1, pn, p2 = p["1"], p["N"], p["2"]
    denom = p1 + p2
    if denom <= 0:
        return 0.5
    return p1 + pn * (p1 / denom)


def pairwise_matrix(model, teams, tournament: str = "FIFA World Cup", date=None) -> dict:
    """Matrice des probabilités de victoire pour chaque paire (calculée une fois)."""
    teams = list(teams)
    probs: dict[tuple[str, str], float] = {}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            pa = knockout_win_prob(model, a, b, tournament, date)
            probs[(a, b)] = pa
            probs[(b, a)] = 1.0 - pa
    return probs


def simulate_tournament(teams, win_prob: dict, n_sims: int = 5000, seed: int = 42) -> dict:
    """Monte-Carlo d'un tournoi à élimination directe (bracket aléatoire).

    Renvoie, par équipe, les probabilités d'atteindre les stades :
    {team: {"champion": p, "finale": p, "demi": p, "quart": p}}.
    Pour un bracket parfait (sans exempt), choisir 4, 8, 16 ou 32 équipes.
    """
    teams = list(teams)
    n = len(teams)
    if n < 2:
        return {t: {"champion": 0.0, "finale": 0.0, "demi": 0.0, "quart": 0.0} for t in teams}

    size = 1
    while size < n:
        size *= 2
    n_rounds = int(math.log2(size))
    rng = random.Random(seed)

    champ = dict.fromkeys(teams, 0)
    final = dict.fromkeys(teams, 0)
    semi = dict.fromkeys(teams, 0)
    quart = dict.fromkeys(teams, 0)

    for _ in range(n_sims):
        bracket = teams + [None] * (size - n)
        rng.shuffle(bracket)
        wins = dict.fromkeys(teams, 0)
        alive = bracket
        while len(alive) > 1:
            nxt = []
            for i in range(0, len(alive), 2):
                a, b = alive[i], alive[i + 1]
                if a is None:
                    w = b
                elif b is None:
                    w = a
                else:
                    w = a if rng.random() < win_prob[(a, b)] else b
                if w is not None:
                    wins[w] += 1
                nxt.append(w)
            alive = nxt
        for t in teams:
            wv = wins[t]
            if wv == n_rounds:
                champ[t] += 1
            if wv >= n_rounds - 1:
                final[t] += 1
            if n_rounds >= 2 and wv >= n_rounds - 2:
                semi[t] += 1
            if n_rounds >= 3 and wv >= n_rounds - 3:
                quart[t] += 1

    return {
        t: {
            "champion": champ[t] / n_sims,
            "finale": final[t] / n_sims,
            "demi": semi[t] / n_sims,
            "quart": quart[t] / n_sims,
        }
        for t in teams
    }


def simulate_from_model(model, teams, n_sims: int = 5000,
                        tournament: str = "FIFA World Cup", seed: int = 42) -> dict:
    """Raccourci : matrice de probabilités puis simulation."""
    probs = pairwise_matrix(model, teams, tournament)
    return simulate_tournament(teams, probs, n_sims=n_sims, seed=seed)
