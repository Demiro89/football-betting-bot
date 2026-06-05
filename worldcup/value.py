"""ÉTAPE 3 — Stratégie de value betting + gestion de bankroll (Kelly).

Principe du value bet :
  - Le bookmaker propose une cote décimale `c` pour une issue.
  - Sa probabilité *implicite* brute est `1/c`, mais elle inclut sa marge
    (« vig/overround »). On retire cette marge pour obtenir la probabilité
    « juste » du marché (devigging par normalisation).
  - On compare à la probabilité du modèle. Il y a *value* si :
        proba_modèle  >  proba_implicite (1/cote)
    soit, de façon équivalente, si l'edge `proba_modèle * cote - 1 > seuil`.

Mise : **critère de Kelly fractionné**, plafonné. Le Kelly « plein » maximise
la croissance logarithmique de la bankroll mais est très volatil ; on en prend
une fraction (par défaut 1/4) et on plafonne à un % de la bankroll pour survivre
à la variance et aux erreurs d'estimation du modèle.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config


def implied_probabilities(odds: dict[str, float]) -> dict[str, float]:
    """Probabilités implicites « justes » du marché (marge bookmaker retirée).

    `odds` : {'1': cote, 'N': cote, '2': cote}. Les issues absentes sont ignorées.
    """
    present = {k: v for k, v in odds.items() if v and v > 1.0}
    if len(present) < 2:
        return {}
    inv = {k: 1.0 / v for k, v in present.items()}
    overround = sum(inv.values())  # > 1 ; l'excédent est la marge du bookmaker
    return {k: v / overround for k, v in inv.items()}


def kelly_stake(prob: float, odds: float, bankroll: float) -> float:
    """Mise via Kelly fractionné plafonné. Renvoie 0 si pari déconseillé.

    prob : probabilité modèle [0,1] ; odds : cote décimale ; bankroll : capital.
    """
    net = odds - 1.0                      # gain net par unité misée
    if net <= 0:
        return 0.0
    edge = prob * odds - 1.0              # espérance par unité misée
    if edge <= 0:
        return 0.0
    kelly = edge / net                    # fraction de Kelly « pleine »
    fraction = min(kelly * config.KELLY_FRACTION, config.MAX_BET_FRACTION)
    stake = bankroll * fraction
    if stake < config.MIN_BET:
        return 0.0
    return round(float(stake), 2)


@dataclass
class ValueBet:
    """Une opportunité détectée sur une issue d'un match."""
    selection: str          # '1' | 'N' | '2'
    label: str              # libellé lisible
    odds: float             # cote bookmaker
    model_prob: float       # proba du modèle
    market_prob: float      # proba implicite (devig)
    edge_pct: float         # value en %
    stake: float            # mise recommandée (Kelly fractionné)


def find_value_bets(
    model_probs: dict[str, float],
    odds: dict[str, float],
    bankroll: float | None = None,
    min_value: float | None = None,
) -> list[ValueBet]:
    """Compare proba modèle vs cotes et renvoie les value bets triés par edge.

    Un pari n'est retenu que si : edge >= seuil ET mise Kelly > 0.
    """
    bankroll = config.BANKROLL if bankroll is None else bankroll
    min_value = config.MIN_VALUE if min_value is None else min_value

    market = implied_probabilities(odds)
    bets: list[ValueBet] = []
    for sel in config.OUTCOMES:
        odd = odds.get(sel)
        p = model_probs.get(sel)
        if not odd or p is None:
            continue
        edge = p * odd - 1.0
        if edge < min_value:
            continue
        stake = kelly_stake(p, odd, bankroll)
        if stake <= 0:
            continue
        bets.append(
            ValueBet(
                selection=sel,
                label=config.OUTCOME_LABELS[sel],
                odds=round(float(odd), 2),
                model_prob=round(float(p), 4),
                market_prob=round(float(market.get(sel, 1.0 / odd)), 4),
                edge_pct=round(float(edge) * 100, 1),
                stake=stake,
            )
        )
    bets.sort(key=lambda b: b.edge_pct, reverse=True)
    return bets
