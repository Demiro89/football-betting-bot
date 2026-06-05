"""Construction de tickets de paris (simples et combinés) + propositions.

⚠️ Honnêteté statistique : un COMBINÉ (parlay) multiplie les cotes, mais aussi
la marge du bookmaker ET la variance. Les paris SIMPLES restent préférables sur
le long terme. On ne propose donc un combiné QUE s'il est constitué de jambes
individuellement « value » (proba modèle > proba implicite), et on suppose
l'indépendance des résultats — il faut donc éviter de combiner des issues
corrélées (ex. deux résultats du même match).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from . import config
from .value import kelly_stake


@dataclass
class TicketLeg:
    """Une sélection (jambe) d'un ticket."""
    match: str
    selection: str          # '1' | 'N' | '2'
    label: str              # libellé lisible
    odds: float             # meilleure cote
    model_prob: float       # proba retenue (modèle/ensemble)
    market_prob: float      # proba implicite du marché (dévigée)


@dataclass
class Ticket:
    """Un ticket = une ou plusieurs jambes combinées."""
    legs: list[TicketLeg]
    combined_odds: float
    model_prob: float       # produit des probas (indépendance supposée)
    market_prob: float      # produit des probas implicites
    edge_pct: float         # value du ticket en %
    stake: float            # mise conseillée (Kelly fractionné)
    potential_return: float  # gain brut potentiel (mise × cote)
    kind: str               # 'simple' | 'combiné'


def build_ticket(legs, bankroll: float | None = None, kind: str | None = None) -> Ticket:
    """Construit un ticket à partir d'une liste de jambes.

    Cote totale = produit des cotes ; proba = produit des probas (indépendance).
    Mise = Kelly fractionné sur (proba combinée, cote combinée).
    """
    legs = list(legs)
    bankroll = config.BANKROLL if bankroll is None else bankroll
    combined_odds = prod(leg.odds for leg in legs)
    model_prob = prod(leg.model_prob for leg in legs)
    market_prob = prod(leg.market_prob for leg in legs)
    edge = model_prob * combined_odds - 1.0
    stake = kelly_stake(model_prob, combined_odds, bankroll)
    kind = kind or ("simple" if len(legs) == 1 else "combiné")
    return Ticket(
        legs=legs,
        combined_odds=round(float(combined_odds), 2),
        model_prob=float(model_prob),
        market_prob=float(market_prob),
        edge_pct=round(float(edge) * 100, 1),
        stake=stake,
        potential_return=round(stake * float(combined_odds), 2),
        kind=kind,
    )


def propose_tickets(
    value_legs,
    bankroll: float | None = None,
    max_combo_legs: int = 3,
    min_combo_edge: float = 0.0,
) -> dict:
    """Propose des tickets à partir de jambes « value ».

    - 'singles' : chaque value bet comme ticket simple (approche recommandée),
      triés par edge décroissant.
    - 'combo'   : un combiné des meilleures jambes value (max `max_combo_legs`),
      sans jamais réunir deux issues du même match (corrélation), retenu
      seulement si son edge >= `min_combo_edge` et sa mise Kelly > 0.
    """
    bankroll = config.BANKROLL if bankroll is None else bankroll
    legs = list(value_legs)

    singles = [build_ticket([leg], bankroll, "simple") for leg in legs]
    singles.sort(key=lambda t: t.edge_pct, reverse=True)

    combo = None
    # Une seule jambe par match (évite les combinaisons corrélées).
    best_per_match: dict[str, TicketLeg] = {}
    for leg in sorted(legs, key=lambda l: l.model_prob * l.odds - 1, reverse=True):
        best_per_match.setdefault(leg.match, leg)
    chosen = list(best_per_match.values())[:max_combo_legs]
    if len(chosen) >= 2:
        ticket = build_ticket(chosen, bankroll, "combiné")
        # On propose le combiné dès que sa value est positive. La mise peut être
        # nulle (Kelly < mise minimale) : c'est alors un combiné « pour le plaisir »,
        # signalé tel quel dans l'interface — on ne le cache pas.
        if ticket.edge_pct / 100.0 >= min_combo_edge:
            combo = ticket

    return {"singles": singles, "combo": combo}
