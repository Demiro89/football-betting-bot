"""Suivi des paris : CLV (Closing Line Value) et ROI réel.

Pourquoi c'est la pièce la plus importante pour « battre les bookmakers » :

  - Le **CLV** compare la cote que vous avez prise à la cote de clôture (juste
    avant le coup d'envoi, quand le marché est le plus efficient). Si vous
    obtenez régulièrement une cote MEILLEURE que la clôture, vous battez le
    marché — c'est le seul indicateur fiable AVANT d'avoir un gros échantillon
    de résultats. Les pros pilotent là-dessus, pas sur le win-rate court terme.

  - Le **ROI** mesure le résultat financier réel une fois les matchs joués.

Format CSV (`data/tracked_bets.csv`) :
    timestamp,match,selection,odds_taken,model_prob,consensus_prob,edge_pct,
    stake,closing_odds,result
  - `closing_odds` : cote de clôture (à renseigner plus tard) -> CLV.
  - `result` : G (gagné) / P (perdu) / A (annulé) / vide (en attente) -> ROI.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone

from . import config

_HEADER = [
    "timestamp", "match", "selection", "odds_taken", "model_prob",
    "consensus_prob", "edge_pct", "stake", "closing_odds", "result",
]


def log_bet(match: str, selection: str, odds_taken: float, model_prob: float,
            consensus_prob: float, edge_pct: float, stake: float) -> None:
    """Ajoute un pari au journal de suivi (au moment où on le place)."""
    config.BETS_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not config.BETS_LOG.exists()
    with config.BETS_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(_HEADER)
        w.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            match, selection, round(odds_taken, 3), round(model_prob, 4),
            round(consensus_prob, 4), round(edge_pct, 2), round(stake, 2), "", "",
        ])


def log_ticket(ticket) -> None:
    """Journalise un ticket (simple ou combiné) comme un pari unique.

    Un combiné se suit exactement comme un pari simple : sa cote est le produit
    des cotes des jambes, et son CLV se calcule pareil (cote prise vs clôture,
    la clôture du combiné étant le produit des cotes de clôture des jambes).
    Le détail des sélections est conservé dans le champ `match`.
    """
    description = " + ".join(f"{leg.match} [{leg.label}]" for leg in ticket.legs)
    selection = "simple" if len(ticket.legs) == 1 else f"combiné x{len(ticket.legs)}"
    log_bet(
        match=description,
        selection=selection,
        odds_taken=ticket.combined_odds,
        model_prob=ticket.model_prob,
        consensus_prob=ticket.market_prob,
        edge_pct=ticket.edge_pct,
        stake=ticket.stake,
    )


def load_bets() -> list[dict]:
    """Charge le journal des paris suivis (liste de dicts)."""
    if not config.BETS_LOG.exists():
        return []
    with config.BETS_LOG.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clv_percent(odds_taken: float, closing_odds: float) -> float:
    """Closing Line Value en %, basé sur les probabilités implicites.

    CLV > 0  => vous avez pris une cote meilleure que la clôture (bon signe).
    Formule : (proba_implicite_clôture / proba_implicite_prise) - 1
            = (cote_prise / cote_clôture) - 1.
    """
    if odds_taken <= 1.0 or closing_odds <= 1.0:
        return 0.0
    return (odds_taken / closing_odds - 1.0) * 100.0


def summary() -> dict:
    """Agrège CLV moyen et ROI réel à partir du journal."""
    bets = load_bets()
    clvs, settled_stake, settled_return, n_settled, n_total = [], 0.0, 0.0, 0, 0
    for b in bets:
        n_total += 1
        try:
            taken = float(b.get("odds_taken") or 0)
            closing = float(b.get("closing_odds") or 0)
            if taken > 1 and closing > 1:
                clvs.append(clv_percent(taken, closing))
        except ValueError:
            pass

        result = (b.get("result") or "").strip().upper()
        try:
            stake = float(b.get("stake") or 0)
            taken = float(b.get("odds_taken") or 0)
        except ValueError:
            continue
        if result in ("G", "P", "A"):
            n_settled += 1
            settled_stake += stake
            if result == "G":
                settled_return += stake * taken
            elif result == "A":
                settled_return += stake  # remboursé

    roi = ((settled_return - settled_stake) / settled_stake * 100.0) if settled_stake else 0.0
    avg_clv = (sum(clvs) / len(clvs)) if clvs else None
    return {
        "n_total": n_total,
        "n_settled": n_settled,
        "n_with_clv": len(clvs),
        "avg_clv_pct": avg_clv,
        "roi_pct": roi if n_settled else None,
        "settled_stake": settled_stake,
        "settled_return": settled_return,
    }
