"""
Suivi du ROI réel.

Lit resultats.csv (que vous remplissez à la main, un pari placé par ligne)
et affiche votre bilan : mise totale, profit/perte, rendement, taux de réussite.

Colonnes attendues : date,match,pari,cote,mise,resultat
  resultat : G = gagné, P = perdu, A = annulé/remboursé, vide = en attente

Usage : python roi.py
"""

import csv
import sys
from pathlib import Path

RESULTS_FILE = Path("docs/resultats.csv")
SIGNIFICANT_SAMPLE = 30  # nombre de paris en dessous duquel le ROI n'est pas fiable


def summarise(rows: list[dict]) -> dict:
    """Calcule le bilan à partir des lignes de resultats.csv."""
    placed: list[tuple[float, float, str]] = []
    pending = invalid = 0
    for row in rows:
        res = (row.get("resultat") or "").strip().upper()
        if res == "":
            pending += 1
            continue
        try:
            cote = float(row["cote"])
            mise = float(row["mise"])
        except (KeyError, ValueError, TypeError):
            invalid += 1
            continue
        if res not in ("G", "P", "A") or cote <= 0 or mise <= 0:
            invalid += 1
            continue
        placed.append((cote, mise, res))

    total_mise = sum(m for _, m, _ in placed)
    total_retour = sum(
        cote * mise if res == "G" else (mise if res == "A" else 0.0)
        for cote, mise, res in placed
    )
    wins = sum(1 for _, _, res in placed if res == "G")
    decisifs = sum(1 for _, _, res in placed if res in ("G", "P"))
    profit = total_retour - total_mise
    return {
        "n": len(placed),
        "pending": pending,
        "invalid": invalid,
        "total_mise": total_mise,
        "total_retour": total_retour,
        "profit": profit,
        "roi": profit / total_mise * 100 if total_mise else 0.0,
        "wins": wins,
        "decisifs": decisifs,
        "taux": wins / decisifs * 100 if decisifs else 0.0,
    }


def main() -> None:
    if not RESULTS_FILE.exists():
        print(f"Fichier {RESULTS_FILE} introuvable — créez-le avec l'en-tête :")
        print("date,match,pari,cote,mise,resultat")
        sys.exit(1)

    with RESULTS_FILE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    s = summarise(rows)

    if s["invalid"]:
        print(f"({s['invalid']} ligne(s) ignorée(s) : cote/mise/resultat invalide)\n")

    if s["n"] == 0:
        print(f"Aucun pari terminé. ({s['pending']} en attente)")
        return

    print(f"=== BILAN ({s['n']} paris terminés, {s['pending']} en attente) ===")
    print(f"Mise totale      : {s['total_mise']:.2f} €")
    print(f"Retour total     : {s['total_retour']:.2f} €")
    print(f"Profit / perte   : {s['profit']:+.2f} €")
    print(f"Rendement (ROI)  : {s['roi']:+.1f} %")
    print(f"Taux de réussite : {s['taux']:.1f} % ({s['wins']}/{s['decisifs']})")
    print()

    if s["n"] < SIGNIFICANT_SAMPLE:
        print(f"⚠️  Échantillon trop petit (< {SIGNIFICANT_SAMPLE} paris) — "
              "résultat non significatif, continuez à mises prudentes.")
    elif s["roi"] < 0:
        print("⚠️  ROI négatif : le modèle perd de l'argent. Arrêtez ou recalibrez.")
    else:
        print("ROI positif — continuez le suivi, gardez des mises prudentes.")


if __name__ == "__main__":
    main()
