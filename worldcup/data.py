"""ÉTAPE 1 — Data pipeline.

Récupération de données de football *gratuites*. On s'appuie sur le jeu de
données public « International football results » (martj42, licence CC0,
~47 000 matchs internationaux de 1872 à aujourd'hui), idéal pour la Coupe du
Monde car il couvre toutes les sélections nationales, les terrains neutres et
le type de compétition.

Pourquoi ce choix plutôt qu'une API payante :
  - 100 % gratuit, sans clé, sans quota.
  - Historique profond -> entraînement robuste d'un modèle ML.
  - Mis à jour après chaque journée internationale.

Pour aller plus loin (xG, possession, stats joueurs), on peut brancher en
complément, dans `enrich_*` ci-dessous :
  - `soccerdata` (scraping FBref/Understat) pour l'xG par équipe ;
  - API-Football (api-sports.io, palier gratuit) pour les compositions/blessures.
Ces enrichissements sont optionnels : le modèle fonctionne sans eux et les
intègre comme features supplémentaires s'ils sont disponibles.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from . import config

log = logging.getLogger("worldcup.data")

# Colonnes attendues dans le CSV source.
_EXPECTED_COLS = {
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
}


def download_results(force: bool = False) -> None:
    """Télécharge le CSV historique et le met en cache local.

    Si le téléchargement échoue mais qu'un cache existe, on le conserve : le
    pipeline reste utilisable hors-ligne.
    """
    if config.RESULTS_CSV.exists() and not force:
        log.info("Cache présent (%s) — pas de téléchargement.", config.RESULTS_CSV.name)
        return

    log.info("Téléchargement des résultats internationaux depuis %s", config.RESULTS_URL)
    try:
        df = pd.read_csv(config.RESULTS_URL)
    except Exception as exc:  # réseau coupé, URL changée, etc.
        if config.RESULTS_CSV.exists():
            log.warning("Téléchargement impossible (%s) — cache local conservé.", exc)
            return
        raise RuntimeError(
            f"Impossible de télécharger les données et aucun cache local : {exc}"
        ) from exc

    config.RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_CSV, index=False)
    log.info("Sauvegardé : %d matchs -> %s", len(df), config.RESULTS_CSV)


def load_results(min_year: int | None = None) -> pd.DataFrame:
    """Charge et nettoie l'historique des matchs internationaux.

    Renvoie un DataFrame trié chronologiquement avec une colonne `result`
    (1 / N / 2 du point de vue de l'équipe à domicile) et `tournament_weight`.
    """
    download_results()
    df = pd.read_csv(config.RESULTS_CSV)

    missing = _EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV source : {missing}")

    # Typage et nettoyage : on retire les matchs sans score (à venir/forfaits).
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    min_year = config.MIN_YEAR if min_year is None else min_year
    df = df[df["date"].dt.year >= min_year]

    # Issue du match, du point de vue domicile : 1 (victoire), N (nul), 2 (défaite).
    def _result(row) -> str:
        if row["home_score"] > row["away_score"]:
            return "1"
        if row["home_score"] < row["away_score"]:
            return "2"
        return "N"

    df["result"] = df.apply(_result, axis=1)
    df["tournament_weight"] = df["tournament"].map(tournament_weight)

    df = df.sort_values("date").reset_index(drop=True)
    log.info("Historique chargé : %d matchs (depuis %d).", len(df), min_year)
    return df


def tournament_weight(name: object) -> float:
    """Importance d'une compétition (multiplicateur Elo + feature)."""
    key = str(name).strip().lower()
    for needle, weight in config.TOURNAMENT_WEIGHTS.items():
        if needle in key:
            return weight
    return config.DEFAULT_TOURNAMENT_WEIGHT


# ---------------------------------------------------------------------------
# Enrichissements optionnels (xG / stats avancées)
# ---------------------------------------------------------------------------
def enrich_with_xg(df: pd.DataFrame) -> pd.DataFrame:
    """Point d'extension : ajoute des colonnes xG si `soccerdata` est installé.

    Volontairement no-op par défaut pour rester sans dépendance réseau lourde.
    Pour l'activer, installez `soccerdata` et complétez cette fonction en
    rapprochant les équipes par nom (cf. `understat`/`fbref` readers).
    """
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    frame = load_results()
    print(frame.tail(5).to_string(index=False))
    print(f"\nÀ jour au {datetime.now(timezone.utc):%Y-%m-%d}. Total : {len(frame)} matchs.")
