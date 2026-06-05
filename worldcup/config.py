"""Configuration centrale du module ML Coupe du Monde.

Tout paramètre ajustable est ici (et surchargeable par variable d'environnement),
pour éviter les « nombres magiques » dispersés dans le code.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache_ml"
MODELS_DIR = ROOT / "models"

# Données historiques (jeu de données public martj42, ~47 000 matchs internationaux).
RESULTS_CSV = CACHE_DIR / "international_results.csv"
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# Matchs à venir + cotes bookmaker (éditable à la main ou alimenté par une API).
FIXTURES_CSV = DATA_DIR / "wc2026_fixtures.csv"

# Modèle entraîné sérialisé.
MODEL_FILE = MODELS_DIR / "wc_model.joblib"

for _d in (DATA_DIR, CACHE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers d'environnement
# ---------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, default))


# ---------------------------------------------------------------------------
# Feature engineering (Elo + forme)
# ---------------------------------------------------------------------------
ELO_START = 1500.0          # rating initial d'une nation inconnue
ELO_K = 20.0                # facteur K de base (avant pondération par tournoi/écart)
ELO_HOME_ADV = 65.0         # bonus Elo de l'équipe à domicile (annulé si terrain neutre)
FORM_WINDOW = 10            # nb de matchs internationaux récents pour la « forme »

# Importance du match -> multiplicateur du K Elo et feature `tournament_weight`.
# Un match de Coupe du Monde fait bien plus bouger les ratings qu'un amical.
TOURNAMENT_WEIGHTS: dict[str, float] = {
    "fifa world cup": 3.0,
    "world cup qualification": 1.8,
    "uefa euro": 2.5,
    "copa américa": 2.5,
    "copa america": 2.5,
    "african cup of nations": 2.2,
    "afc asian cup": 2.0,
    "confederations cup": 2.2,
    "uefa nations league": 1.8,
    "friendly": 1.0,
}
DEFAULT_TOURNAMENT_WEIGHT = 1.5

# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
# Année à partir de laquelle on garde les matchs (le football d'avant-guerre
# n'est pas représentatif du jeu moderne).
MIN_YEAR = _env_int("WC_MIN_YEAR", 1990)
RANDOM_STATE = 42
# Part des matchs (les plus récents) réservés au test lors du backtest walk-forward.
TEST_FRACTION = _env_float("WC_TEST_FRACTION", 0.20)

# ---------------------------------------------------------------------------
# Stratégie de mise
# ---------------------------------------------------------------------------
BANKROLL = _env_float("BANKROLL", 100.0)
MIN_VALUE = _env_float("MIN_VALUE_PERCENT", 5.0) / 100.0   # edge minimal pour signaler
KELLY_FRACTION = _env_float("KELLY_FRACTION", 0.25)        # quart de Kelly (sécurisé)
MAX_BET_FRACTION = _env_float("MAX_BET_FRACTION", 0.05)    # plafond 5 % de bankroll
MIN_BET = _env_float("MIN_BET", 1.0)

# Libellés des issues (ordre = labels 0/1/2 du modèle).
OUTCOMES = ("1", "N", "2")
OUTCOME_LABELS = {
    "1": "Victoire domicile",
    "N": "Match nul",
    "2": "Victoire extérieur",
}
