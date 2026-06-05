"""ÉTAPE 1bis — Feature engineering (sans fuite de données).

On transforme l'historique brut en une matrice de features exploitables par le
modèle. Tout est calculé en **un seul passage chronologique** : pour chaque
match, on émet d'abord les features (qui ne dépendent QUE du passé), PUIS on met
à jour l'état des équipes avec le résultat. Cela garantit l'absence de fuite
(« data leakage ») — condition indispensable pour un backtest honnête.

Features produites (du point de vue de l'équipe à domicile) :
  - elo_diff        : différence de rating Elo (avantage terrain inclus).
  - home_elo, away_elo
  - form_gf_diff    : diff. de buts marqués/match sur la forme récente.
  - form_ga_diff    : diff. de buts encaissés/match sur la forme récente.
  - form_pts_diff   : diff. de points/match (3/1/0) sur la forme récente.
  - rest_diff       : diff. de jours de repos depuis le dernier match.
  - neutral         : terrain neutre (1) ou non (0).
  - tournament_weight : importance du match.

L'objet `EloFormBuilder` sert à la fois à construire la matrice d'entraînement
et à mémoriser l'état FINAL des équipes pour prédire des matchs futurs.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

# Ordre canonique des colonnes de features (réutilisé à l'entraînement ET à la prédiction).
FEATURE_COLUMNS = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "form_gf_diff",
    "form_ga_diff",
    "form_pts_diff",
    "rest_diff",
    "neutral",
    "tournament_weight",
]

# Encodage des labels pour le modèle multiclasse.
LABEL_TO_INT = {"1": 0, "N": 1, "2": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


@dataclass
class _TeamState:
    """État courant d'une sélection nationale."""
    elo: float = config.ELO_START
    last_date: pd.Timestamp | None = None
    # Forme récente : on garde les FORM_WINDOW derniers (buts marqués, encaissés, points).
    recent: deque = field(default_factory=lambda: deque(maxlen=config.FORM_WINDOW))


class EloFormBuilder:
    """Construit les features Elo + forme et maintient l'état des équipes."""

    def __init__(self) -> None:
        self._state: dict[str, _TeamState] = defaultdict(_TeamState)

    # -- accès / lecture -----------------------------------------------------
    def _form(self, team: str) -> tuple[float, float, float]:
        """Buts marqués/match, encaissés/match, points/match sur la forme récente."""
        st = self._state[team]
        if not st.recent:
            return 1.0, 1.0, 1.0  # valeurs neutres pour une équipe sans historique récent
        gf = np.mean([r[0] for r in st.recent])
        ga = np.mean([r[1] for r in st.recent])
        pts = np.mean([r[2] for r in st.recent])
        return float(gf), float(ga), float(pts)

    def _feature_row(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        neutral: bool,
        weight: float,
    ) -> dict[str, float]:
        """Vecteur de features PRÉ-match (n'utilise que l'état passé)."""
        # Normalisation de la date : les cotes en direct arrivent avec fuseau
        # horaire (tz-aware) alors que l'historique est sans fuseau (tz-naive).
        # On retire le fuseau pour permettre la soustraction des dates.
        date = pd.Timestamp(date)
        if date.tzinfo is not None:
            date = date.tz_localize(None)

        hs, as_ = self._state[home], self._state[away]
        home_adv = 0.0 if neutral else config.ELO_HOME_ADV
        elo_diff = (hs.elo + home_adv) - as_.elo

        h_gf, h_ga, h_pts = self._form(home)
        a_gf, a_ga, a_pts = self._form(away)

        def _rest(st: _TeamState) -> float:
            if st.last_date is None:
                return 30.0  # repos « par défaut » si premier match observé
            return min((date - pd.Timestamp(st.last_date)).days, 365)

        return {
            "elo_diff": elo_diff,
            "home_elo": hs.elo,
            "away_elo": as_.elo,
            "form_gf_diff": h_gf - a_gf,
            "form_ga_diff": h_ga - a_ga,
            "form_pts_diff": h_pts - a_pts,
            "rest_diff": _rest(hs) - _rest(as_),
            "neutral": 1.0 if neutral else 0.0,
            "tournament_weight": float(weight),
        }

    # -- mise à jour ---------------------------------------------------------
    def _update(
        self,
        home: str,
        away: str,
        hg: int,
        ag: int,
        date: pd.Timestamp,
        neutral: bool,
        weight: float,
    ) -> None:
        """Met à jour Elo + forme APRÈS avoir émis les features du match."""
        hs, as_ = self._state[home], self._state[away]
        home_adv = 0.0 if neutral else config.ELO_HOME_ADV

        # Probabilité attendue (logistique Elo classique).
        exp_home = 1.0 / (1.0 + 10 ** (-((hs.elo + home_adv) - as_.elo) / 400.0))

        if hg > ag:
            score_home, hp, ap = 1.0, 3, 0
        elif hg < ag:
            score_home, hp, ap = 0.0, 0, 3
        else:
            score_home, hp, ap = 0.5, 1, 1

        # K pondéré par l'importance du match et l'ampleur du score (méthode Elo football).
        gd = abs(hg - ag)
        gd_mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)
        k = config.ELO_K * weight * gd_mult
        delta = k * (score_home - exp_home)

        hs.elo += delta
        as_.elo -= delta
        hs.recent.append((hg, ag, hp))
        as_.recent.append((ag, hg, ap))
        hs.last_date = as_.last_date = date

    # -- API publique --------------------------------------------------------
    def build_training_matrix(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Parcourt l'historique trié et renvoie (X, y) sans fuite de données."""
        rows: list[dict[str, float]] = []
        labels: list[int] = []
        for r in df.itertuples(index=False):
            feats = self._feature_row(
                r.home_team, r.away_team, r.date, bool(r.neutral), float(r.tournament_weight)
            )
            rows.append(feats)
            labels.append(LABEL_TO_INT[r.result])
            self._update(
                r.home_team, r.away_team, int(r.home_score), int(r.away_score),
                r.date, bool(r.neutral), float(r.tournament_weight),
            )
        X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        y = np.array(labels, dtype=int)
        return X, y

    def features_for_fixture(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        neutral: bool,
        tournament: str,
    ) -> pd.DataFrame:
        """Vecteur de features pour un match FUTUR, à partir de l'état courant.

        À appeler après `build_training_matrix` (qui a « rempli » l'état avec
        tout l'historique). Les équipes inconnues prennent les valeurs par défaut.
        """
        from .data import tournament_weight as _tw

        feats = self._feature_row(home, away, date, neutral, _tw(tournament))
        return pd.DataFrame([feats], columns=FEATURE_COLUMNS)

    def known_teams(self) -> list[str]:
        """Liste triée des sélections disposant d'un rating (pour l'UI)."""
        return sorted(self._state.keys())

    def elo_of(self, team: str) -> float:
        return self._state[team].elo
