"""ÉTAPE 2 — Modèle de Machine Learning.

Classifieur multiclasse prédisant les trois issues d'un match : 1 / N / 2.

Choix techniques :
  - XGBoost (`multi:softprob`) si disponible, sinon repli sur
    `HistGradientBoostingClassifier` de scikit-learn — le code reste 100 %
    fonctionnel sans XGBoost.
  - **Calibration des probabilités** (isotone) : un classifieur précis n'est
    pas forcément bien *calibré*. Or, pour le value betting, ce sont les
    probabilités — pas le label — qui comptent. On calibre donc sur un
    jeu de validation dédié.
  - Persistance via joblib (modèle + builder Elo + métadonnées).

On expose une classe `MatchPredictor` qui encapsule features + modèle calibré.
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config
from .features import FEATURE_COLUMNS, INT_TO_LABEL, EloFormBuilder

log = logging.getLogger("worldcup.model")

try:  # XGBoost est préféré mais optionnel.
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover - dépend de l'environnement
    _HAS_XGB = False


def _base_estimator():
    """Estimateur de base (non calibré)."""
    if _HAS_XGB:
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.0,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    log.warning("XGBoost indisponible — repli sur HistGradientBoostingClassifier.")
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=400,
        l2_regularization=1.0,
        random_state=config.RANDOM_STATE,
    )


class MatchPredictor:
    """Encapsule le builder de features et le classifieur calibré."""

    def __init__(self) -> None:
        self.builder = EloFormBuilder()
        self.clf: CalibratedClassifierCV | None = None
        self.classes_ = np.array([0, 1, 2])  # 0=1, 1=N, 2=2
        self.metadata: dict = {}

    # -- entraînement --------------------------------------------------------
    def fit(self, df: pd.DataFrame, calibrate: bool = True) -> "MatchPredictor":
        """Entraîne le modèle sur l'historique complet.

        Le builder Elo est « rempli » par tout l'historique : son état final
        sert ensuite à prédire les matchs à venir.
        """
        X, y = self.builder.build_training_matrix(df)

        base = _base_estimator()
        if calibrate:
            # Calibration isotone via validation croisée légère. Un modèle
            # précis n'est pas forcément bien calibré ; pour le value betting,
            # ce sont les probabilités qui comptent.
            self.clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        else:
            self.clf = base
        self.clf.fit(X, y)

        self.metadata = {
            "n_train": int(len(X)),
            "backend": "xgboost" if _HAS_XGB else "sklearn-histgb",
            "calibrated": bool(calibrate),
            "last_match_date": str(df["date"].max().date()),
            "features": FEATURE_COLUMNS,
        }
        log.info("Modèle entraîné sur %d matchs (%s).", len(X), self.metadata["backend"])
        return self

    # -- prédiction ----------------------------------------------------------
    def predict_proba_features(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilités (n, 3) pour des features déjà calculées."""
        if self.clf is None:
            raise RuntimeError("Modèle non entraîné/chargé.")
        proba = self.clf.predict_proba(X[FEATURE_COLUMNS])
        return proba

    def predict_match(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
    ) -> dict[str, float]:
        """Probabilités {'1','N','2'} pour un match à venir."""
        X = self.builder.features_for_fixture(home, away, date, neutral, tournament)
        proba = self.predict_proba_features(X)[0]
        return {INT_TO_LABEL[i]: float(proba[i]) for i in range(3)}

    # -- persistance ---------------------------------------------------------
    def save(self, path=None) -> None:
        path = path or config.MODEL_FILE
        joblib.dump(
            {"clf": self.clf, "builder": self.builder, "metadata": self.metadata},
            path,
        )
        log.info("Modèle sauvegardé : %s", path)

    @classmethod
    def load(cls, path=None) -> "MatchPredictor":
        path = path or config.MODEL_FILE
        blob = joblib.load(path)
        obj = cls()
        obj.clf = blob["clf"]
        obj.builder = blob["builder"]
        obj.metadata = blob.get("metadata", {})
        return obj
