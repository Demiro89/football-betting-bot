#!/usr/bin/env python3
"""Entraînement + évaluation du modèle ML Coupe du Monde.

Usage :
    python train.py            # backtest walk-forward PUIS entraînement final + sauvegarde
    python train.py --no-eval  # entraîne et sauvegarde directement
    python train.py --eval-only

Le backtest walk-forward (aucune fuite de données) entraîne sur le passé et
teste sur les matchs les plus récents. On compare le modèle à une **baseline**
(fréquences de base des issues) via le log-loss et le score de Brier. Règle
d'or : un modèle qui ne bat pas la baseline ne peut pas être rentable — ne
pariez pas avec.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from worldcup import config, data
from worldcup.features import LABEL_TO_INT, EloFormBuilder
from worldcup.model import MatchPredictor, _base_estimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train")


def _multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Score de Brier multiclasse (moyenne des Brier one-vs-rest). Plus bas = mieux."""
    scores = []
    for c in range(proba.shape[1]):
        scores.append(brier_score_loss((y_true == c).astype(int), proba[:, c]))
    return float(np.mean(scores))


def walk_forward_eval(df: pd.DataFrame) -> None:
    """Entraîne sur les matchs anciens, teste sur les plus récents (sans fuite)."""
    n = len(df)
    split = int(n * (1 - config.TEST_FRACTION))
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    log.info("Walk-forward : %d matchs d'entraînement, %d de test (depuis %s).",
             len(train_df), len(test_df), test_df["date"].min().date())

    # Features SANS fuite : un seul builder qui voit le train, puis génère le test
    # match par match en se mettant à jour au fur et à mesure (comme en réel).
    builder = EloFormBuilder()
    X_train, y_train = builder.build_training_matrix(train_df)

    base = _base_estimator()
    from sklearn.calibration import CalibratedClassifierCV
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_train, y_train)

    # Génération « en ligne » du jeu de test : on prédit AVANT de mettre à jour l'état.
    rows, y_test = [], []
    for r in test_df.itertuples(index=False):
        feats = builder.features_for_fixture(
            r.home_team, r.away_team, r.date, bool(r.neutral), r.tournament
        )
        rows.append(feats.iloc[0])
        y_test.append(LABEL_TO_INT[r.result])
        builder._update(r.home_team, r.away_team, int(r.home_score), int(r.away_score),
                        r.date, bool(r.neutral), float(r.tournament_weight))
    X_test = pd.DataFrame(rows).reset_index(drop=True)
    y_test = np.array(y_test)

    proba = clf.predict_proba(X_test)
    preds = proba.argmax(axis=1)

    # Baseline : fréquences de base des issues estimées sur le train.
    base_freq = np.bincount(y_train, minlength=3) / len(y_train)
    base_proba = np.tile(base_freq, (len(y_test), 1))

    model_ll = log_loss(y_test, proba, labels=[0, 1, 2])
    base_ll = log_loss(y_test, base_proba, labels=[0, 1, 2])
    model_brier = _multiclass_brier(y_test, proba)
    base_brier = _multiclass_brier(y_test, base_proba)
    acc = accuracy_score(y_test, preds)

    print("\n" + "=" * 60)
    print("  BACKTEST WALK-FORWARD")
    print("=" * 60)
    print(f"  Matchs testés      : {len(y_test)}")
    print(f"  Exactitude (top-1) : {acc:.3f}")
    print(f"  Log-loss  modèle   : {model_ll:.4f}   | baseline : {base_ll:.4f}")
    print(f"  Brier     modèle   : {model_brier:.4f}   | baseline : {base_brier:.4f}")
    verdict = "✅ bat la baseline" if model_ll < base_ll else "❌ NE bat PAS la baseline"
    print(f"  Verdict            : {verdict}")
    if model_ll >= base_ll:
        print("  ⚠️  Modèle non exploitable en l'état — ne pariez pas avec.")
    print("=" * 60 + "\n")


def train_final(df: pd.DataFrame) -> None:
    """Entraîne sur tout l'historique et sauvegarde le modèle pour l'app."""
    predictor = MatchPredictor().fit(df, calibrate=True)
    predictor.save()
    log.info("Modèle final prêt : %s", config.MODEL_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement modèle ML Coupe du Monde")
    parser.add_argument("--no-eval", action="store_true", help="sauter le backtest")
    parser.add_argument("--eval-only", action="store_true", help="backtest sans sauvegarde")
    parser.add_argument("--min-year", type=int, default=None, help="année min des données")
    args = parser.parse_args()

    df = data.load_results(min_year=args.min_year)

    if not args.no_eval:
        walk_forward_eval(df)
    if not args.eval_only:
        train_final(df)


if __name__ == "__main__":
    main()
