"""Module ML de value betting orienté Coupe du Monde.

Sous-modules :
  - config   : paramètres et chemins.
  - data     : récupération des données historiques internationales (gratuites).
  - features : feature engineering (Elo, forme récente, H2H, importance du match).
  - model    : classifieur XGBoost multiclasse 1/N/2 (+ fallback scikit-learn).
  - value    : détection de value bets et mise via Kelly fractionné.
  - fixtures : chargement des matchs à venir et de leurs cotes.
"""

__all__ = ["config", "data", "features", "model", "value", "fixtures"]
