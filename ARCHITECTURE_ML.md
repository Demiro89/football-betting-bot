# Module ML — Value Betting Coupe du Monde

Application de détection de **value bets** orientée Coupe du Monde, basée sur un
modèle de **Machine Learning** (XGBoost / Random Forest), des **cotes en temps
réel multi-bookmakers** et une interface **Streamlit**. Ce module est
**autonome** : il cohabite avec le bot Poisson historique (`main.py`) sans
interférer.

> ⚠️ **Avertissement honnête.** Les paris sportifs comportent un risque réel de
> perte et **personne ne garantit un « taux de réussite élevé »**. Les
> bookmakers ont des modèles redoutables et la plupart des parieurs perdent. Ce
> projet met en place les **edges réels et prouvés** que les professionnels
> utilisent (line shopping, lignes sharp/soft, CLV) pour maximiser l'espérance
> sur le long terme — pas pour promettre des gains. Pariez de façon responsable.

## Comment on cherche à battre les bookmakers (edges réels)

| Levier | Fichier | Pourquoi ça marche |
|---|---|---|
| **Line shopping** | `live_odds.best_odds` | Prendre systématiquement la meilleure cote parmi tous les books ajoute plusieurs % d'EV, sans pari supplémentaire. L'edge le plus fiable. |
| **Consensus de marché** | `live_odds.consensus_probabilities` | Moyenne dévigée de plusieurs books ≈ vraie probabilité. Détecter un book qui s'en écarte (sharp/soft) est la stratégie pro la plus rentable. |
| **Ensemble ML + marché** | `value.ensemble_probabilities` | On ancre la proba sur le marché (dur à battre) et le ML corrige : moins de faux signaux que le ML seul. |
| **CLV (Closing Line Value)** | `tracking` | Seul indicateur fiable AVANT d'avoir un gros échantillon : un CLV moyen positif prouve qu'on bat le marché. |
| **Kelly fractionné** | `value.kelly_stake` | Dimensionne la mise pour survivre à la variance et aux erreurs d'estimation. |

## Vue d'ensemble

```
                ┌────────────────────────────────────────────────────────┐
                │                     DATA PIPELINE                        │
   Source       │  data.py : ~47 000 matchs internationaux (martj42, CC0) │
   gratuite ───▶│  -> nettoyage, issue 1/N/2, poids du tournoi            │
                └───────────────────────────┬────────────────────────────┘
                                             ▼
                ┌────────────────────────────────────────────────────────┐
                │                FEATURE ENGINEERING                       │
                │  features.py : passage chronologique unique (0 fuite)    │
                │  - Elo (avantage terrain, importance du match)           │
                │  - forme récente (buts ±, points/match)                  │
                │  - repos, terrain neutre, importance compétition         │
                └───────────────────────────┬────────────────────────────┘
                                             ▼
                ┌────────────────────────────────────────────────────────┐
                │                   MODÈLE ML (1/N/2)                      │
                │  model.py : XGBoost multi:softprob (fallback sklearn)    │
                │  + calibration isotone des probabilités                  │
                └───────────────────────────┬────────────────────────────┘
                                             ▼
                ┌─────────────────────────┐   ┌───────────────────────────┐
                │   VALUE BETTING          │   │   BACKTEST (train.py)     │
                │   value.py               │   │   walk-forward, log-loss, │
                │   p_modèle > 1/cote ?    │   │   Brier vs baseline       │
                │   + Kelly fractionné     │   └───────────────────────────┘
                └───────────┬──────────────┘
                            ▼
                ┌────────────────────────────────────────────────────────┐
                │                 INTERFACE (app.py)                       │
                │  Streamlit : matchs, prédictions %, alertes value,       │
                │  mise recommandée selon la bankroll                      │
                └────────────────────────────────────────────────────────┘
```

## Étapes du raisonnement

### 1. Données (gratuites, sans clé)
`worldcup/data.py` télécharge le jeu public **« International football results »**
(martj42, licence CC0) : toutes les sélections nationales, terrains neutres,
type de compétition, de 1872 à aujourd'hui. C'est la source idéale pour la
Coupe du Monde. Le fichier est mis en cache localement (utilisable hors-ligne).

*Enrichissements optionnels* (xG, possession, stats joueurs) : points
d'extension `enrich_with_xg()` prévus pour brancher `soccerdata` (FBref/Understat)
ou API-Football si vous le souhaitez. Le modèle fonctionne sans eux.

### 2. Features (sans fuite de données)
`worldcup/features.py` calcule, **en un seul passage chronologique**, un Elo par
nation (avantage terrain + pondération par l'importance du match et l'ampleur du
score) et des indicateurs de forme récente. Pour chaque match, les features sont
émises **avant** la mise à jour avec le résultat : aucune information future ne
fuite, condition d'un backtest honnête.

### 3. Modèle ML (1 / N / 2)
`worldcup/model.py` entraîne un **XGBoost** multiclasse (`multi:softprob`), avec
repli automatique sur `HistGradientBoostingClassifier` si XGBoost est absent.
Les probabilités sont **calibrées** (isotone) car, pour le value betting, c'est
la qualité des probabilités — pas le simple label — qui détermine la rentabilité.

### 4. Value betting + Kelly
`worldcup/value.py` retire la marge du bookmaker (devigging par normalisation)
pour obtenir la proba « juste » du marché, puis signale une **value** quand
`proba_modèle > 1/cote`. La mise suit le **critère de Kelly fractionné** (1/4 par
défaut) plafonné à 5 % de la bankroll — pour survivre à la variance et aux
erreurs d'estimation.

### 5. Interface Streamlit
`app.py` affiche les prochains matchs, les prédictions en %, les **alertes value
bets** en évidence et la **mise recommandée** selon la bankroll saisie.

## Utilisation

```bash
pip install -r requirements-ml.txt

# 1) Entraîner + valider le modèle (backtest walk-forward affiché)
python train.py

# 2) (optionnel) cotes en direct : clé gratuite sur the-odds-api.com
export THE_ODDS_API_KEY=ta_cle      # ou dans le fichier .env

# 3) Lancer l'interface temps réel
streamlit run app.py
```

Dans l'app : barre latérale → **Source des matchs = Cotes en direct (API)**,
**Auto-refresh** activé. Sans clé, l'app bascule automatiquement sur le CSV
local.

**Mode temps réel.** L'app interroge The Odds API (compétitions configurables
via `SPORT_KEYS`), applique le line shopping et le consensus, puis se rafraîchit
toutes les *N* secondes (`st.fragment`). Les réponses API sont mises en cache
`ODDS_TTL_SECONDS` pour ne pas brûler le quota gratuit (500 requêtes/mois).

Éditez sinon `data/wc2026_fixtures.csv` pour vos matchs et leurs cotes :

```csv
date,home_team,away_team,neutral,tournament,odd_1,odd_N,odd_2
2026-06-11,Mexico,Spain,False,FIFA World Cup,3.40,3.30,2.15
```

Les noms d'équipes doivent correspondre à ceux du jeu de données (ex.
« United States », « South Korea », « IR Iran »).

## Garde-fous (rentabilité, pas illusion)

1. **Validation walk-forward** : `python train.py` compare le modèle à une
   baseline (fréquences de base) en log-loss et score de Brier. *Un modèle qui
   ne bat pas la baseline ne peut pas être rentable.*
2. **Value stricte** : on ne parie que si la proba du modèle dépasse la proba
   implicite du marché (marge retirée).
3. **Kelly fractionné plafonné** : mise prudente, dimensionnée pour la survie.
4. **Suivi du ROI réel** : tant que vous avez < ~30 paris, le résultat n'est pas
   significatif. Si le ROI reste négatif au-delà, arrêtez ou recalibrez.

## Tests

```bash
python -m unittest test_worldcup -v   # tests du module ML (offline)
python -m unittest discover -v        # toute la suite du dépôt
```
