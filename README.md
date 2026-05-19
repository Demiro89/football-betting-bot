# Football Betting Bot

Bot de value betting football : modele de Poisson / Dixon-Coles confronte aux
cotes d'un bookmaker (via The Odds API), avec notifications Telegram.

## Configuration

Variables d'environnement requises (voir `.env.example`) :

| Variable | Description |
|---|---|
| `API_FOOTBALL_KEY` | Cle API-Football (api-sports.io) |
| `THE_ODDS_API_KEY` | Cle The Odds API |
| `BOT_TOKEN` | Token du bot Telegram |
| `CHAT_ID` | ID du chat Telegram |

Variables optionnelles : `BANKROLL` (defaut 100), `MIN_VALUE_PERCENT` (defaut 6),
`KELLY_FRACTION` (defaut 0.25), `MAX_BET_FRACTION` (defaut 0.05),
`MIN_BET` (defaut 1), `DAYS_AHEAD` (defaut 3), `BOOKMAKER` (defaut unibet),
`ENABLE_TOTALS` (defaut false ; active le marche Over/Under mais double le
cout The Odds API par requete), `TELEGRAM_QUIET` (defaut true ; aucun message
quand il n'y a pas de nouveau pari), `DEDUP_HOURS` (defaut 24 ; fenetre pendant
laquelle un meme pari n'est notifie qu'une fois).

## Lancer en local

```bash
pip install -r requirements.txt
cp .env.example .env   # puis renseignez vos cles
python main.py
```

## Backtest

Avant de parier en reel, evaluez la qualite du modele :

```bash
python backtest.py [saison] [league_id ...]
```

Le backtest rejoue les saisons passees en walk-forward (aucune fuite de
donnees) et compare le modele a une baseline de frequences de base
(score de Brier, log-loss, calibration). Un modele qui ne bat pas la
baseline ne peut pas etre rentable.

## Suivi du ROI

En argent reel, mesurez vos resultats reels. Apres chaque pari place, ajoutez
une ligne a `docs/resultats.csv` (modifiable directement sur GitHub via l'icone
crayon) :

```
date,match,pari,cote,mise,resultat
2026-05-20,Arsenal vs Chelsea,1,2.10,5,G
```

`resultat` : `G` = gagne, `P` = perdu, `A` = annule/rembourse, vide = en attente.

Pour afficher le bilan : `python roi.py`, ou onglet **Actions** ->
workflow **« Bilan ROI »** -> **Run workflow**.

Tant que vous avez moins de ~30 paris, le resultat n'est pas significatif.
Si le ROI reste negatif au-dela, le modele perd de l'argent : arretez ou
recalibrez.

## Tableau de bord mobile

Un tableau de bord web (dossier `docs/`) affiche le bilan, la courbe de
bankroll et l'historique des paris, optimise pour mobile.

Activation unique : **Settings -> Pages -> Source : Deploy from a branch ->
Branch `main` / dossier `/docs` -> Save**. Le tableau de bord est alors servi
sur `https://<utilisateur>.github.io/football-betting-bot/`.

Il se met a jour automatiquement a chaque modification de `docs/resultats.csv`.
Sur mobile, « Ajouter a l'ecran d'accueil » lui donne l'aspect d'une appli.
Les alertes restent envoyees par Telegram.

## Tests

```bash
python -m unittest discover -v
```

La suite (`test_bot.py`) couvre le modele, les cotes, le Kelly, le cache,
la couche HTTP et le backtest. Tous les appels reseau sont simules. Les
tests tournent automatiquement a chaque push (workflow `tests.yml`).

## Fonctionnement autonome

Le workflow `.github/workflows/bot-value.yml` execute le bot via GitHub Actions,
**2 fois par jour** (08h et 20h UTC), sans aucune machine a laisser allumee.
Renseignez les cles dans **Settings -> Secrets and variables -> Actions**.

Le bot tourne sans intervention et sans spammer :

- **Deduplication** : un meme pari n'est notifie qu'une fois par fenetre
  `DEDUP_HOURS` (re-notifie seulement si la value progresse nettement).
- **Mode silencieux** (`TELEGRAM_QUIET`) : aucun message quand il n'y a rien
  de nouveau, plus un unique heartbeat « bot actif » par jour.
- **Alertes de panne** : message Telegram si les cotes sont inaccessibles
  (cle/quota) ou en cas d'erreur d'execution. Une simple absence de match
  (hors-saison) ne declenche pas de fausse alerte.
- L'etat (cache, paris notifies, `bets_log.csv`) persiste entre les runs via
  `actions/cache`.

**Quotas** — la configuration par defaut tient dans les paliers gratuits :

- The Odds API : 5 ligues x 2 runs x ~31 jours = ~310 requetes/mois (gratuit = 500).
- API-Football : cache des stats de 72h => peu d'appels par jour (gratuit = 100/jour).

Pour analyser plus de championnats ou tourner plus souvent, decommentez des
ligues dans `LEAGUES` (`main.py`) et/ou modifiez le cron — cela demande un
plan The Odds API payant.

GitHub desactive les workflows planifies apres 60 jours d'inactivite du depot.

Ne committez jamais vos cles d'API dans le depot (`.env` est ignore par git).

## Avertissement

Les paris sportifs comportent un risque de perte. Le modele n'a pas ete valide
par backtest : testez en simulation et suivez le ROI (`bets_log.csv`) avant tout
engagement reel.
