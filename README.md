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

Variables optionnelles : `BANKROLL` (defaut 1000), `MIN_VALUE_PERCENT` (defaut 6),
`KELLY_FRACTION` (defaut 0.25), `MAX_BET_FRACTION` (defaut 0.05),
`MIN_BET` (defaut 5), `DAYS_AHEAD` (defaut 3), `BOOKMAKER` (defaut unibet),
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

## Tests

```bash
python -m unittest discover -v
```

La suite (`test_bot.py`) couvre le modele, les cotes, le Kelly, le cache,
la couche HTTP et le backtest. Tous les appels reseau sont simules. Les
tests tournent automatiquement a chaque push (workflow `tests.yml`).

## Fonctionnement 24/7

Le workflow `.github/workflows/bot-value.yml` execute le bot en continu via
GitHub Actions (cron toutes les 4h par defaut). Renseignez les cles dans
**Settings -> Secrets and variables -> Actions**.

Le bot est concu pour tourner souvent sans spammer :

- **Deduplication** : un meme pari n'est notifie qu'une fois par fenetre
  `DEDUP_HOURS` (re-notifie seulement si la value progresse nettement).
- **Mode silencieux** (`TELEGRAM_QUIET`) : aucun message quand il n'y a rien
  de nouveau, plus un unique heartbeat « bot actif » par jour.
- **Alertes de panne** : message Telegram si aucune cote n'est recuperee
  (cle/quota) ou en cas d'erreur d'execution.
- L'etat (cache, paris notifies, `bets_log.csv`) persiste entre les runs via
  `actions/cache`.

**Quotas** : The Odds API gratuit = 500 requetes/mois. A 6 runs/jour x ~10
ligues, le quota gratuit est depasse — repassez le cron a `0 9 * * *`
(1 run/jour), reduisez la liste `LEAGUES`, ou prenez un palier payant.

GitHub desactive les workflows planifies apres 60 jours d'inactivite du depot.

Ne committez jamais vos cles d'API dans le depot (`.env` est ignore par git).

## Avertissement

Les paris sportifs comportent un risque de perte. Le modele n'a pas ete valide
par backtest : testez en simulation et suivez le ROI (`bets_log.csv`) avant tout
engagement reel.
