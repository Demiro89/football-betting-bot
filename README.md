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
cout The Odds API par requete).

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

## Automatisation

Le workflow `.github/workflows/bot-value.yml` execute le bot via GitHub Actions.
Renseignez les cles dans **Settings -> Secrets and variables -> Actions**.

Ne committez jamais vos cles d'API dans le depot (`.env` est ignore par git).

## Avertissement

Les paris sportifs comportent un risque de perte. Le modele n'a pas ete valide
par backtest : testez en simulation et suivez le ROI (`bets_log.csv`) avant tout
engagement reel.
