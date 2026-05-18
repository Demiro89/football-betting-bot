import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import os

# ==================== CLÉS ====================
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def envoyer_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    requests.post(url, json=payload)

print(f"🚀 BOT VALUE BETTING DÉMARRÉ - {datetime.now().strftime('%H:%M')}")

LEAGUES = [39, 140, 78, 135, 61, 88, 94, 40]
SEASON = 2025
BOOKMAKER = "unibet"

def api_football(endpoint, params=None):
    if params is None: params = {}
    r = requests.get(f"https://v3.football.api-sports.io{endpoint}", 
                     headers={"x-apisports-key": API_FOOTBALL_KEY}, params=params)
    return r.json().get("response", []) if r.status_code == 200 else []

def api_odds():
    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/soccer_la_liga/soccer_bundesliga/soccer_serie_a/soccer_ligue_one/soccer_eredivisie/soccer_primeira_liga/soccer_championship/odds"
    params = {"apiKey": THE_ODDS_API_KEY, "regions": "eu", "markets": "h2h", "bookmakers": BOOKMAKER, "oddsFormat": "decimal"}
    r = requests.get(url, params=params)
    return r.json() if r.status_code == 200 else []

value_bets = []
odds_data = api_odds()

for league_id in LEAGUES:
    fixtures = api_football("/fixtures", {"league": league_id, "season": SEASON, "status": "NS"})
    past = api_football("/fixtures", {"league": league_id, "season": SEASON, "status": "FT"}) or \
           api_football("/fixtures", {"league": league_id, "season": 2024, "status": "FT"})

    df = pd.DataFrame([{"home": m["teams"]["home"]["name"], "away": m["teams"]["away"]["name"],
                        "hg": m["goals"]["home"] or 0, "ag": m["goals"]["away"] or 0} for m in past])
    
    home_avg = df["hg"].mean() if len(df) > 0 else 1.4
    away_avg = df["ag"].mean() if len(df) > 0 else 1.2

    for f in fixtures:
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]

        lambda_home = home_avg * 1.05
        lambda_away = away_avg * 0.95
        hg_sim = poisson.rvs(lambda_home, size=20000)
        ag_sim = poisson.rvs(lambda_away, size=20000)

        proba1 = round(100 * np.mean(hg_sim > ag_sim), 1)
        probaN = round(100 * np.mean(hg_sim == ag_sim), 1)
        proba2 = round(100 * np.mean(hg_sim < ag_sim), 1)

        cote1 = coteN = cote2 = None
        for event in odds_data:
            if home in str(event.get("home_team")) and away in str(event.get("away_team")):
                for bm in event.get("bookmakers", []):
                    if bm["key"] == BOOKMAKER:
                        outcomes = bm.get("markets", [{}])[0].get("outcomes", [])
                        cote1 = next((o["price"] for o in outcomes if o.get("name") == home), None)
                        coteN = next((o["price"] for o in outcomes if o.get("name") == "Draw"), None)
                        cote2 = next((o["price"] for o in outcomes if o.get("name") == away), None)
                        break
                break

        if cote1 and coteN and cote2:
            for pari, value, cote, proba in [("1", (proba1/100 * cote1) - 1, cote1, proba1),
                                             ("N", (probaN/100 * coteN) - 1, coteN, probaN),
                                             ("2", (proba2/100 * cote2) - 1, cote2, proba2)]:
                if value > 0.06:
                    mise = 20.0
                    value_bets.append({"Match": f"{home} vs {away}", "Pari": pari, "Cote": cote,
                                       "Value %": round(value*100, 1), "Mise €": mise})

# ====================== ENVOI SUR TELEGRAM ======================
if value_bets:
    df = pd.DataFrame(value_bets).sort_values("Value %", ascending=False)
    message = f"<b>🔥 VALUE BETS DÉTECTÉS ({len(df)})</b>\n\n"
    for _, row in df.iterrows():
        message += f"📅 {row['Match']}\n   {row['Pari']} @ {row['Cote']} → +{row['Value %']}%\n   Mise recommandée : {row['Mise €']} €\n\n"
    message += f"💰 Bankroll : 1000 €"
else:
    message = f"<b>ℹ️ Bot exécuté à {datetime.now().strftime('%H:%M')}</b>\nAucun value bet > 6% trouvé cette heure.\nLe bot tourne correctement 24/7."

envoyer_telegram(message)
print("✅ Statut envoyé sur Telegram")
print("✅ Exécution terminée")
