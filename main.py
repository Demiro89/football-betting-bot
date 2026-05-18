import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import os

# ==================== CONFIG ====================
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

BANKROLL = float(os.getenv("BANKROLL", 1000))
MIN_VALUE_PERCENT = float(os.getenv("MIN_VALUE_PERCENT", 6))
MISE_PERCENT = 0.02

# 12 championnats
LEAGUES = [39, 140, 78, 135, 61, 88, 94, 40, 144, 95, 136, 79]
SEASON = 2025
BOOKMAKER = "unibet"

# ==================== FONCTIONS ====================
def envoyer_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    requests.post(url, json=payload)

def kelly_criterion(value, proba):
    b = value / 100
    p = proba / 100
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, min(kelly, 0.05))

print(f"🚀 BOT PRO DÉMARRÉ - {datetime.now().strftime('%H:%M')} | Bankroll: {BANKROLL}€ | Seuil: {MIN_VALUE_PERCENT}% | 12 championnats")

value_bets = []
odds_data = api_odds()

for league_id in LEAGUES:
    fixtures = api_football("/fixtures", {"league": league_id, "season": SEASON, "status": "NS"})
    past = api_football("/fixtures", {"league": league_id, "season": SEASON, "status": "FT"}) or api_football("/fixtures", {"league": league_id, "season": 2024, "status": "FT"})

    df = pd.DataFrame([{"home": m["teams"]["home"]["name"], "away": m["teams"]["away"]["name"], "hg": m["goals"]["home"] or 0, "ag": m["goals"]["away"] or 0} for m in past])
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
                if value > (MIN_VALUE_PERCENT / 100):
                    mise = round(BANKROLL * kelly_criterion(value, proba), 2)
                    value_bets.append({
                        "Match": f"{home} vs {away}",
                        "Pari": pari,
                        "Cote": cote,
                        "Value %": round(value*100, 1),
                        "Mise €": mise,
                        "Proba %": proba
                    })

# ==================== MESSAGE TELEGRAM ====================
if value_bets:
    df = pd.DataFrame(value_bets).sort_values("Value %", ascending=False)
    message = f"<b>🔥 {len(df)} VALUE BETS DÉTECTÉS</b>\n\n"
    for _, row in df.iterrows():
        message += f"📅 <b>{row['Match']}</b>\n"
        message += f"   🎯 <b>{row['Pari']}</b> @ {row['Cote']} → <b>+{row['Value %']}%</b>\n"
        message += f"   💰 Mise : <b>{row['Mise €']} €</b> | Proba: {row['Proba %']}%\n\n"
    message += f"💼 Bankroll : <b>{BANKROLL} €</b>"
else:
    message = f"<b>ℹ️ Bot exécuté à {datetime.now().strftime('%H:%M')}</b>\n"
    message += f"Aucun value bet > {MIN_VALUE_PERCENT}% trouvé cette heure.\n"
    message += "Le bot tourne correctement 24/7."

envoyer_telegram(message)
print("✅ Message envoyé sur Telegram")
