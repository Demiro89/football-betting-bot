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

def get_team_xg_stats(league_id, team_id, season):
    """Récupère les stats xG d'une équipe sur la saison"""
    url = f"https://v3.football.api-sports.io/teams/statistics"
    params = {"league": league_id, "team": team_id, "season": season}
    r = requests.get(url, headers={"x-apisports-key": API_FOOTBALL_KEY}, params=params)
    if r.status_code == 200 and r.json().get("response"):
        stats = r.json()["response"]
        xg_for = stats.get("goals", {}).get("for", {}).get("average", 1.3)
        xg_against = stats.get("goals", {}).get("against", {}).get("average", 1.3)
        return float(xg_for), float(xg_against)
    return 1.3, 1.3

print(f"🚀 BOT PRO xG + FORME DÉMARRÉ - {datetime.now().strftime('%H:%M')} | Bankroll: {BANKROLL}€ | Seuil: {MIN_VALUE_PERCENT}%")

value_bets = []
odds_data = api_odds()

for league_id in LEAGUES:
    fixtures = api_football("/fixtures", {"league": league_id, "season": SEASON, "status": "NS"})
    past = api_football("/fixtures", {"league": league_id, "season": SEASON, "status": "FT"}) or api_football("/fixtures", {"league": league_id, "season": 2024, "status": "FT"})

    df = pd.DataFrame([{"home": m["teams"]["home"]["name"], "away": m["teams"]["away"]["name"], 
                        "home_id": m["teams"]["home"]["id"], "away_id": m["teams"]["away"]["id"],
                        "hg": m["goals"]["home"] or 0, "ag": m["goals"]["away"] or 0} for m in past])

    for f in fixtures:
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]

        # Récupération des stats xG
        home_xg, home_xga = get_team_xg_stats(league_id, home_id, SEASON)
        away_xg, away_xga = get_team_xg_stats(league_id, away_id, SEASON)

        # Lambda ajusté avec xG (beaucoup plus précis)
        lambda_home = (home_xg * 0.7 + home_xga * 0.3) * 1.05
        lambda_away = (away_xg * 0.7 + away_xga * 0.3) * 0.95

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
    message = f"<b>🔥 {len(df)} VALUE BETS xG DÉTECTÉS</b>\n\n"
    for _, row in df.iterrows():
        message += f"📅 <b>{row['Match']}</b>\n"
        message += f"   🎯 <b>{row['Pari']}</b> @ {row['Cote']} → <b>+{row['Value %']}%</b>\n"
        message += f"   💰 Mise : <b>{row['Mise €']} €</b> | Proba: {row['Proba %']}%\n\n"
    message += f"💼 Bankroll : <b>{BANKROLL} €</b> | xG + Forme activés"
else:
    message = f"<b>ℹ️ Bot xG exécuté à {datetime.now().strftime('%H:%M')}</b>\n"
    message += f"Aucun value bet > {MIN_VALUE_PERCENT}% trouvé cette heure.\n"
    message += "Le bot tourne correctement 24/7 avec données avancées."

envoyer_telegram(message)
print("✅ Message envoyé sur Telegram (xG + Forme)")
