import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import os
import time

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

# Cache pour les stats xG (évite de recharger à chaque fois)
xg_cache = {}
xg_cache_time = {}

def api_call_with_retry(url, headers, params=None, max_retries=3, timeout=10):
    """Appel API avec retry et timeout"""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:  # Rate limit
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.Timeout:
            print(f"Timeout sur {url}, tentative {attempt + 1}/{max_retries}")
            time.sleep(1)
        except Exception as e:
            print(f"Erreur API: {e}")
            time.sleep(1)
    return None

def get_team_xg_stats(league_id, team_id, season):
    """Récupère les stats xG avec cache"""
    cache_key = f"{league_id}_{team_id}_{season}"
    
    # Si en cache et moins de 6 heures
    if cache_key in xg_cache and (datetime.now() - xg_cache_time.get(cache_key, datetime.min)).seconds < 21600:
        return xg_cache[cache_key]
    
    url = "https://v3.football.api-sports.io/teams/statistics"
    params = {"league": league_id, "team": team_id, "season": season}
    data = api_call_with_retry(url, {"x-apisports-key": API_FOOTBALL_KEY}, params)
    
    if data and data.get("response"):
        stats = data["response"]
        xg_for = float(stats.get("goals", {}).get("for", {}).get("average", 1.3))
        xg_against = float(stats.get("goals", {}).get("against", {}).get("average", 1.3))
        xg_cache[cache_key] = (xg_for, xg_against)
        xg_cache_time[cache_key] = datetime.now()
        return xg_for, xg_against
    
    return 1.3, 1.3

# ==================== FONCTIONS ====================
def envoyer_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    requests.post(url, json=payload, timeout=10)

def kelly_criterion(value, proba):
    b = value / 100
    p = proba / 100
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, min(kelly, 0.05))

print(f"🚀 BOT PRO xG + BLESSURES + LATENCES DÉMARRÉ - {datetime.now().strftime('%H:%M')} | Bankroll: {BANKROLL}€")

value_bets = []
odds_data = api_odds()

for league_id in LEAGUES:
    fixtures = api_football("/fixtures", {"league": league_id, "season": SEASON, "status": "NS"})

    for f in fixtures:
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]

        # Récupération des stats xG (avec cache)
        home_xg, home_xga = get_team_xg_stats(league_id, home_id, SEASON)
        away_xg, away_xga = get_team_xg_stats(league_id, away_id, SEASON)

        # Ajustement si joueur clé blessé
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
    message = f"<b>🔥 {len(df)} VALUE BETS xG + BLESSURES</b>\n\n"
    for _, row in df.iterrows():
        message += f"📅 <b>{row['Match']}</b>\n"
        message += f"   🎯 <b>{row['Pari']}</b> @ {row['Cote']} → <b>+{row['Value %']}%</b>\n"
       
