import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime
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

# Cache
xg_cache = {}
xg_cache_time = {}

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

def api_call_with_retry(url, headers, params=None, max_retries=3, timeout=10):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
        except:
            time.sleep(1)
    return None

def get_team_xg_stats(league_id, team_id, season):
    cache_key = f"{league_id}_{team_id}_{season}"
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

def get_player_workload(player_id, team_id):
    """Récupère la charge de travail d'un joueur (minutes jouées)"""
    url = "https://v3.football.api-sports.io/players"
    params = {"id": player_id, "team": team_id, "season": SEASON}
    data = api_call_with_retry(url, {"x-apisports-key": API_FOOTBALL_KEY}, params)
    
    if data and data.get("response"):
        player = data["response"][0]
        stats = player.get("statistics", [{}])[0]
        minutes = stats.get("games", {}).get("minutes", 0)
        age = player.get("player", {}).get("age", 25)
        return minutes, age
    return 0, 25

def predict_injury_risk(player_id, team_id):
    """Prédit le risque de blessure avec des règles simples"""
    minutes, age = get_player_workload(player_id, team_id)
    risk = 0
    
    # Règle 1: Trop de minutes dans les 10 derniers jours
    if minutes > 300:
        risk += 0.4
    
    # Règle 2: Joueur âgé avec beaucoup de matchs
    if age > 32 and minutes > 200:
        risk += 0.3
    
    # Règle 3: Blessure récente (simplifié)
    if minutes > 400:  # Beaucoup de minutes = risque de fatigue
        risk += 0.2
    
    return min(risk, 0.8)  # Max 80% de risque

def get_injured_and_suspended_players(team_id):
    url = "https://v3.football.api-sports.io/injuries"
    params = {"team": team_id, "season": SEASON}
    r = requests.get(url, headers={"x-apisports-key": API_FOOTBALL_KEY}, params=params)
    injured = []
    suspended = []
    
    if r.status_code == 200 and r.json().get("response"):
        for injury in r.json()["response"]:
            player_name = injury.get("player", {}).get("name", "")
            reason = injury.get("player", {}).get("reason", "").lower()
            
            if player_name in KEY_PLAYERS.get(team_id, []):
                if "suspension" in reason or "suspend" in reason:
                    suspended.append(player_name)
                else:
                    injured.append(player_name)
    
    return injured, suspended

def envoyer_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    requests.post(url, json=payload, timeout=10)

def kelly_criterion_optimized(value, proba, bankroll, max_bet_percent=0.05):
    if value <= 0:
        return 0
    b = value / 100
    p = proba / 100
    q = 1 - p
    kelly_full = (b * p - q) / b
    kelly_half = kelly_full * 0.5
    bet_percent = min(kelly_half, max_bet_percent)
    bet_amount = max(5, round(bankroll * bet_percent))
    max_bet = round(bankroll * max_bet_percent)
    return min(bet_amount, max_bet)

print(f"🚀 BOT PRO FINAL xG + IA BLESSURES DÉMARRÉ - {datetime.now().strftime('%H:%M')} | Bankroll: {BANKROLL}€")

value_bets = []
odds_data = api_odds()

for league_id in LEAGUES:
    fixtures = api_football("/fixtures", {"league": league_id, "season": SEASON, "status": "NS"})

    for f in fixtures:
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]

        home_xg, home_xga = get_team_xg_stats(league_id, home_id, SEASON)
        away_xg, away_xga = get_team_xg_stats(league_id, away_id, SEASON)

        home_injured, home_suspended = get_injured_and_suspended_players(home_id)
        away_injured, away_suspended = get_injured_and_suspended_players(away_id)

        # Prédiction IA des blessures (règles simples)
        home_injury_risk = 0
        away_injury_risk = 0
        
        for player_name in KEY_PLAYERS.get(home_id, []):
            # Simulation: on prend le premier joueur de la liste
            player_id = 1  # À remplacer par le vrai ID du joueur
            risk = predict_injury_risk(player_id, home_id)
            if risk > 0.5:
                home_injury_risk += risk
        
        for player_name in KEY_PLAYERS.get(away_id, []):
            player_id = 1
            risk = predict_injury_risk(player_id, away_id)
            if risk > 0.5:
                away_injury_risk += risk

        lambda_home = (home_xg * 0.7 + home_xga * 0.3) * 1.05
        lambda_away = (away_xg * 0.7 + away_xga * 0.3) * 0.95

        # Ajustement IA
        if home_injury_risk > 0.5:
            lambda_home -= 0.3 * home_injury_risk
        if away_injury_risk > 0.5:
            lambda_away -= 0.3 * away_injury_risk

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
                    mise = kelly_criterion_optimized(value, proba, BANKROLL)
                    value_bets.append({
                        "Match": f"{home} vs {away}",
                        "Pari": pari,
                        "Cote": cote,
                        "Value %": round(value*100, 1),
                        "Mise €": mise,
                        "Proba %": proba
                    })

if value_bets:
    df = pd.DataFrame(value_bets).sort_values("Value %", ascending=False)
    message = f"<b>🔥 {len(df)} VALUE BETS xG + IA BLESSURES</b>\n\n"
    for _, row in df.iterrows():
        message += f"📅 <b>{row['Match']}</b>\n"
        message += f"   🎯 <b>{row['Pari']}</b> @ {row['Cote']} → <b>+{row['Value %']}%</b>\n"
        message += f"   💰 Mise : <b>{row['Mise €']} €</b> | Proba: {row['Proba %']}%\n\n"
    message += f"💼 Bankroll : <b>{BANKROLL} €</b>"
else:
    message = f"<b>ℹ️ Bot xG + IA Blessures exécuté à {datetime.now().strftime('%H:%M')}</b>\n"
    message += f"Aucun value bet > {MIN_VALUE_PERCENT}% trouvé cette heure.\n"
    message += "Le bot tourne correctement 24/7 avec IA."

envoyer_telegram(message)
print("✅ Message envoyé sur Telegram (xG + IA Blessures)")
