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

LEAGUES = [39, 140, 78, 135, 61]  # Seulement les 5 grands pour les blessures
SEASON = 2025
BOOKMAKER = "unibet"

# ==================== JOUEURS CLÉS (à mettre à jour régulièrement) ====================
KEY_PLAYERS = {
    33: ["Erling Haaland", "Kevin De Bruyne", "Phil Foden"],           # Man City
    40: ["Mohamed Salah", "Darwin Núñez", "Luis Díaz"],                # Liverpool
    50: ["Harry Kane", "Jamal Musiala", "Leroy Sané"],                 # Bayern
    529: ["Kylian Mbappé", "Vinicius Jr", "Jude Bellingham"],          # Real Madrid
    541: ["Robert Lewandowski", "Lamine Yamal", "Raphinha"],           # Barcelona
    489: ["Lautaro Martínez", "Marcus Thuram", "Nicolo Barella"],      # Inter
    496: ["Victor Osimhen", "Khvicha Kvaratskhelia", "Scott McTominay"], # Napoli
    85: ["Ollie Watkins", "Morgan Rogers", "Emiliano Buendía"],        # Aston Villa
    157: ["Alexander Isak", "Anthony Gordon", "Bruno Guimarães"],      # Newcastle
    33: ["Erling Haaland", "Kevin De Bruyne", "Phil Foden"],           # (double pour sécurité)
}

def get_injured_key_players(team_id):
    """Récupère les joueurs clés blessés d'une équipe"""
    url = "https://v3.football.api-sports.io/injuries"
    params = {"team": team_id, "season": SEASON}
    r = requests.get(url, headers={"x-apisports-key": API_FOOTBALL_KEY}, params=params)
    injured = []
    if r.status_code == 200 and r.json().get("response"):
        for injury in r.json()["response"]:
            player_name = injury.get("player", {}).get("name", "")
            if player_name in KEY_PLAYERS.get(team_id, []):
                injured.append(player_name)
    return injured

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

print(f"🚀 BOT PRO xG + BLESSURES DÉMARRÉ - {datetime.now().strftime('%H:%M')} | Bankroll: {BANKROLL}€")

value_bets = []
odds_data = api_odds()

for league_id in LEAGUES:
    fixtures = api_football("/fixtures", {"league": league_id, "season": SEASON, "status": "NS"})

    for f in fixtures:
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]

        # Vérification des blessures
        home_injured = get_injured_key_players(home_id)
        away_injured = get_injured_key_players(away_id)

        # Récupération des stats xG
        home_xg, home_xga = get_team_xg_stats(league_id, home_id, SEASON)
        away_xg, away_xga = get_team_xg_stats(league_id, away_id, SEASON)

        # Ajustement si joueur clé blessé
        lambda_home = (home_xg * 0.7 + home_xga * 0.3) * 1.05
        lambda_away = (away_xg * 0.7 + away_xga * 0.3) * 0.95

        if home_injured:
            lambda_home -= 0.4 * len(home_injured)  # -0.4 but par joueur clé blessé
        if away_injured:
            lambda_away -= 0.4 * len(away_injured)

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
                        "Proba %": proba,
                        "Blessés": f"🏠 {', '.join(home_injured) if home_injured else 'Aucun'} | 🚌 {', '.join(away_injured) if away_injured else 'Aucun'}"
                    })

# ==================== MESSAGE TELEGRAM ====================
if value_bets:
    df = pd.DataFrame(value_bets).sort_values("Value %", ascending=False)
    message = f"<b>🔥 {len(df)} VALUE BETS xG + BLESSURES</b>\n\n"
    for _, row in df.iterrows():
        message += f"📅 <b>{row['Match']}</b>\n"
        message += f"   🎯 <b>{row['Pari']}</b> @ {row['Cote']} → <b>+{row['Value %']}%</b>\n"
        message += f"   💰 Mise : <b>{row['Mise €']} €</b> | Proba: {row['Proba %']}%\n"
        if row['Blessés'] != "🏠 Aucun | 🚌 Aucun":
            message += f"   ⚠️ {row['Blessés']}\n"
        message += "\n"
    message += f"💼 Bankroll : <b>{BANKROLL} €</b>"
else:
    message = f"<b>ℹ️ Bot xG + Blessures exécuté à {datetime.now().strftime('%H:%M')}</b>\n"
    message += f"Aucun value bet > {MIN_VALUE_PERCENT}% trouvé cette heure.\n"
    message += "Le bot tourne correctement 24/7 avec données avancées."

envoyer_telegram(message)
print("✅ Message envoyé sur Telegram (xG + Blessures)")
