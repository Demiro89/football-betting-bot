import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== CONFIG ====================
API_KEY = os.getenv("API_FOOTBALL_KEY")
LEAGUE_ID = 61          # Ligue 1 France
SEASON = 2025
BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {"x-apisports-key": API_KEY}

def api_get(endpoint, params=None):
    if params is None:
        params = {}
    r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params)
    if r.status_code != 200:
        print(f"❌ API Error {r.status_code}")
        print(r.text)
        return []
    return r.json().get("response", [])

# ==================== RÉCUPÉRATION DONNÉES ====================
print("🔄 Récupération des données Ligue 1...")

fixtures = api_get("/fixtures", {"league": LEAGUE_ID, "season": SEASON, "status": "NS"})
past_fixtures = api_get("/fixtures", {"league": LEAGUE_ID, "season": SEASON, "status": "FT"})

print(f"✅ {len(fixtures)} matchs à venir trouvés")
print(f"✅ {len(past_fixtures)} matchs terminés récupérés")

# ==================== CALCUL LAMB DAS ====================
def calculate_lambdas(past_matches):
    if not past_matches:
        return {"league_home_avg": 1.4, "league_away_avg": 1.2, "attack": {}, "recent_form": {}}
    
    df = pd.DataFrame([{
        "home_team": m["teams"]["home"]["name"],
        "away_team": m["teams"]["away"]["name"],
        "home_goals": m["goals"]["home"] or 0,
        "away_goals": m["goals"]["away"] or 0
    } for m in past_matches])

    league_home_avg = df["home_goals"].mean()
    league_away_avg = df["away_goals"].mean()

    attack = pd.concat([
        df[["home_team", "home_goals"]].rename(columns={"home_team":"team", "home_goals":"goals"}),
        df[["away_team", "away_goals"]].rename(columns={"away_team":"team", "away_goals":"goals"})
    ]).groupby("team")["goals"].mean()

    recent_form = {}
    all_teams = set(df["home_team"]) | set(df["away_team"])
    for team in all_teams:
        team_matches = df[(df["home_team"] == team) | (df["away_team"] == team)].tail(8)
        if len(team_matches) == 0:
            recent_form[team] = 1.0
            continue
        weights = np.exp(np.linspace(-2, 0, len(team_matches)))
        weights /= weights.sum()
        scored = team_matches.apply(lambda r: r["home_goals"] if r["home_team"] == team else r["away_goals"], axis=1)
        avg_scored = (scored * weights).mean()
        recent_form[team] = avg_scored / league_home_avg   # simplified

    return {
        "league_home_avg": league_home_avg,
        "league_away_avg": league_away_avg,
        "attack": attack.to_dict(),
        "recent_form": recent_form
    }

lambdas_data = calculate_lambdas(past_fixtures)

# ==================== PRÉDICTION ====================
def predict_match(home_team, away_team, n_sim=20000):
    h_attack = lambdas_data["attack"].get(home_team, lambdas_data.get("league_home_avg", 1.4))
    a_attack = lambdas_data["attack"].get(away_team, lambdas_data.get("league_away_avg", 1.2))
    h_form = lambdas_data["recent_form"].get(home_team, 1.0)
    a_form = lambdas_data["recent_form"].get(away_team, 1.0)

    lambda_home = h_attack * a_form * 1.35
    lambda_away = a_attack * h_form * 1.25

    hg = poisson.rvs(lambda_home, size=n_sim)
    ag = poisson.rvs(lambda_away, size=n_sim)

    return {
        "match": f"{home_team} vs {away_team}",
        "lambda_home": round(lambda_home, 2),
        "lambda_away": round(lambda_away, 2),
        "proba_1": round(100 * np.mean(hg > ag), 1),
        "proba_N": round(100 * np.mean(hg == ag), 1),
        "proba_2": round(100 * np.mean(hg < ag), 1),
        "proba_over_2.5": round(100 * np.mean(hg + ag > 2.5), 1)
    }

# ==================== AFFICHAGE ====================
print("\n🔥 PRÉDICTIONS DES PROCHAINS MATCHS DE LIGUE 1\n")
for fixture in fixtures[:12]:
    home = fixture["teams"]["home"]["name"]
    away = fixture["teams"]["away"]["name"]
    date = fixture["fixture"]["date"][:16].replace("T", " ")
    pred = predict_match(home, away)
    print(f"📅 {date} | {pred['match']}")
    print(f"   Buts attendus : {pred['lambda_home']} - {pred['lambda_away']}")
    print(f"   1 : {pred['proba_1']}% | N : {pred['proba_N']}% | 2 : {pred['proba_2']}% | Over 2.5 : {pred['proba_over_2.5']}%")
    print("-" * 65)

print("\n✅ Bot prêt ! Prochaine étape : ajouter les cotes FDJ et le calcul de value.")