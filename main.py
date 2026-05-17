import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== CONFIG ====================
API_KEY = os.getenv('API_FOOTBALL_KEY')
if not API_KEY:
    print('❌ Mets ta clé API_FOOTBALL_KEY dans le fichier .env')
    print('Exemple : API_FOOTBALL_KEY=ta_cle_ici')
    exit(1)

LEAGUE_ID = 61      # Ligue 1 France
SEASON = 2025
BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {"x-apisports-key": API_KEY}

def api_get(endpoint, params=None):
    if params is None:
        params = {}
    r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params)
    if r.status_code != 200:
        print(f"❌ Erreur API {r.status_code}")
        print(r.text)
        exit(1)
    return r.json().get("response", [])

print("🔄 Récupération des données Ligue 1...")
fixtures = api_get("/fixtures", {"league": LEAGUE_ID, "season": SEASON, "status": "NS"})
past = api_get("/fixtures", {"league": LEAGUE_ID, "season": SEASON, "status": "FT"})

print(f"✅ {len(fixtures)} matchs à venir | {len(past)} matchs terminés")

def calculate_lambdas(past_matches):
    df = pd.DataFrame([{
        "home_team": m["teams"]["home"]["name"],
        "away_team": m["teams"]["away"]["name"],
        "home_goals": m["goals"]["home"] or 0,
        "away_goals": m["goals"]["away"] or 0
    } for m in past_matches])

    league_home_avg = df["home_goals"].mean()
    league_away_avg = df["away_goals"].mean()

    attack = pd.concat([
        df[["home_team", "home_goals"]].rename(columns={"home_team":"team","home_goals":"goals"}),
        df[["away_team", "away_goals"]].rename(columns={"away_team":"team","away_goals":"goals"})
    ]).groupby("team")["goals"].mean()

    recent_form = {}
    for team in set(df["home_team"]) | set(df["away_team"]):
        team_matches = df[(df["home_team"] == team) | (df["away_team"] == team)].tail(8)
        if len(team_matches) == 0:
            recent_form[team] = 1.0
            continue
        weights = np.exp(np.linspace(-2, 0, len(team_matches)))
        weights /= weights.sum()
        scored = team_matches.apply(lambda r: r["home_goals"] if r["home_team"] == team else r["away_goals"], axis=1)
        recent_form[team] = (scored * weights).mean() / (league_home_avg if team_matches["home_team"].iloc[0] == team else league_away_avg)

    return {
        "league_home_avg": league_home_avg,
        "league_away_avg": league_away_avg,
        "attack": attack.to_dict(),
        "recent_form": recent_form
    }

lambdas_data = calculate_lambdas(past)

def predict(home, away, n_sim=20000):
    h_attack = lambdas_data["attack"].get(home, lambdas_data["league_home_avg"])
    a_attack = lambdas_data["attack"].get(away, lambdas_data["league_away_avg"])
    h_form = lambdas_data["recent_form"].get(home, 1.0)
    a_form = lambdas_data["recent_form"].get(away, 1.0)

    lambda_home = h_attack * a_form * lambdas_data["league_home_avg"] / 1.4
    lambda_away = a_attack * h_form * lambdas_data["league_away_avg"] / 1.4

    hg = poisson.rvs(lambda_home, size=n_sim)
    ag = poisson.rvs(lambda_away, size=n_sim)

    return {
        "match": f"{home} vs {away}",
        "lambda_home": round(lambda_home, 2),
        "lambda_away": round(lambda_away, 2),
        "proba_1": round(100 * np.mean(hg > ag), 1),
        "proba_N": round(100 * np.mean(hg == ag), 1),
        "proba_2": round(100 * np.mean(hg < ag), 1),
        "proba_over_2.5": round(100 * np.mean(hg + ag > 2.5), 1)
    }

print("\n🔥 PRÉDICTIONS PROCHAINS MATCHS LIGUE 1\n")
for f in fixtures[:15]:
    h = f["teams"]["home"]["name"]
    a = f["teams"]["away"]["name"]
    date = f["fixture"]["date"][:16]
    p = predict(h, a)
    print(f"📅 {date} | {p['match']}")
    print(f"   Buts attendus : {p['lambda_home']} - {p['lambda_away']}")
    print(f"   1 : {p['proba_1']}% | N : {p['proba_N']}% | 2 : {p['proba_2']}% | Over 2.5 : {p['proba_over_2.5']}%")
    print("-"*70)
