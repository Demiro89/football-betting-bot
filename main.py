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

# Paramètres personnalisables
BANKROLL = float(os.getenv("BANKROLL", 1000))           # Bankroll de départ
MIN_VALUE_PERCENT = float(os.getenv("MIN_VALUE_PERCENT", 6))  # Seuil minimum
MISE_PERCENT = 0.02                                     # % de bankroll par mise (conservateur)

LEAGUES = [39, 140, 78, 135, 61, 88, 94, 40, 61, 144]   # + Ligue 2 + Championship
SEASON = 2025
BOOKMAKER = "unibet"

# ==================== FONCTIONS ====================
def envoyer_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    requests.post(url, json=payload)

def kelly_criterion(value, proba):
    """Kelly Criterion simplifié"""
    b = value / 100
    p = proba / 100
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, min(kelly, 0.05))  # Max 5% de bankroll

print(f"🚀 BOT PRO DÉMARRÉ - {datetime.now().strftime('%H:%M')} | Bankroll: {BANKROLL}€ | Seuil: {MIN_VALUE_PERCENT}%")

# ... (le reste du code de détection reste identique)

value_bets = []
odds_data = api_odds()

for league_id in LEAGUES:
    # ... (même logique que avant)

    if cote1 and coteN and cote2:
        for pari, value, cote, proba in [...]:
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

# ==================== MESSAGE TELEGRAM AMÉLIORÉ ====================
if value_bets:
    df = pd.DataFrame(value_bets).sort_values("Value %", ascending=False)
    message = f"<b>🔥 {len(df)} VALUE BETS DÉTECTÉS</b>\n\n"
    for _, row in df.iterrows():
        message += f"📅 <b>{row['Match']}</b>\n"
        message += f"   🎯 <b>{row['Pari']}</b> @ {row['Cote']} → <b>+{row['Value %']}%</b>\n"
        message += f"   💰 Mise recommandée : <b>{row['Mise €']} €</b> | Proba: {row['Proba %']}%\n\n"
    message += f"💼 Bankroll actuel : <b>{BANKROLL} €</b>"
else:
    message = f"<b>ℹ️ Bot exécuté à {datetime.now().strftime('%H:%M')}</b>\n"
    message += f"Aucun value bet > {MIN_VALUE_PERCENT}% trouvé cette heure.\n"
    message += "Le bot tourne correctement 24/7."

envoyer_telegram(message)
print("✅ Message envoyé sur Telegram")
