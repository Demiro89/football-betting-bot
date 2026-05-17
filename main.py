import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
import time
from datetime import datetime
import os

# ==================== CLÉS ====================
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

BANKROLL = 1000.0
MISE_PERCENT = 0.02

def envoyer_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload)
        print("✅ Message envoyé sur Telegram")
    except Exception as e:
        print("❌ Erreur Telegram :", e)

print(f"🚀 BOT LANCÉ - {datetime.now().strftime('%Y-%m-%d %H:%M')}")

LEAGUES = [39, 140, 78, 135, 61, 88, 94, 40]
SEASON = 2025
BOOKMAKER = "unibet"

# ... (le reste du code reste identique jusqu'à la fin)

# ====================== FIN DU BOT ======================
if value_bets:
    df = pd.DataFrame(value_bets).sort_values("Value %", ascending=False)
    message = f"<b>🔥 VALUE BETS DÉTECTÉS ({len(df)})</b>\n\n"
    for _, row in df.iterrows():
        message += f"📅 {row['Match']}\n   {row['Pari']} @ {row['Cote']} → +{row['Value %']}%\n   Mise : {row['Mise €']} €\n\n"
    message += f"💰 Bankroll : {BANKROLL:.2f} €"
else:
    message = f"<b>ℹ️ Bot exécuté à {datetime.now().strftime('%H:%M')}</b>\nAucun value bet > 6% trouvé cette heure.\n{len(fixtures) if 'fixtures' in locals() else 0} matchs analysés."

envoyer_telegram(message)
print("✅ Statut envoyé sur Telegram")
