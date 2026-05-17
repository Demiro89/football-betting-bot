import requests
import os
from datetime import datetime

print("=== DIAGNOSTIC TELEGRAM ===")
print(f"Heure : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

print(f"BOT_TOKEN chargé ? {'OUI' if BOT_TOKEN else 'NON'}")
print(f"CHAT_ID chargé ? {'OUI' if CHAT_ID else 'NON'}")

if BOT_TOKEN and CHAT_ID:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    message = f"<b>🧪 TEST DIAGNOSTIC {datetime.now().strftime('%H:%M')}</b>\nSecrets OK ! Le bot fonctionne."
    r = requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"})
    print(f"Status code : {r.status_code}")
    print(f"Réponse Telegram : {r.text}")
else:
    print("❌ Problème : secrets non chargés")
