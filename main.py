import requests
import os
from datetime import datetime

print("=== DIAGNOSTIC FINAL ===")
print(f"Heure : {datetime.now().strftime('%H:%M:%S')}")

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

print(f"BOT_TOKEN : {'✅ chargé' if BOT_TOKEN else '❌ NON chargé'}")
print(f"CHAT_ID   : {'✅ chargé' if CHAT_ID else '❌ NON chargé'}")

if BOT_TOKEN and CHAT_ID:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    message = f"<b>✅ TEST FINAL {datetime.now().strftime('%H:%M')}</b>\nLe bot est maintenant opérationnel !"
    r = requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"})
    print(f"Status Telegram : {r.status_code}")
    print(r.text)
else:
    print("❌ Problème de secrets")
