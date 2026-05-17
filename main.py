import requests
import os
from datetime import datetime

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

print(f"🚀 Test Telegram - {datetime.now().strftime('%H:%M:%S')}")
print(f"BOT_TOKEN trouvé : {'Oui' if BOT_TOKEN else 'NON (problème secret)'}")
print(f"CHAT_ID trouvé : {'Oui' if CHAT_ID else 'NON (problème secret)'}")

def test_telegram():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    message = f"<b>✅ TEST BOT</b>\nBot démarré avec succès à {datetime.now().strftime('%H:%M')}\nTout semble OK !"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    
    r = requests.post(url, json=payload)
    if r.status_code == 200:
        print("✅ Message envoyé avec succès sur Telegram")
    else:
        print(f"❌ Erreur Telegram : {r.status_code}")
        print(r.text)

test_telegram()
print("Test terminé")
