import requests
import os
from datetime import datetime

print("=== DIAGNOSTIC TELEGRAM ===")
print(f"Heure : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

print(f"BOT_TOKEN chargé ? {'OUI' if BOT_TOKEN else 'NON'}")
if BOT_TOKEN:
    print(f"Longueur BOT_TOKEN : {len(BOT_TOKEN)} caractères")
    print(f" Début : {BOT_TOKEN[:15]}...")

print(f"CHAT_ID chargé ? {'OUI' if CHAT_ID else 'NON'}")
if CHAT_ID:
    print(f"CHAT_ID : {CHAT_ID}")

# Test envoi Telegram
url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
message = f"<b>🧪 TEST DIAGNOSTIC</b>\nBot lancé à {datetime.now().strftime('%H:%M:%S')}\nVérifions si les secrets fonctionnent..."

payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}

print("Envoi du message vers Telegram...")
r = requests.post(url, json=payload)

print(f"Status code : {r.status_code}")
print(f"Réponse Telegram : {r.text}")

if r.status_code == 200:
    print("✅ Message envoyé avec succès !")
else:
    print("❌ ÉCHEC de l'envoi")
