import os
from datetime import datetime

print("=== NOUVELLE VERSION V4 ===")
print(f"Exécution à : {datetime.now().strftime('%H:%M:%S')}")
print("Si tu vois ce message, le fichier a bien été mis à jour !")

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if BOT_TOKEN and CHAT_ID:
    import requests
    message = f"<b>✅ NOUVELLE VERSION V4</b>\nBot mis à jour à {datetime.now().strftime('%H:%M')}\nLe vrai bot est prêt à être activé."
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"})
    print("Message V4 envoyé sur Telegram")
else:
    print("Problème de secrets")
