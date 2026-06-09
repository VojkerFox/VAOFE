import os
import requests
from dotenv import load_dotenv

# Ladataan muuttujat .env-tiedostosta
load_dotenv()

token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

message = "VojkerEngine: Yhteys testattu! Järjestelmä on valmis toimintaan."
url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"

print(f"Lähetetään viestiä osoitteeseen: {url}")
response = requests.get(url)

if response.status_code == 200:
    print("✅ Viesti lähetetty onnistuneesti! Tarkista Telegram.")
else:
    print(f"❌ Virhe: {response.status_code} - {response.text}")