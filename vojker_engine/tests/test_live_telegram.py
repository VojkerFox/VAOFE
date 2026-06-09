import sys
import os
from dotenv import load_dotenv

# Lisätään polku
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from src.notify.telegram_bot import TelegramNotifier

def test_live_message():
    notifier = TelegramNotifier()
    print("Yritetään lähettää testiä...")
    success = notifier.send_alert("VojkerEngine: Testiviesti - Järjestelmä valmis!")
    if success:
        print("Testiviesti lähetetty onnistuneesti!")
    else:
        print("Testiviesti epäonnistui. Tarkista .env (TOKEN ja CHAT_ID).")

if __name__ == "__main__":
    test_live_message()