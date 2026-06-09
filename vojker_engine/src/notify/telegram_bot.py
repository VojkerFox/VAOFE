import requests
import os

class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def format_signal(self, pair, signal, volume, delta, confidence=85):
        side = "OSTO (LONG)" if signal == 1.0 else "MYYNTI (SHORT)"
        message = (
            f"🚨 *SIGNAALI: {pair}*\n\n"
            f"Direction: {side}\n"
            f"Confidence: {confidence}%\n"
            f"Volume: {volume}\n"
            f"Delta: {delta}\n\n"
            f"💡 *SUOSITUS:*\n"
            f"Kaupan koko: 0.10 lot\n"
            f"Pysäytys (SL): 15 pistettä\n"
            f"Tavoite (TP): 30 pistettä"
        )
        return message

    def send_alert(self, message):
        try:
            params = {'chat_id': self.chat_id, 'text': message, 'parse_mode': 'Markdown'}
            response = requests.get(self.base_url, params=params)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram-virhe: {e}")
            return False