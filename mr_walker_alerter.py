import os
import time
import requests
import MetaTrader5 as mt5
from dotenv import load_dotenv

# --- LADATAAN YMPÄRISTÖMUUTTUJAT .ENV TIEDOSTOSTA ---
load_dotenv()

# Haetaan tunnukset tismalleen sinun .env-tiedostosi avaimilla
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
raw_chat_id = os.getenv("TELEGRAM_CHAT_ID")

# SIISTIMINEN: Poistetaan heittomerkit (') ja mahdolliset lainausmerkit tai välilyönnit kooditasolla
TELEGRAM_CHAT_ID = raw_chat_id.strip("'\" ") if raw_chat_id else None

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("❌ VIRHE: TELEGRAM_BOT_TOKEN tai TELEGRAM_CHAT_ID puuttuu .env-tiedostosta!")
    exit(1)

LEVELS = {
    "EURUSD": {"symbol": "EURUSD", "buy_above": 1.15906, "sell_below": 1.15900},
    "GBPUSD": {"symbol": "GBPUSD", "buy_above": 1.34175, "sell_below": 1.33230},
    "AUDUSD": {"symbol": "AUDUSD", "buy_above": 0.70555, "sell_below": 0.69780},
    "USDCAD": {"symbol": "USDCAD", "buy_above": 1.40250, "sell_below": 1.39290},
    "EURAUD": {"symbol": "EURAUD", "buy_above": 1.64580, "sell_below": 1.64050},
    "GBPCHF": {"symbol": "GBPCHF", "buy_above": 1.07010, "sell_below": 1.06455},
    "GBPJPY": {"symbol": "GBPJPY", "buy_above": 214.990, "sell_below": 213.830},
    "GBPAUD": {"symbol": "GBPAUD", "buy_above": 1.91220, "sell_below": 1.90050},
    "GBPCAD": {"symbol": "GBPCAD", "buy_above": 1.87500, "sell_below": 1.86505},
    "EURCAD": {"symbol": "EURCAD", "buy_above": 1.61775, "sell_below": 1.61025},
    "AUDJPY": {"symbol": "AUDJPY", "buy_above": 113.188, "sell_below": 112.020},
    "KULTA":  {"symbol": "XAUUSD", "buy_above": 4247.08, "sell_below": 4023.80}, 
    "XRPUSD": {"symbol": "XRPUSD", "buy_above": 1.1480,  "sell_below": 1.0865},
    "BTC":    {"symbol": "BTCUSD", "buy_above": 64172.10, "sell_below": 60713.41}
}

alert_history = {key: {"buy_triggered": False, "sell_triggered": False} for key in LEVELS}

def send_telegram_alert(display_name, order_type, price, target_level):
    emoji = "🟢" if "OSTO" in order_type else "🔴"
    
    message = (
        f"🦅 *MR. WALKER ML – REALTIME ALERT* 🦅\n"
        f"-----------------------------------------\n"
        f"📊 *Instrument:* {display_name}\n"
        f"{emoji} *Action:* {order_type}\n"
        f"💵 *Current Price:* `{price}`\n"
        f"🎯 *Triggered Level:* `{target_level}`\n"
        f"-----------------------------------------\n"
        f"⚠️ *REMINDER – RUN YOUR CHECKLIST:*\n"
        f"1. High Impact News upcoming?\n"
        f"2. Price at true S/R boundary?\n"
        f"3. Wait for M15 Break & Retest confirmation!\n"
        f"4. Is there enough room to the next H1 level?"
    )
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"✅ Telegram-hälytys lähetetty parille {display_name}")
        else:
            print(f"❌ Telegram-virhe: {response.text}")
    except Exception as e:
        print(f"❌ Yhteysvirhe Telegramiin: {e}")

def main():
    if not mt5.initialize():
        print("MT5 alustus epäonnistui.")
        return
    
    print("Mr. Walker ML Alerter aktivoitu live-Telegram -syötteellä (Kooditason .env-fiksi käytössä).")
    
    try:
        while True:
            for name, config in LEVELS.items():
                mt5_symbol = config["symbol"]
                tick = mt5.symbol_info_tick(mt5_symbol)
                if tick is None:
                    continue
                
                current_price = tick.last if tick.last != 0 else tick.bid
                
                # OSTO
                if current_price > config["buy_above"]:
                    if not alert_history[name]["buy_triggered"]:
                        send_telegram_alert(name, "OSTO (Buy Above)", current_price, config["buy_above"])
                        alert_history[name]["buy_triggered"] = True
                else:
                    alert_history[name]["buy_triggered"] = False
                
                # MYYNTI
                if current_price < config["sell_below"]:
                    if not alert_history[name]["sell_triggered"]:
                        send_telegram_alert(name, "MYYNTI (Sell Below)", current_price, config["sell_below"])
                        alert_history[name]["sell_triggered"] = True
                else:
                    alert_history[name]["sell_triggered"] = False
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nHälytysjärjestelmä suljettu manuaalisesti.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()