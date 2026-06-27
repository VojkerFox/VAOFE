import time
import MetaTrader5 as mt5
import requests
import numpy as np  # TÄMÄ RIVI PUUTTUI!

# ASETUKSET
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "XAUUSD"]
TIMEFRAMES = {
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}
TOKEN = "8658806596:AAH3jFlP7LKuHY8wMXBt02kD9UMC9SacZRI"
CHAT_ID = "260783230"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={msg}"
    requests.get(url)

def check_structure():
    if not mt5.initialize(): return
    for symbol in PAIRS:
        for name, tf in TIMEFRAMES.items():
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, 50)
            if rates is None or len(rates) < 2: continue
            
            close = rates[-1]['close']
            # Nyt np.max toimii, koska numpy on tuotu
            hh = np.max([r['high'] for r in rates[:-1]])
            ll = np.min([r['low'] for r in rates[:-1]])
            
            if close > hh:
                send_telegram(f"🚨 {symbol} {name} HH BREAK: Price {close:.5f} > {hh:.5f}")
                time.sleep(1) # Estetään Telegram-spammi
            elif close < ll:
                send_telegram(f"🚨 {symbol} {name} LL BREAK: Price {close:.5f} < {ll:.5f}")
                time.sleep(1)
    mt5.shutdown()

print("Monitorointi aloitettu...")
while True:
    check_structure()
    time.sleep(300) # Skannaa 5 min välein