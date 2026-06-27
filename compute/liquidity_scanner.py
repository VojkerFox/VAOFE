import MetaTrader5 as mt5
import pandas as pd
import json
import time
import os
from datetime import datetime

SYMBOLS = ["EURUSD", "GBPUSD", "XAUUSD"] # Lisää muut tähän
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liquidity_matrix.json")

def run():
    if not mt5.initialize(): return
    while True:
        data = {}
        for s in SYMBOLS:
            rates = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M15, 0, 100)
            if rates is None: continue
            df = pd.DataFrame(rates)
            live = float(df['close'].iloc[-1])
            # Yksinkertaistettu laskenta
            data[s] = {"live_price": live, "status": "Normaali", "etaisyys_bsl": 10, "etaisyys_ssl": 10, "absorptio": "Ei"}
        
        with open(OUTPUT_FILE, "w") as f:
            json.dump(data, f)
        time.sleep(5)

if __name__ == "__main__":
    run()