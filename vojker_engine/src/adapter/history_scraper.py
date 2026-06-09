import sys
import os
import MetaTrader5 as mt5
from datetime import datetime, timedelta

# Lisätään projektin juuri hakupolkuun
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.postgres_db import save_to_db

def initialize_mt5():
    if not mt5.initialize():
        print("MetaTrader5:n alustus epäonnistui")
        quit()

def scrape_and_save_to_vofe_db():
    initialize_mt5()
    
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP"]
    
    # 1. Kelataan aikaa 3 päivää taaksepäin (torstaille), jolloin markkinat olivat auki
    start_date = datetime.now() - timedelta(days=3)
    
    for pair in pairs:
        print(f"Haetaan dataa: {pair}...")
        
        if not mt5.symbol_select(pair, True):
            print(f"⚠️ Symbolia '{pair}' ei löydy. Tarkista nimi MT5:stä.")
            continue
            
        # 2. Haetaan 2000 tickiä alkaen torstaista eteenpäin
        ticks = mt5.copy_ticks_from(pair, start_date, 2000, mt5.COPY_TICKS_ALL)
        
        if ticks is not None and len(ticks) > 0:
            count = 0
            for tick in ticks:
                save_to_db(pair, tick)
                count += 1
            print(f"✅ Data tallennettu kantaan: {pair} ({count} tickiä)")
        else:
            print(f"❌ Ei dataa parille: {pair}.")
            
    mt5.shutdown()
    print("Datan haku valmis.")

if __name__ == "__main__":
    scrape_and_save_to_vofe_db()