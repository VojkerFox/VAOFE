import MetaTrader5 as mt5
import sys
import os

# Lisätään projektin juuri pathiin, jotta importit toimivat
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

def initialize_mt5():
    if not mt5.initialize(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"MT5 alustus epäonnistui: {mt5.last_error()}")
        return False
    print("MT5 alustettu onnistuneesti.")
    return True

def get_tick_data(symbol):
    if not mt5.symbol_select(symbol, True):
        print(f"Symbolia {symbol} ei löytynyt.")
        return None
    tick = mt5.symbol_info_tick(symbol)
    return tick

if __name__ == "__main__":
    if initialize_mt5():
        data = get_tick_data("EURUSD")
        if data:
            print(f"Viimeisin tick: Bid={data.bid}, Ask={data.ask}, Vol={data.last}")
        mt5.shutdown()