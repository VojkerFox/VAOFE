import time
import MetaTrader5 as mt5

# --- 1. SYÖTETYT TASOT JA SYMBOLIMAPPPING ---
# Huom: Muuta tarvittaessa oikeanpuoleiset nimet vastaamaan välittäjäsi (broker) symboleita (esim. 'GOLD' tai 'BTCUSD')
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

# Tilanhallinta, ettei skripti huuda jokaisella tickillä uudestaan
alert_history = {key: {"buy_triggered": False, "sell_triggered": False} for key in LEVELS}

def trigger_walker_alert(display_name, order_type, price, target_level):
    """
    Tämä funktio laukaisee hälytyksen. Tähän voit myöhemmin integroida 
    sen aiemmin suunnitellun Check-list -promptin tai Telegram-botin.
    """
    print("\n" + "="*60)
    print(f"🚨 MR. WALKER ML ALERT: {display_name} 🚨")
    print(f"Suunta: {order_type.upper()}")
    print(f"Nykyinen hinta: {price}")
    print(f"Rikottu taso: {target_level}")
    print("="*60 + "\n")

def main():
    # Alustetaan yhteys MT5-terminaaliin
    if not mt5.initialize():
        print("MT5 alustus epäonnistui. Varmista, että MT5-alusta on auki koneella.")
        return
    
    print("Mr. Walker ML Alerter aktivoitu. Seurataan 14 instrumenttia...")
    
    try:
        while True:
            for name, config in LEVELS.items():
                mt5_symbol = config["symbol"]
                
                # Haetaan viimeisin tick-data terminaalista
                tick = mt5.symbol_info_tick(mt5_symbol)
                if tick is None:
                    continue
                
                current_price = tick.last if tick.last != 0 else tick.bid
                
                # OSTO-tason tarkistus (Price > buy_above)
                if current_price > config["buy_above"]:
                    if not alert_history[name]["buy_triggered"]:
                        trigger_walker_alert(name, "OSTO (Buy Above)", current_price, config["buy_above"])
                        alert_history[name]["buy_triggered"] = True
                else:
                    # Nollataan hälytysvalmius, jos hinta palaa laatikon sisälle
                    alert_history[name]["buy_triggered"] = False
                
                # MYYNTI-tason tarkistus (Price < sell_below)
                if current_price < config["sell_below"]:
                    if not alert_history[name]["sell_triggered"]:
                        trigger_walker_alert(name, "MYYNTI (Sell Below)", current_price, config["sell_below"])
                        alert_history[name]["sell_triggered"] = True
                else:
                    # Nollataan hälytysvalmius, jos hinta palaa laatikon sisälle
                    alert_history[name]["sell_triggered"] = False
            
            # Kevyt lepotila (0.5 sekuntia), ettei prosessori käy kuumana
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nHälytysjärjestelmä suljettu manuaalisesti.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()