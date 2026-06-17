import os
import time
import requests
import MetaTrader5 as mt5
import jax.numpy as jnp
from datetime import datetime

# ==========================================
# 1. ASETUKSET JA TELEGRAM
# ==========================================
TELEGRAM_BOT_TOKEN = "8658806596:AAH3jFlP7LKuHY8wMXBt02kD9UMC9SacZRI"
TELEGRAM_CHAT_ID = "260783230"

SYMBOLS = ["EURUSD", "GBPUSD", "XAUUSD", "BTCUSD", "USDCAD", "AUDUSD", "EURAUD", "GBPJPY"]

TIMEFRAMES = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30
}

# SWING_STRENGTH määrittää, kuinka iso "laakso" tai "vuori" pitää olla.
# 10 tarkoittaa, että kynttilän pitää olla matalin/korkein 21 kynttilän alueella (10 vas., 10 oik.)
SWING_STRENGTH = 10 

last_alerted_candle = {}

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        pass

# ==========================================
# 2. MACRO SWING BOS -TUNNISTUS
# ==========================================
def detect_bos_structure(highs, lows, closes, strength=10):
    """
    Etsii aitoa MACRO-tason markkinarakennetta (Keltainen viiva) 
    ja suodattaa pois mikro-tärinät (Vihreä viiva).
    """
    # 1. Poistetaan elävä live-kynttilä (-1) analyysistä
    hist_h = highs[:-1]
    hist_l = lows[:-1]
    hist_c = closes[:-1]
    
    recent_swing_high = float('inf')
    recent_swing_low = 0.0
    
    found_h, found_l = False, False
    
    # 2. Etsitään uusinta Macro Swing -pistettä
    # Aloitetaan etsintä riittävän kaukaa, jotta kynttilällä on 'strength' määrä kynttilöitä oikealla
    start_idx = len(hist_h) - strength - 1
    
    if start_idx >= strength:
        for i in range(start_idx, strength - 1, -1):
            
            # Otetaan ikkuna: esim. 10 kynttilää vasemmalle, 10 oikealle
            window_h = hist_h[i - strength : i + strength + 1]
            window_l = hist_l[i - strength : i + strength + 1]
            
            # Jos tämä kynttilä on koko ikkunan korkein -> Macro Swing High
            if not found_h and hist_h[i] == jnp.max(window_h):
                recent_swing_high = hist_h[i]
                found_h = True
                
            # Jos tämä kynttilä on koko ikkunan matalin -> Macro Swing Low (Keltainen viiva)
            if not found_l and hist_l[i] == jnp.min(window_l):
                recent_swing_low = hist_l[i]
                found_l = True
                
            if found_h and found_l:
                break
                
    # Hätävara, jos kartta on täysin pystysuora
    if not found_h: recent_swing_high = jnp.max(hist_h)
    if not found_l: recent_swing_low = jnp.min(hist_l)
    
    # 3. Määritetään murtuma (BOS)
    # hist_c[-1] on juuri sulkeutunut kynttilä, hist_c[-2] on sitä edeltävä kynttilä
    prev_close = hist_c[-2]
    curr_close = hist_c[-1]
    
    # Varmistetaan, että aiemmin hinta oli tason "turvallisella" puolella ja juuri nyt sulki yli
    bull_bos = (prev_close <= recent_swing_high) and (curr_close > recent_swing_high)
    bear_bos = (prev_close >= recent_swing_low) and (curr_close < recent_swing_low)
    
    return bool(bull_bos), bool(bear_bos), float(recent_swing_high), float(recent_swing_low)

# ==========================================
# 3. PÄÄSILMUKKA (LIVE HAISTELIJA)
# ==========================================
def run_bos_sniffer():
    if not mt5.initialize():
        print("❌ MT5 alustus epäonnistui.")
        return
        
    print("🦅 MACRO SWING BOS -HAISTELIJA KÄYNNISTETTY 🦅")
    send_telegram_alert(f"🦅 <b>MACRO SWING BOS -HAISTELIJA KÄYNNISTETTY</b> 🦅\nFiltteri asetettu: Vain vahvat {SWING_STRENGTH} kynttilän rakenteet hyväksytään (Keltainen viiva).")
    
    while True:
        try:
            for symbol in SYMBOLS:
                for tf_name, tf_value in TIMEFRAMES.items():
                    
                    rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 150)
                    if rates is None or len(rates) < 100:
                        continue
                        
                    highs = jnp.array([r['high'] for r in rates])
                    lows = jnp.array([r['low'] for r in rates])
                    closes = jnp.array([r['close'] for r in rates])
                    
                    completed_time = int(rates[-2]['time'])
                    completed_close_price = float(rates[-2]['close'])
                    
                    # Kutsutaan funktiota SWING_STRENGTH -parametrilla
                    bull_bos, bear_bos, res_level, sup_level = detect_bos_structure(highs, lows, closes, strength=SWING_STRENGTH)
                    
                    state_key = f"{symbol}_{tf_name}"
                    last_time = last_alerted_candle.get(state_key, 0)
                    
                    if completed_time > last_time:
                        if bull_bos:
                            msg = (f"🟢 <b>MACRO BULLISH BOS DETECTED</b>\n\n"
                                   f"<b>Pari:</b> {symbol}\n"
                                   f"<b>TF:</b> {tf_name}\n"
                                   f"<b>Murrettu Katto:</b> {res_level:.5f}\n"
                                   f"<b>Sulki tasoon:</b> {completed_close_price:.5f}\n\n"
                                   f"<i>Rakennemurtuma vahvistettu!</i>")
                            send_telegram_alert(msg)
                            print(f"{datetime.now().strftime('%H:%M:%S')} | {symbol} {tf_name} | BULL BOS | {completed_close_price:.5f} > {res_level:.5f}")
                            last_alerted_candle[state_key] = completed_time
                            
                        elif bear_bos:
                            msg = (f"🔴 <b>MACRO BEARISH BOS DETECTED</b>\n\n"
                                   f"<b>Pari:</b> {symbol}\n"
                                   f"<b>TF:</b> {tf_name}\n"
                                   f"<b>Murrettu Lattia:</b> {sup_level:.5f}\n"
                                   f"<b>Sulki tasoon:</b> {completed_close_price:.5f}\n\n"
                                   f"<i>Rakennemurtuma vahvistettu!</i>")
                            send_telegram_alert(msg)
                            print(f"{datetime.now().strftime('%H:%M:%S')} | {symbol} {tf_name} | BEAR BOS | {completed_close_price:.5f} < {sup_level:.5f}")
                            last_alerted_candle[state_key] = completed_time

            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\nHaistelija sammutettu.")
            send_telegram_alert("💤 <b>MACRO BOS-HAISTELIJA SAMMUTETTU</b>")
            break
        except Exception as e:
            time.sleep(10)

    mt5.shutdown()

if __name__ == "__main__":
    run_bos_sniffer()