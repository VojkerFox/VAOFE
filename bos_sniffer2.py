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

SWING_STRENGTH = 10

# Kaksivaiheisen tutkan globaali tila salkulle
# Rakennus: radar_states[state_key] = {'state': 'IDLE'/'BOS_PENDING', 'level': float, 'dir': 'BULL'/'BEAR', 'time': int}
radar_states = {}
last_alerted_candle = {}

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        pass

# ==========================================
# 2. RAKENNETUNNISTUS (Sama vakaa Macro-ydin)
# ==========================================
def detect_bos_structure(highs, lows, closes, strength=10):
    hist_h = highs[:-1]
    hist_l = lows[:-1]
    hist_c = closes[:-1]
    
    recent_swing_high = float('inf')
    recent_swing_low = 0.0
    found_h, found_l = False, False
    
    start_idx = len(hist_h) - strength - 1
    if start_idx >= strength:
        for i in range(start_idx, strength - 1, -1):
            window_h = hist_h[i - strength : i + strength + 1]
            window_l = hist_l[i - strength : i + strength + 1]
            
            if not found_h and hist_h[i] == jnp.max(window_h):
                recent_swing_high = hist_h[i]
                found_h = True
            if not found_l and hist_l[i] == jnp.min(window_l):
                recent_swing_low = hist_l[i]
                found_l = True
            if found_h and found_l:
                break
                
    if not found_h: recent_swing_high = jnp.max(hist_h)
    if not found_l: recent_swing_low = jnp.min(hist_l)
    
    prev_close = hist_c[-2]
    curr_close = hist_c[-1]
    
    bull_bos = (prev_close <= recent_swing_high) and (curr_close > recent_swing_high)
    bear_bos = (prev_close >= recent_swing_low) and (curr_close < recent_swing_low)
    
    return bool(bull_bos), bool(bear_bos), float(recent_swing_high), float(recent_swing_low)

# ==========================================
# 3. KAKSIVAIHEINEN TUTKASILMUKKA
# ==========================================
def run_radar_sniffer():
    if not mt5.initialize():
        print("❌ MT5 alustus epäonnistui.")
        return
        
    print("🦅 SNIFFER 2: TWO-STAGE RADAR ENGINES KÄYNNISTETTY 🦅")
    send_telegram_alert("🦅 <b>Sniffer2-signaali: TWO-STAGE RADAR KÄYNNISTETTY</b> 🦅\nVaihe 1 (BOS) lukitsee tason ja Vaihe 2 (Retest) ilmoittaa iskupaikasta.")
    
    while True:
        try:
            for symbol in SYMBOLS:
                # Haetaan piste/pip-koko instrumentille dynaamisesti
                sym_info = mt5.symbol_info(symbol)
                if sym_info is None: continue
                point = sym_info.point
                pip_factor = point * 10 if sym_info.digits in [3, 5] else point
                
                for tf_name, tf_value in TIMEFRAMES.items():
                    state_key = f"{symbol}_{tf_name}"
                    
                    # Alustetaan tyhjä tila tarvittaessa
                    if state_key not in radar_states:
                        radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'time': 0}
                        
                    rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 150)
                    if rates is None or len(rates) < 100: continue
                    
                    highs = jnp.array([r['high'] for r in rates])
                    lows = jnp.array([r['low'] for r in rates])
                    closes = jnp.array([r['close'] for r in rates])
                    
                    completed_time = int(rates[-2]['time'])
                    live_price = float(rates[-1]['close']) # Nykyinen sekuntihinta
                    
                    # --- VAIHE 1: IDLE-TILASSA ETSITÄÄN BOS-MURTUMAA ---
                    if radar_states[state_key]['state'] == 'IDLE':
                        bull_bos, bear_bos, res_level, sup_level = detect_bos_structure(highs, lows, closes, strength=SWING_STRENGTH)
                        
                        if completed_time > last_alerted_candle.get(state_key, 0):
                            if bull_bos:
                                radar_states[state_key] = {'state': 'BOS_PENDING', 'level': res_level, 'dir': 'BULL', 'time': completed_time}
                                last_alerted_candle[state_key] = completed_time
                                
                                msg = (f"🔍 <b>Sniffer2-signaali: VAIHE 1 (BOS)</b>\n\n"
                                       f"<b>Pari:</b> {symbol} | <b>TF:</b> {tf_name}\n"
                                       f"🟢 BULLISH murtuma sulkeutunut.\n"
                                       f"<b>Lukittu taso (Napauta kopioidaksesi):</b>\n"
                                       f"<code>{res_level:.5f}</code>\n\n"
                                       f"<i>Tutka siirtyy seurantaan. Odotetaan kääntymistä Retest-alueelle...</i>")
                                send_telegram_alert(msg)
                                print(f"{datetime.now().strftime('%H:%M:%S')} | [Sniffer2] {symbol} {tf_name} | VAIHE 1 LUKITTU: {res_level:.5f}")
                                
                            elif bear_bos:
                                radar_states[state_key] = {'state': 'BOS_PENDING', 'level': sup_level, 'dir': 'BEAR', 'time': completed_time}
                                last_alerted_candle[state_key] = completed_time
                                
                                msg = (f"🔍 <b>Sniffer2-signaali: VAIHE 1 (BOS)</b>\n\n"
                                       f"<b>Pari:</b> {symbol} | <b>TF:</b> {tf_name}\n"
                                       f"🔴 BEARISH murtuma sulkeutunut.\n"
                                       f"<b>Lukittu taso (Napauta kopioidaksesi):</b>\n"
                                       f"<code>{sup_level:.5f}</code>\n\n"
                                       f"<i>Tutka siirtyy seurantaan. Odotetaan kääntymistä Retest-alueelle...</i>")
                                send_telegram_alert(msg)
                                print(f"{datetime.now().strftime('%H:%M:%S')} | [Sniffer2] {symbol} {tf_name} | VAIHE 1 LUKITTU: {sup_level:.5f}")
                                
                    # --- VAIHE 2: SEURATAAN JOS HINTA PALAA RETEST-ALUEELLE ---
                    elif radar_states[state_key]['state'] == 'BOS_PENDING':
                        target_level = radar_states[state_key]['level']
                        direction = radar_states[state_key]['dir']
                        
                        # Vanhennus: Jos murtumasta on kulunut yli 24 kynttilää, perutaan tutka turvallisuussyistä
                        if len(rates) >= 26 and int(rates[-1]['time']) - radar_states[state_key]['time'] > (24 * tf_value):
                            radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'time': 0}
                            continue
                            
                        # Määritetään retest-vyöhyke dynaamisesti (BOS-taso ± 3.5 pipsiä)
                        zone_buffer = 3.5 * pip_factor
                        
                        if direction == 'BULL':
                            # Retest-alue: hinta on palannut lähelle tasoa ylhäältä päin, mutta ei romahtanut läpi
                            is_in_retest_zone = (target_level - (1.0 * pip_factor) <= live_price <= target_level + zone_buffer)
                            # Vahvistus: M5 kynttilän wick tai live-hinnan pieni hylkäys takaisin ylös
                            if is_in_retest_zone and live_price >= target_level:
                                msg = (f"🔥 <b>Sniffer2-signaali: VAIHE 2 (RETEST VAHVISTETTU)</b> 🔥\n\n"
                                       f"<b>Pari:</b> {symbol} | <b>TF:</b> {tf_name}\n"
                                       f"📈 Ostajat puolustavat Macro-tasoa!\n"
                                       f"<b>Entry-taso (Kopioi):</b> <code>{live_price:.5f}</code>\n"
                                       f"<b>BOS-tuki:</b> <code>{target_level:.5f}</code>\n\n"
                                       f"<i>A+ Semi-Auto Setup Valmis 1.5 Lot iskuun! SL edellisen laakson alle.</i>")
                                send_telegram_alert(msg)
                                print(f"🎯 [Sniffer2] {symbol} {tf_name} | VAIHE 2 RETEST OK! Hinta: {live_price:.5f}")
                                # Palautetaan tila takaisin hakuun onnistuneen testin jälkeen
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'time': 0}
                                
                        elif direction == 'BEAR':
                            # Retest-alue: hinta on noussut takaisin lähelle tasoa alhaalta päin
                            is_in_retest_zone = (target_level - zone_buffer <= live_price <= target_level + (1.0 * pip_factor))
                            if is_in_retest_zone and live_price <= target_level:
                                msg = (f"🔥 <b>Sniffer2-signaali: VAIHE 2 (RETEST VAHVISTETTU)</b> 🔥\n\n"
                                       f"<b>Pari:</b> {symbol} | <b>TF:</b> {tf_name}\n"
                                       f"📉 Myyjät puolustavat Macro-tasoa!\n"
                                       f"<b>Entry-taso (Kopioi):</b> <code>{live_price:.5f}</code>\n"
                                       f"<b>BOS-vastus:</b> <code>{target_level:.5f}</code>\n\n"
                                       f"<i>A+ Semi-Auto Setup Valmis 1.5 Lot iskuun! SL edellisen huipun päälle.</i>")
                                send_telegram_alert(msg)
                                print(f"🎯 [Sniffer2] {symbol} {tf_name} | VAIHE 2 RETEST OK! Hinta: {live_price:.5f}")
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'time': 0}

            # Skannataan markkinaa dynaamisesti 5 sekunnin välein live-hintoja varten
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\nRadar-haistelija sammutettu.")
            send_telegram_alert("💤 <b>Sniffer2-tutka sammutettu taustalta.</b>")
            break
        except Exception as e:
            time.sleep(5)

    mt5.shutdown()

if __name__ == "__main__":
    run_radar_sniffer()