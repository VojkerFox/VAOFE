import os
import time
import requests
import logging
import MetaTrader5 as mt5
import jax.numpy as jnp
from datetime import datetime
from typing import Tuple, Dict, Any

# ==========================================
# 1. KONFIGURAATIO JA ASETUKSET
# ==========================================
TELEGRAM_BOT_TOKEN = "8658806596:AAH3jFlP7LKuHY8wMXBt02kD9UMC9SacZRI"
TELEGRAM_CHAT_ID = "260783230"

SYMBOLS = ["EURUSD", "GBPUSD", "XAUUSD", "BTCUSD", "USDCAD", "AUDUSD", "EURAUD", "GBPJPY"]

TIMEFRAMES = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30
}

# --- Algoritmin Hienosäätö ---
SWING_STRENGTH = 10         # Macro-tason vahvuus
RETEST_APPROACH_PIPS = 8.0  # Valuuttojen lähestymisalue
BOUNCE_CONFIRM_PIPS = 3.5   # Valuuttojen kimmokevahvistus
FAKE_OUT_PIPS = 4.0         # Fake-out raja (SL-suoja)
MAX_WAIT_CANDLES = 24       # Maksimiodotus (aidot kynttilät)
SCAN_INTERVAL_SEC = 3       # Skannausnopeus

# ==========================================
# 2. LOGGING & ALUSTUS
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Sniffer3")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

radar_states: Dict[str, Dict[str, Any]] = {}

def send_telegram_alert(message: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=5)
    except:
        pass

def get_pip_size(pair: str) -> float:
    if "JPY" in pair: return 0.01
    if "GOLD" in pair or "XAU" in pair: return 0.1
    if "BTC" in pair: return 1.0
    if "US30" in pair or "SPX" in pair: return 1.0
    return 0.0001

# ==========================================
# 3. JAX RAKENNETUNNISTUS (Sis. Grace Period)
# ==========================================
def detect_bos_structure(highs: jnp.ndarray, lows: jnp.ndarray, closes: jnp.ndarray, times: list, strength: int = 10):
    hist_h = highs[:-1]
    hist_l = lows[:-1]
    
    recent_swing_high, recent_swing_low = float('inf'), 0.0
    found_h, found_l = False, False
    
    start_idx = len(hist_h) - strength - 1
    if start_idx >= strength:
        for i in range(start_idx, strength - 1, -1):
            window_h = hist_h[i - strength : i + strength + 1]
            window_l = hist_l[i - strength : i + strength + 1]
            if not found_h and hist_h[i] == jnp.max(window_h):
                recent_swing_high = float(hist_h[i])
                found_h = True
            if not found_l and hist_l[i] == jnp.min(window_l):
                recent_swing_low = float(hist_l[i])
                found_l = True
            if found_h and found_l: break
                
    if not found_h: recent_swing_high = float(jnp.max(hist_h))
    if not found_l: recent_swing_low = float(jnp.min(hist_l))
    
    bull_bos, bear_bos = False, False
    bos_time = 0
    
    # GRACE PERIOD: Katsotaan 3 viimeistä suljettua kynttilää. 
    # Vaikka käynnistäisit botin myöhässä, se nappaa juuri tapahtuneen murtuman!
    for i in range(-4, -1):
        prev_c = float(closes[i-1])
        curr_c = float(closes[i])
        
        if prev_c <= recent_swing_high and curr_c > recent_swing_high:
            bull_bos, bear_bos = True, False
            bos_time = times[i]
        elif prev_c >= recent_swing_low and curr_c < recent_swing_low:
            bear_bos, bull_bos = True, False
            bos_time = times[i]
            
    return bull_bos, bear_bos, recent_swing_high, recent_swing_low, bos_time

# ==========================================
# 4. KAKSIVAIHEINEN TUTKASILMUKKA
# ==========================================
def run_sniffer3() -> None:
    if not mt5.initialize():
        logger.error("MT5 alustus epäonnistui.")
        return
        
    logger.info("🦅 SNIFFER 3: BULLETPROOF RADAR KÄYNNISTETTY 🦅")
    send_telegram_alert("🦅 <b>Sniffer 3: BULLETPROOF RADAR KÄYNNISTETTY</b> 🦅\nAikalukot korjattu, Grace Period aktivoitu. Odotetaan A+ setuppeja.")
    
    try:
        while True:
            for symbol in SYMBOLS:
                if mt5.symbol_info(symbol) is None: continue
                pip_factor = get_pip_size(symbol)
                
                # Kullalle ja BTC:lle dynaamisesti leveämmät retest-alueet
                if "XAU" in symbol or "GOLD" in symbol:
                    zone_appr = 20.0 * pip_factor  # 2.0 USD lähestyminen
                    bounce_req = 5.0 * pip_factor  # 0.5 USD kimmoke
                    zone_fake = 10.0 * pip_factor  # 1.0 USD fakeout sieto
                elif "BTC" in symbol:
                    zone_appr = 150.0 * pip_factor
                    bounce_req = 50.0 * pip_factor
                    zone_fake = 100.0 * pip_factor
                else:
                    zone_appr = RETEST_APPROACH_PIPS * pip_factor
                    bounce_req = BOUNCE_CONFIRM_PIPS * pip_factor
                    zone_fake = FAKE_OUT_PIPS * pip_factor
                
                for tf_name, tf_value in TIMEFRAMES.items():
                    state_key = f"{symbol}_{tf_name}"
                    if state_key not in radar_states:
                        radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': 0, 'pivot': 0.0}
                        
                    rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 150)
                    if rates is None or len(rates) < 100: continue
                    
                    highs = jnp.array([r['high'] for r in rates])
                    lows = jnp.array([r['low'] for r in rates])
                    closes = jnp.array([r['close'] for r in rates])
                    times = [int(r['time']) for r in rates]
                    
                    live_price = float(rates[-1]['close']) 
                    current_state = radar_states[state_key]['state']
                    
                    # ---------------------------------------------------------
                    # VAIHE 1: IDLE
                    # ---------------------------------------------------------
                    if current_state == 'IDLE':
                        bull_bos, bear_bos, res_level, sup_level, bos_time = detect_bos_structure(highs, lows, closes, times, SWING_STRENGTH)
                        
                        if bull_bos and bos_time > radar_states[state_key]['bos_time']:
                            radar_states[state_key] = {'state': 'BOS_PENDING', 'level': res_level, 'dir': 'BULL', 'bos_time': bos_time, 'pivot': 0.0}
                            logger.info(f"[{symbol} {tf_name}] VAIHE 1: BULL BOS @ {res_level:.5f}")
                            send_telegram_alert(f"🔍 <b>Sniffer3: VAIHE 1 (BOS)</b>\n<b>{symbol} {tf_name}</b> | 🟢 BULLISH\n<b>Taso:</b> <code>{res_level:.5f}</code>\n<i>Odotetaan vetäytymistä...</i>")
                            
                        elif bear_bos and bos_time > radar_states[state_key]['bos_time']:
                            radar_states[state_key] = {'state': 'BOS_PENDING', 'level': sup_level, 'dir': 'BEAR', 'bos_time': bos_time, 'pivot': 0.0}
                            logger.info(f"[{symbol} {tf_name}] VAIHE 1: BEAR BOS @ {sup_level:.5f}")
                            send_telegram_alert(f"🔍 <b>Sniffer3: VAIHE 1 (BOS)</b>\n<b>{symbol} {tf_name}</b> | 🔴 BEARISH\n<b>Taso:</b> <code>{sup_level:.5f}</code>\n<i>Odotetaan vetäytymistä...</i>")

                    # ---------------------------------------------------------
                    # VAIHE 1.5 & VAIHE 2: RETEST & BOUNCE
                    # ---------------------------------------------------------
                    elif current_state in ['BOS_PENDING', 'RETEST_TOUCHED']:
                        target_level = radar_states[state_key]['level']
                        direction = radar_states[state_key]['dir']
                        bos_time = radar_states[state_key]['bos_time']
                        
                        # Korjattu aikalukko: Lasketaan aidot kynttilät!
                        candles_passed = sum(1 for t in times if t >= bos_time) - 1
                        if candles_passed > MAX_WAIT_CANDLES:
                            radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'pivot': 0.0}
                            logger.info(f"[{symbol} {tf_name}] Setup peruttu (Aikalukko: {candles_passed} kynttilää).")
                            continue

                        if direction == 'BULL':
                            # Onko hinta romahtanut liian alas? (Fakeout)
                            if live_price < target_level - zone_fake:
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'pivot': 0.0}
                                
                            # Onko hinta saapunut lähestymisalueelle?
                            elif live_price <= target_level + zone_appr:
                                if current_state == 'BOS_PENDING':
                                    radar_states[state_key]['state'] = 'RETEST_TOUCHED'
                                    radar_states[state_key]['pivot'] = live_price
                                    logger.warning(f"[{symbol} {tf_name}] Vyöhykkeellä. Odotetaan kimmoketta ylös...")
                                else:
                                    # Päivitetään Pivot, jos mennään alemmas
                                    if live_price < radar_states[state_key]['pivot']:
                                        radar_states[state_key]['pivot'] = live_price
                                    
                                    # VAIHE 2 LAUKAISU: Hinta pomppaa Pivotista ylös!
                                    if live_price >= radar_states[state_key]['pivot'] + bounce_req:
                                        logger.info(f"🎯 [{symbol} {tf_name}] VAIHE 2 RETEST OK! Entry: {live_price:.5f}")
                                        msg = (f"🔥 <b>Sniffer3: VAIHE 2 (RETEST VAHVISTETTU)</b> 🔥\n\n"
                                               f"<b>Pari:</b> {symbol} | <b>TF:</b> {tf_name}\n"
                                               f"📈 Kimmoke vahvistettu! (Pivot-käännös ylös)\n"
                                               f"<b>Entry:</b> <code>{live_price:.5f}</code>\n"
                                               f"<b>Tuki (SL):</b> <code>{target_level:.5f}</code>\n\n"
                                               f"<i>A+ Setup Valmis 1.5 Lot iskulle!</i>")
                                        send_telegram_alert(msg)
                                        radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'pivot': 0.0}

                        elif direction == 'BEAR':
                            if live_price > target_level + zone_fake:
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'pivot': 0.0}
                                
                            elif live_price >= target_level - zone_appr:
                                if current_state == 'BOS_PENDING':
                                    radar_states[state_key]['state'] = 'RETEST_TOUCHED'
                                    radar_states[state_key]['pivot'] = live_price
                                    logger.warning(f"[{symbol} {tf_name}] Vyöhykkeellä. Odotetaan kimmoketta alas...")
                                else:
                                    if live_price > radar_states[state_key]['pivot']:
                                        radar_states[state_key]['pivot'] = live_price
                                        
                                    if live_price <= radar_states[state_key]['pivot'] - bounce_req:
                                        logger.info(f"🎯 [{symbol} {tf_name}] VAIHE 2 RETEST OK! Entry: {live_price:.5f}")
                                        msg = (f"🔥 <b>Sniffer3: VAIHE 2 (RETEST VAHVISTETTU)</b> 🔥\n\n"
                                               f"<b>Pari:</b> {symbol} | <b>TF:</b> {tf_name}\n"
                                               f"📉 Kimmoke vahvistettu! (Pivot-käännös alas)\n"
                                               f"<b>Entry:</b> <code>{live_price:.5f}</code>\n"
                                               f"<b>Vastus (SL):</b> <code>{target_level:.5f}</code>\n\n"
                                               f"<i>A+ Setup Valmis 1.5 Lot iskulle!</i>")
                                        send_telegram_alert(msg)
                                        radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'pivot': 0.0}

            time.sleep(SCAN_INTERVAL_SEC)
            
    except KeyboardInterrupt:
        logger.info("Sniffer3 sammutettu (Ctrl+C).")
        send_telegram_alert("💤 <b>Sniffer3 sammutettu.</b>")
    except Exception as e:
        logger.error(f"Kriittinen virhe: {e}", exc_info=True)
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    run_sniffer3()