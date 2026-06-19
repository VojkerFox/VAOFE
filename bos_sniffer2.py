import os
import sys
import time
import requests
import logging
import MetaTrader5 as mt5
import jax.numpy as jnp
from datetime import datetime
from typing import Tuple, Dict, Any
import json

# Lisätään compute-kansio polkuun, jotta chart_maker löytyy
sys.path.append(os.path.join(os.path.dirname(__file__), 'compute'))
try:
    from chart_maker import generate_setup_chart
except ImportError:
    print("⚠️ Ei voitu tuoda chart_makeria. Varmista että compute/chart_maker.py on olemassa.")

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

SWING_STRENGTH = 10
RETEST_APPROACH_PIPS = 8.0
BOUNCE_CONFIRM_PIPS = 3.5
FAKE_OUT_PIPS = 8.0
MAX_WAIT_CANDLES = 24
SCAN_INTERVAL_SEC = 3

# ==========================================
# 2. LOGGING & TELEGRAM KUVANLÄHETYS
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Sniffer3")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

radar_states: Dict[str, Dict[str, Any]] = {}

def send_telegram_alert(message: str, reply_to_msg_id: int = None, photo_path: str = None, symbol: str = None) -> int:
    """
    Lähettää joko pelkän tekstin (Vaihe 1) TAI kuvan + painikkeet (Vaihe 2).
    """
    # Jos on kuva, käytetään sendPhoto-endpointia
    if photo_path and os.path.exists(photo_path):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        
        # Luodaan Action-painikkeet
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": f"🟢 Avaa MT5 ({symbol})", "url": f"tg://resolve?domain=mt5"} 
                ],
                [
                    {"text": "📊 Avaa VAOFE Dashboard", "url": "http://localhost:8501"}
                ]
            ]
        }
        
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": message,
            "parse_mode": "HTML",
            "reply_markup": json.dumps(keyboard)
        }
        
        if reply_to_msg_id:
            data["reply_to_message_id"] = reply_to_msg_id
            
        with open(photo_path, 'rb') as photo:
            files = {"photo": photo}
            try:
                response = requests.post(url, data=data, files=files, timeout=10)
                res_data = response.json()
                if res_data.get("ok"):
                    return res_data["result"]["message_id"]
            except Exception as e:
                logger.error(f"Telegram kuvanlähetysvirhe: {e}")
                
    # Jos EI ole kuvaa (Vaihe 1), käytetään tavallista sendMessage-endpointia
    else:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        if reply_to_msg_id:
            payload["reply_to_message_id"] = reply_to_msg_id
            
        try:
            response = requests.post(url, json=payload, timeout=5)
            res_data = response.json()
            if res_data.get("ok"):
                return res_data["result"]["message_id"]
        except Exception as e:
            logger.error(f"Telegram tekstivirhe: {e}")
            
    return 0

def get_pip_size(pair: str) -> float:
    if "JPY" in pair: return 0.01
    if "GOLD" in pair or "XAU" in pair: return 0.1
    if "BTC" in pair: return 1.0
    if "US30" in pair or "SPX" in pair: return 1.0
    return 0.0001

# ==========================================
# 3. JAX RAKENNETUNNISTUS 
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
        
    logger.info("🦅 SNIFFER 3: VISUAL SNIPER KÄYNNISTETTY 🦅")
    send_telegram_alert("🦅 <b>Sniffer 3: VISUAL SNIPER KÄYNNISTETTY</b> 🦅\nUI-Päivitys: Graafit, Inline-painikkeet ja Reply-ketjutus aktivoitu!")
    
    try:
        while True:
            for symbol in SYMBOLS:
                if mt5.symbol_info(symbol) is None: continue
                pip_factor = get_pip_size(symbol)
                
                if "XAU" in symbol or "GOLD" in symbol:
                    zone_appr = 20.0 * pip_factor  
                    bounce_req = 5.0 * pip_factor  
                    zone_fake = 15.0 * pip_factor  
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
                        radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': 0, 'pivot': 0.0, 'msg_id': 0, 'setup_id': ''}
                        
                    rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 150)
                    if rates is None or len(rates) < 100: continue
                    
                    highs = jnp.array([r['high'] for r in rates])
                    lows = jnp.array([r['low'] for r in rates])
                    closes = jnp.array([r['close'] for r in rates])
                    times = [int(r['time']) for r in rates]
                    
                    live_price = float(rates[-1]['close']) 
                    current_state = radar_states[state_key]['state']
                    
                    if current_state == 'IDLE':
                        bull_bos, bear_bos, res_level, sup_level, bos_time = detect_bos_structure(highs, lows, closes, times, SWING_STRENGTH)
                        
                        if bull_bos and bos_time > radar_states[state_key]['bos_time']:
                            setup_id = f"#{symbol}_{tf_name}_{datetime.fromtimestamp(bos_time).strftime('%H%M')}"
                            msg = (f"🔍 <b>Sniffer3: VAIHE 1 (BOS)</b>\n"
                                   f"Tunniste: {setup_id}\n\n"
                                   f"<b>{symbol} {tf_name}</b> | 🟢 BULLISH\n"
                                   f"<b>Taso:</b> <code>{res_level:.5f}</code>\n"
                                   f"<i>Odotetaan vetäytymistä...</i>")
                            msg_id = send_telegram_alert(msg)
                            radar_states[state_key] = {'state': 'BOS_PENDING', 'level': res_level, 'dir': 'BULL', 'bos_time': bos_time, 'pivot': 0.0, 'msg_id': msg_id, 'setup_id': setup_id}
                            
                        elif bear_bos and bos_time > radar_states[state_key]['bos_time']:
                            setup_id = f"#{symbol}_{tf_name}_{datetime.fromtimestamp(bos_time).strftime('%H%M')}"
                            msg = (f"🔍 <b>Sniffer3: VAIHE 1 (BOS)</b>\n"
                                   f"Tunniste: {setup_id}\n\n"
                                   f"<b>{symbol} {tf_name}</b> | 🔴 BEARISH\n"
                                   f"<b>Taso:</b> <code>{sup_level:.5f}</code>\n"
                                   f"<i>Odotetaan vetäytymistä...</i>")
                            msg_id = send_telegram_alert(msg)
                            radar_states[state_key] = {'state': 'BOS_PENDING', 'level': sup_level, 'dir': 'BEAR', 'bos_time': bos_time, 'pivot': 0.0, 'msg_id': msg_id, 'setup_id': setup_id}

                    elif current_state in ['BOS_PENDING', 'RETEST_TOUCHED']:
                        target_level = radar_states[state_key]['level']
                        direction = radar_states[state_key]['dir']
                        bos_time = radar_states[state_key]['bos_time']
                        orig_msg_id = radar_states[state_key]['msg_id']
                        setup_id = radar_states[state_key]['setup_id']
                        
                        candles_passed = sum(1 for t in times if t >= bos_time) - 1
                        if candles_passed > MAX_WAIT_CANDLES:
                            radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'pivot': 0.0, 'msg_id': 0, 'setup_id': ''}
                            continue

                        if direction == 'BULL':
                            if live_price < target_level - zone_fake:
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'pivot': 0.0, 'msg_id': 0, 'setup_id': ''}
                            elif current_state == 'BOS_PENDING' and live_price <= target_level + zone_appr:
                                radar_states[state_key]['state'] = 'RETEST_TOUCHED'
                                radar_states[state_key]['pivot'] = live_price
                            elif current_state == 'RETEST_TOUCHED':
                                if live_price < radar_states[state_key]['pivot']:
                                    radar_states[state_key]['pivot'] = live_price
                                if live_price >= radar_states[state_key]['pivot'] + bounce_req:
                                    # UUTTA: Generoidaan kuva juuri ennen lähetystä!
                                    chart_path = f"setup_{symbol}.png"
                                    try:
                                        # Otetaan 60 viimeistä kynttilää graafiin
                                        generate_setup_chart(symbol, rates[-60:], target_level, radar_states[state_key]['pivot'], direction, chart_path)
                                    except Exception as e:
                                        logger.error(f"Graafin piirto epäonnistui: {e}")
                                        chart_path = None
                                        
                                    msg = (f"🔥 <b>Sniffer3: VAIHE 2 (RETEST VAHVISTETTU)</b> 🔥\n\n"
                                           f"Tunniste: {setup_id}\n"
                                           f"<b>{symbol} {tf_name}</b> | 📈 Pivot-käännös ylös!\n"
                                           f"<b>Entry:</b> <code>{live_price:.5f}</code>\n"
                                           f"<b>Tuki (SL):</b> <code>{target_level:.5f}</code>")
                                    
                                    send_telegram_alert(msg, reply_to_msg_id=orig_msg_id, photo_path=chart_path, symbol=symbol)
                                    radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'pivot': 0.0, 'msg_id': 0, 'setup_id': ''}

                        elif direction == 'BEAR':
                            if live_price > target_level + zone_fake:
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'pivot': 0.0, 'msg_id': 0, 'setup_id': ''}
                            elif current_state == 'BOS_PENDING' and live_price >= target_level - zone_appr:
                                radar_states[state_key]['state'] = 'RETEST_TOUCHED'
                                radar_states[state_key]['pivot'] = live_price
                            elif current_state == 'RETEST_TOUCHED':
                                if live_price > radar_states[state_key]['pivot']:
                                    radar_states[state_key]['pivot'] = live_price
                                if live_price <= radar_states[state_key]['pivot'] - bounce_req:
                                    # UUTTA: Generoidaan kuva juuri ennen lähetystä!
                                    chart_path = f"setup_{symbol}.png"
                                    try:
                                        generate_setup_chart(symbol, rates[-60:], target_level, radar_states[state_key]['pivot'], direction, chart_path)
                                    except Exception as e:
                                        logger.error(f"Graafin piirto epäonnistui: {e}")
                                        chart_path = None
                                        
                                    msg = (f"🔥 <b>Sniffer3: VAIHE 2 (RETEST VAHVISTETTU)</b> 🔥\n\n"
                                           f"Tunniste: {setup_id}\n"
                                           f"<b>{symbol} {tf_name}</b> | 📉 Pivot-käännös alas!\n"
                                           f"<b>Entry:</b> <code>{live_price:.5f}</code>\n"
                                           f"<b>Vastus (SL):</b> <code>{target_level:.5f}</code>")
                                           
                                    send_telegram_alert(msg, reply_to_msg_id=orig_msg_id, photo_path=chart_path, symbol=symbol)
                                    radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'pivot': 0.0, 'msg_id': 0, 'setup_id': ''}

            time.sleep(SCAN_INTERVAL_SEC)
            
    except KeyboardInterrupt:
        logger.info("Sniffer3 sammutettu.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    run_sniffer3()