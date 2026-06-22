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

# Lisätään compute-kansio polkuun, jotta omat moduulit löytyvät
sys.path.append(os.path.join(os.path.dirname(__file__), 'compute'))

# Yritetään tuoda graafikone (Jos puuttuu, tehdään tyhjä funktio jottei ohjelma kaadu)
try:
    from chart_maker import generate_setup_chart
except ImportError as e:
    print(f"⚠️ Ei voitu tuoda chart_makeria: {e}")
    def generate_setup_chart(*args, **kwargs): pass

# Yritetään tuoda tietokantaseuranta (Jos puuttuu, tehdään tyhjä funktio jottei ohjelma kaadu)
try:
    from sniffer_tracker import log_trade_signal
except ImportError as e:
    print(f"⚠️ Ei voitu tuoda sniffer_trackeria: {e}")
    def log_trade_signal(*args, **kwargs): pass

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

# --- ASYMMETRINEN RAKENNEFYSIIKKA (UUSI) ---
LEFT_SWING_BARS = 15        # Vaatii pitkän historian vasemmalle (rakentaa vahvan tason)
RIGHT_SWING_BARS = 2        # Vaatii vain 2 kynttilää oikealle (sallii nopeat/ryömivät murtumat!)
MAX_WAIT_CANDLES = 72       # KORJATTU: Nostettu 72 kynttilään (6 tuntia). Antaa hinnan konsolidoida rauhassa!
SCAN_INTERVAL_SEC = 3

# UUDET DYNAAMISET ASETUKSET (Elonin Askel 1: Make less dumb)
MIN_BOUNCE_PIPS = 3.0       # Lattia (Kohinan suodatus)
MAX_BOUNCE_PIPS = 22.0      # Katto (Uutiskaaoksen suodatus)
ATR_MULTIPLIER = 0.50       # Kimmokkeen tulee olla 50% ATR:stä
FAKEOUT_MULTIPLIER = 1.2    # Sallii 120% ATR:n kokoisen liquidity sweepin läpi tason

# ==========================================
# 2. LOGGING & TELEGRAM KUVANLÄHETYS
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Sniffer3")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

radar_states: Dict[str, Dict[str, Any]] = {}

def send_telegram_alert(message: str, reply_to_msg_id: int = None, photo_path: str = None, symbol: str = None) -> int:
    """
    Lähettää joko pelkän tekstin (Vaihe 1) TAI kuvan (Vaihe 2).
    """
    if photo_path and os.path.exists(photo_path):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": message,
            "parse_mode": "HTML"
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
                else:
                    logger.error(f"Telegram API hylkäsi kuvan: {res_data}")
            except Exception as e:
                logger.error(f"Telegram kuvanlähetysvirhe: {e}")
                
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
# 3. JAX RAKENNETUNNISTUS (ASYMMETRINEN)
# ==========================================
def detect_bos_structure(highs: jnp.ndarray, lows: jnp.ndarray, closes: jnp.ndarray, times: list, left_bars: int = 15, right_bars: int = 2):
    hist_h = highs[:-1]
    hist_l = lows[:-1]
    recent_swing_high, recent_swing_low = float('inf'), 0.0
    found_h, found_l = False, False
    
    # Asymmetrinen ikkuna mahdollistaa äkilliset murtumat konsolidaation jälkeen
    start_idx = len(hist_h) - right_bars - 1
    if start_idx >= left_bars:
        for i in range(start_idx, left_bars - 1, -1):
            window_h = hist_h[i - left_bars : i + right_bars + 1]
            window_l = hist_l[i - left_bars : i + right_bars + 1]
            
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
    send_telegram_alert("🦅 <b>Sniffer 3: VISUAL SNIPER KÄYNNISTETTY</b> 🦅\nUI-Päivitys: Asymmetrinen Swing-fysiikka aktivoitu. Nappaa nyt myös nopeat ryömintämurtumat!")
    
    try:
        while True:
            for symbol in SYMBOLS:
                if mt5.symbol_info(symbol) is None: continue
                pip_factor = get_pip_size(symbol)
                
                for tf_name, tf_value in TIMEFRAMES.items():
                    state_key = f"{symbol}_{tf_name}"
                    if state_key not in radar_states:
                        radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': 0, 'breakout_peak': 0.0, 'pullback_extreme': 0.0, 'msg_id': 0, 'setup_id': ''}
                        
                    rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 150)
                    if rates is None or len(rates) < 100: continue
                    
                    highs = jnp.array([r['high'] for r in rates])
                    lows = jnp.array([r['low'] for r in rates])
                    closes = jnp.array([r['close'] for r in rates])
                    times = [int(r['time']) for r in rates]
                    
                    # ----------------------------------------------------
                    # DYNAAMISET FYSIKAN RAJAT (ATR)
                    # ----------------------------------------------------
                    recent_ranges = highs[-14:] - lows[-14:]
                    current_atr = float(jnp.mean(recent_ranges))
                    
                    raw_bounce_req = current_atr * ATR_MULTIPLIER
                    raw_fake_out = current_atr * FAKEOUT_MULTIPLIER
                    
                    if "XAU" in symbol or "GOLD" in symbol:
                        min_req = 5.0 * pip_factor
                        max_req = 20.0 * pip_factor
                        zone_fake = max(15.0 * pip_factor, raw_fake_out)
                    elif "BTC" in symbol:
                        min_req = 50.0 * pip_factor
                        max_req = 200.0 * pip_factor
                        zone_fake = max(150.0 * pip_factor, raw_fake_out)
                    else:
                        min_req = MIN_BOUNCE_PIPS * pip_factor
                        max_req = MAX_BOUNCE_PIPS * pip_factor
                        zone_fake = max(8.0 * pip_factor, raw_fake_out) 
                        
                    bounce_req = max(min_req, min(raw_bounce_req, max_req))
                    # ----------------------------------------------------

                    live_price = float(rates[-1]['close']) 
                    current_state = radar_states[state_key]['state']
                    
                    if current_state == 'IDLE':
                        bull_bos, bear_bos, res_level, sup_level, bos_time = detect_bos_structure(highs, lows, closes, times, LEFT_SWING_BARS, RIGHT_SWING_BARS)
                        
                        if bull_bos and bos_time > radar_states[state_key]['bos_time']:
                            setup_id = f"#{symbol}_{tf_name}_{datetime.fromtimestamp(bos_time).strftime('%H%M')}"
                            
                            chart_path = f"bos_{symbol}.png"
                            try:
                                generate_setup_chart(symbol, rates[-60:], res_level, live_price, "BULL", chart_path)
                            except Exception as e:
                                logger.error(f"Vaihe 1 graafin piirto epäonnistui: {e}")
                                chart_path = None

                            msg = (f"🔍 <b>Sniffer3: VAIHE 1 (BOS)</b>\n"
                                   f"Tunniste: {setup_id}\n\n"
                                   f"<b>{symbol} {tf_name}</b> | 🟢 BULLISH\n"
                                   f"<b>Taso:</b> <code>{res_level:.5f}</code>\n"
                                   f"<i>Odotetaan vetäytymistä...</i>")
                            
                            msg_id = send_telegram_alert(msg, photo_path=chart_path, symbol=symbol)
                            radar_states[state_key] = {'state': 'BOS_PENDING', 'level': res_level, 'dir': 'BULL', 'bos_time': bos_time, 'breakout_peak': live_price, 'pullback_extreme': live_price, 'msg_id': msg_id, 'setup_id': setup_id}
                            
                        elif bear_bos and bos_time > radar_states[state_key]['bos_time']:
                            setup_id = f"#{symbol}_{tf_name}_{datetime.fromtimestamp(bos_time).strftime('%H%M')}"
                            
                            chart_path = f"bos_{symbol}.png"
                            try:
                                generate_setup_chart(symbol, rates[-60:], sup_level, live_price, "BEAR", chart_path)
                            except Exception as e:
                                logger.error(f"Vaihe 1 graafin piirto epäonnistui: {e}")
                                chart_path = None

                            msg = (f"🔍 <b>Sniffer3: VAIHE 1 (BOS)</b>\n"
                                   f"Tunniste: {setup_id}\n\n"
                                   f"<b>{symbol} {tf_name}</b> | 🔴 BEARISH\n"
                                   f"<b>Taso:</b> <code>{sup_level:.5f}</code>\n"
                                   f"<i>Odotetaan vetäytymistä...</i>")
                            
                            msg_id = send_telegram_alert(msg, photo_path=chart_path, symbol=symbol)
                            radar_states[state_key] = {'state': 'BOS_PENDING', 'level': sup_level, 'dir': 'BEAR', 'bos_time': bos_time, 'breakout_peak': live_price, 'pullback_extreme': live_price, 'msg_id': msg_id, 'setup_id': setup_id}

                    elif current_state == 'BOS_PENDING':
                        target_level = radar_states[state_key]['level']
                        direction = radar_states[state_key]['dir']
                        bos_time = radar_states[state_key]['bos_time']
                        orig_msg_id = radar_states[state_key]['msg_id']
                        setup_id = radar_states[state_key]['setup_id']
                        
                        # --- UUSI: STAIR-STEP LOGIIKKA (JATKUVAN RAKENTEEN PÄIVITYS) ---
                        # Tarkistetaan, onko odotuksen aikana muodostunut UUSI murtuma (Stair-step)
                        new_bull_bos, new_bear_bos, new_res, new_sup, new_bos_time = detect_bos_structure(highs, lows, closes, times, LEFT_SWING_BARS, RIGHT_SWING_BARS)
                        
                        if direction == 'BULL' and new_bull_bos and new_bos_time > bos_time and new_res > target_level:
                            # Markkina teki "portaan" ylöspäin! Päivitetään seurattava taso uuteen (keltainen viiva)
                            logger.info(f"[{symbol} {tf_name}] STAIR-STEP BULL: Taso päivitetty {target_level:.5f} -> {new_res:.5f}")
                            radar_states[state_key] = {'state': 'BOS_PENDING', 'level': new_res, 'dir': 'BULL', 'bos_time': new_bos_time, 'breakout_peak': live_price, 'pullback_extreme': live_price, 'msg_id': orig_msg_id, 'setup_id': setup_id}
                            
                            msg = (f"📈 <b>Sniffer3: STAIR-STEP PÄIVITYS</b>\n"
                                   f"Tunniste: {setup_id}\n\n"
                                   f"<b>{symbol} {tf_name}</b> | 🟢 BULL JATKUU\n"
                                   f"<b>Uusi Tuki:</b> <code>{new_res:.5f}</code>\n"
                                   f"<i>Alkuperäinen BOS (sininen) hylätty. Momentum jatkuu, seurataan uutta ylempää porrasta (keltainen)...</i>")
                            send_telegram_alert(msg, reply_to_msg_id=orig_msg_id)
                            continue # Aloitetaan uuden tason seuranta heti uuden luupin alusta

                        elif direction == 'BEAR' and new_bear_bos and new_bos_time > bos_time and new_sup < target_level:
                            # Markkina teki "portaan" alaspäin!
                            logger.info(f"[{symbol} {tf_name}] STAIR-STEP BEAR: Taso päivitetty {target_level:.5f} -> {new_sup:.5f}")
                            radar_states[state_key] = {'state': 'BOS_PENDING', 'level': new_sup, 'dir': 'BEAR', 'bos_time': new_bos_time, 'breakout_peak': live_price, 'pullback_extreme': live_price, 'msg_id': orig_msg_id, 'setup_id': setup_id}
                            
                            msg = (f"📉 <b>Sniffer3: STAIR-STEP PÄIVITYS</b>\n"
                                   f"Tunniste: {setup_id}\n\n"
                                   f"<b>{symbol} {tf_name}</b> | 🔴 BEAR JATKUU\n"
                                   f"<b>Uusi Vastus:</b> <code>{new_sup:.5f}</code>\n"
                                   f"<i>Alkuperäinen BOS hylätty. Momentum jatkuu, seurataan uutta alempaa porrasta...</i>")
                            send_telegram_alert(msg, reply_to_msg_id=orig_msg_id)
                            continue

                        # --- NORMAALI ODOTUSLOGIIKKA JATKUU ---
                        candles_passed = sum(1 for t in times if t >= bos_time) - 1
                        if candles_passed > MAX_WAIT_CANDLES:
                            radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'breakout_peak': 0.0, 'pullback_extreme': 0.0, 'msg_id': 0, 'setup_id': ''}
                            continue

                        min_pullback = 1.0 * pip_factor # Vähintään 1 pip vetäytyminen ("ei saa olla nolla")
                        chop_tolerance = 1.5 * pip_factor # UUSI: Konsolidointi-toleranssi!

                        if direction == 'BULL':
                            # UUSI LOGIIKKA: Nollataan extreme vasta, kun tehdään oikeasti uusi merkittävä huippu, ei mikrosahauksesta.
                            if live_price > radar_states[state_key]['breakout_peak'] + chop_tolerance:
                                radar_states[state_key]['breakout_peak'] = live_price
                                radar_states[state_key]['pullback_extreme'] = live_price
                            elif live_price < radar_states[state_key]['pullback_extreme']:
                                radar_states[state_key]['pullback_extreme'] = live_price

                            if live_price < target_level - zone_fake:
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'breakout_peak': 0.0, 'pullback_extreme': 0.0, 'msg_id': 0, 'setup_id': ''}
                                continue

                            pullback_depth = radar_states[state_key]['breakout_peak'] - radar_states[state_key]['pullback_extreme']
                            
                            if pullback_depth >= min_pullback and live_price >= radar_states[state_key]['pullback_extreme'] + bounce_req:
                                chart_path = f"setup_{symbol}.png"
                                try:
                                    generate_setup_chart(symbol, rates[-60:], target_level, radar_states[state_key]['pullback_extreme'], direction, chart_path)
                                except Exception as e:
                                    logger.error(f"Graafin piirto epäonnistui: {e}")
                                    chart_path = None
                                    
                                msg = (f"🔥 <b>Sniffer3: VAIHE 2 (KIMMOKE VAHVISTETTU)</b> 🔥\n\n"
                                       f"Tunniste: {setup_id}\n"
                                       f"<b>{symbol} {tf_name}</b> | 📈 Momentum jatkuu!\n"
                                       f"<b>Entry:</b> <code>{live_price:.5f}</code>\n"
                                       f"<b>Tuki (SL):</b> <code>{target_level:.5f}</code>")
                                
                                send_telegram_alert(msg, reply_to_msg_id=orig_msg_id, photo_path=chart_path, symbol=symbol)
                                try:
                                    log_trade_signal(symbol, tf_name, "BULL", live_price, target_level)
                                except Exception as e:
                                    logger.error(f"Tietokantatallennus epäonnistui: {e}")
                                    
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'breakout_peak': 0.0, 'pullback_extreme': 0.0, 'msg_id': 0, 'setup_id': ''}

                        elif direction == 'BEAR':
                            # UUSI LOGIIKKA: Nollataan extreme vasta, kun tehdään oikeasti uusi merkittävä pohja.
                            if live_price < radar_states[state_key]['breakout_peak'] - chop_tolerance:
                                radar_states[state_key]['breakout_peak'] = live_price
                                radar_states[state_key]['pullback_extreme'] = live_price
                            elif live_price > radar_states[state_key]['pullback_extreme']:
                                radar_states[state_key]['pullback_extreme'] = live_price

                            if live_price > target_level + zone_fake:
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'breakout_peak': 0.0, 'pullback_extreme': 0.0, 'msg_id': 0, 'setup_id': ''}
                                continue

                            pullback_depth = radar_states[state_key]['pullback_extreme'] - radar_states[state_key]['breakout_peak']
                            
                            if pullback_depth >= min_pullback and live_price <= radar_states[state_key]['pullback_extreme'] - bounce_req:
                                chart_path = f"setup_{symbol}.png"
                                try:
                                    generate_setup_chart(symbol, rates[-60:], target_level, radar_states[state_key]['pullback_extreme'], direction, chart_path)
                                except Exception as e:
                                    logger.error(f"Graafin piirto epäonnistui: {e}")
                                    chart_path = None
                                    
                                msg = (f"🔥 <b>Sniffer3: VAIHE 2 (KIMMOKE VAHVISTETTU)</b> 🔥\n\n"
                                       f"Tunniste: {setup_id}\n"
                                       f"<b>{symbol} {tf_name}</b> | 📉 Momentum jatkuu!\n"
                                       f"<b>Entry:</b> <code>{live_price:.5f}</code>\n"
                                       f"<b>Vastus (SL):</b> <code>{target_level:.5f}</code>")
                                       
                                send_telegram_alert(msg, reply_to_msg_id=orig_msg_id, photo_path=chart_path, symbol=symbol)
                                try:
                                    log_trade_signal(symbol, tf_name, "BEAR", live_price, target_level)
                                except Exception as e:
                                    logger.error(f"Tietokantatallennus epäonnistui: {e}")
                                    
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'bos_time': bos_time, 'breakout_peak': 0.0, 'pullback_extreme': 0.0, 'msg_id': 0, 'setup_id': ''}

            time.sleep(SCAN_INTERVAL_SEC)
            
    except KeyboardInterrupt:
        logger.info("Sniffer3 sammutettu.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    run_sniffer3()