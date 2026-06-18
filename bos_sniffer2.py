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
SWING_STRENGTH = 10         # Kuinka monta kynttilää vaaditaan Macro-pohjan/-huipun vahvistukseen
RETEST_ZONE_PIPS = 3.5      # Kuinka lähelle tasoa hinnan pitää tulla (Pips)
FAKE_OUT_PIPS = 2.0         # Kuinka kauas tason "väärälle" puolelle hinta saa mennä ennen peruutusta
MAX_WAIT_CANDLES = 24       # Kuinka monta kynttilää odotetaan retestiä, ennen kuin set-up perutaan
SCAN_INTERVAL_SEC = 5       # Tutkan päivitysväli (sekuntia)

# ==========================================
# 2. LOGGING-JÄRJESTELMÄN ALUSTUS
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TwoStageRadar")

# Vaimennetaan JAX:n harmiton TPU-varoitus (estetään terminaalin spämmi)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

# Globaalit tilamuuttujat
radar_states: Dict[str, Dict[str, Any]] = {}
last_alerted_candle: Dict[str, int] = {}

# ==========================================
# 3. TELEGRAM-YHTEYS
# ==========================================
def send_telegram_alert(message: str) -> None:
    """Lähettää asynkronisesti viestin Telegramiin ja nappaa mahdolliset yhteysvirheet."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Telegram-virhe: Ei voitu lähettää viestiä. Syy: {e}")

# ==========================================
# 4. JAX RAKENNETUNNISTUS (MACRO BOS)
# ==========================================
def detect_bos_structure(highs: jnp.ndarray, lows: jnp.ndarray, closes: jnp.ndarray, strength: int = 10) -> Tuple[bool, bool, float, float]:
    """
    Etsii Macro-tason markkinarakenteen (Swing High/Low) ja palauttaa BOS-murtumat.
    """
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
                recent_swing_high = float(hist_h[i])
                found_h = True
            if not found_l and hist_l[i] == jnp.min(window_l):
                recent_swing_low = float(hist_l[i])
                found_l = True
            if found_h and found_l:
                break
                
    if not found_h: recent_swing_high = float(jnp.max(hist_h))
    if not found_l: recent_swing_low = float(jnp.min(hist_l))
    
    prev_close = hist_c[-2]
    curr_close = hist_c[-1]
    
    bull_bos = (prev_close <= recent_swing_high) and (curr_close > recent_swing_high)
    bear_bos = (prev_close >= recent_swing_low) and (curr_close < recent_swing_low)
    
    return bool(bull_bos), bool(bear_bos), recent_swing_high, recent_swing_low

# ==========================================
# 5. KAKSIVAIHEINEN TUTKASILMUKKA (TWO-STAGE RADAR)
# ==========================================
def run_radar_sniffer() -> None:
    if not mt5.initialize():
        logger.error("MT5 alustus epäonnistui. Tarkista, että MetaTrader 5 on auki.")
        return
        
    logger.info("🦅 SNIFFER 2: TWO-STAGE RADAR KÄYNNISTETTY 🦅")
    send_telegram_alert(
        "🦅 <b>Sniffer2: TWO-STAGE RADAR KÄYNNISTETTY</b> 🦅\n"
        "Ammattilaisversio: Parametrit optimoitu, Bounce-tarkastaja aktivoitu."
    )
    
    try:
        while True:
            for symbol in SYMBOLS:
                sym_info = mt5.symbol_info(symbol)
                if sym_info is None:
                    continue
                    
                # Dynaaminen Pips-laskenta
                point = sym_info.point
                pip_factor = point * 10 if sym_info.digits in [3, 5] else point
                
                # Dynaamiset vyöhykerajat asetusmuuttujista
                zone_buffer_in = RETEST_ZONE_PIPS * pip_factor
                zone_buffer_out = FAKE_OUT_PIPS * pip_factor
                
                for tf_name, tf_value in TIMEFRAMES.items():
                    state_key = f"{symbol}_{tf_name}"
                    
                    if state_key not in radar_states:
                        radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'time': 0}
                        
                    # Haemme 150 kynttilää turvamarginaalilla
                    rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 150)
                    if rates is None or len(rates) < 100: 
                        continue
                    
                    highs = jnp.array([r['high'] for r in rates])
                    lows = jnp.array([r['low'] for r in rates])
                    closes = jnp.array([r['close'] for r in rates])
                    
                    completed_time = int(rates[-2]['time'])
                    live_price = float(rates[-1]['close']) 
                    
                    current_state = radar_states[state_key]['state']
                    
                    # ---------------------------------------------------------
                    # VAIHE 1: IDLE-TILASSA ETSITÄÄN BOS-MURTUMAA
                    # ---------------------------------------------------------
                    if current_state == 'IDLE':
                        bull_bos, bear_bos, res_level, sup_level = detect_bos_structure(highs, lows, closes, strength=SWING_STRENGTH)
                        
                        if completed_time > last_alerted_candle.get(state_key, 0):
                            if bull_bos:
                                radar_states[state_key] = {'state': 'BOS_PENDING', 'level': res_level, 'dir': 'BULL', 'time': completed_time}
                                last_alerted_candle[state_key] = completed_time
                                logger.info(f"[{symbol} {tf_name}] VAIHE 1 LUKITTU: BULL BOS @ {res_level:.5f}")
                                
                                msg = (f"🔍 <b>Sniffer2: VAIHE 1 (BOS)</b>\n\n"
                                       f"<b>Pari:</b> {symbol} | <b>TF:</b> {tf_name}\n"
                                       f"🟢 BULLISH murtuma.\n"
                                       f"<b>Lukittu taso:</b> <code>{res_level:.5f}</code>\n"
                                       f"<i>Odotetaan vetäytymistä alueelle...</i>")
                                send_telegram_alert(msg)
                                
                            elif bear_bos:
                                radar_states[state_key] = {'state': 'BOS_PENDING', 'level': sup_level, 'dir': 'BEAR', 'time': completed_time}
                                last_alerted_candle[state_key] = completed_time
                                logger.info(f"[{symbol} {tf_name}] VAIHE 1 LUKITTU: BEAR BOS @ {sup_level:.5f}")
                                
                                msg = (f"🔍 <b>Sniffer2: VAIHE 1 (BOS)</b>\n\n"
                                       f"<b>Pari:</b> {symbol} | <b>TF:</b> {tf_name}\n"
                                       f"🔴 BEARISH murtuma.\n"
                                       f"<b>Lukittu taso:</b> <code>{sup_level:.5f}</code>\n"
                                       f"<i>Odotetaan vetäytymistä alueelle...</i>")
                                send_telegram_alert(msg)
                                
                    # ---------------------------------------------------------
                    # VAIHE 1.5: ODOTETAAN KOSKETUSTA VYÖHYKKEESEEN
                    # ---------------------------------------------------------
                    elif current_state == 'BOS_PENDING':
                        target_level = radar_states[state_key]['level']
                        direction = radar_states[state_key]['dir']
                        
                        # Aikalukko: Perutaan setup, jos kestää liian kauan
                        candles_passed = (int(rates[-1]['time']) - radar_states[state_key]['time']) / max(1, tf_value)
                        if candles_passed > MAX_WAIT_CANDLES:
                            radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'time': 0}
                            logger.info(f"[{symbol} {tf_name}] Setup peruttu (Aikalukko laukesi).")
                            continue
                            
                        if direction == 'BULL':
                            if target_level - zone_buffer_out <= live_price <= target_level + zone_buffer_in:
                                radar_states[state_key]['state'] = 'RETEST_TOUCHED'
                                logger.warning(f"[{symbol} {tf_name}] Vyöhykettä kosketettu, odotetaan kimmoketta...")
                            elif live_price < target_level - zone_buffer_out:
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'time': 0} # Suora Fake-out
                                
                        elif direction == 'BEAR':
                            if target_level - zone_buffer_in <= live_price <= target_level + zone_buffer_out:
                                radar_states[state_key]['state'] = 'RETEST_TOUCHED'
                                logger.warning(f"[{symbol} {tf_name}] Vyöhykettä kosketettu, odotetaan kimmoketta...")
                            elif live_price > target_level + zone_buffer_out:
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'time': 0}

                    # ---------------------------------------------------------
                    # VAIHE 2: ODOTETAAN KIMMOKETTA (BOUNCE) VYÖHYKKEELTÄ
                    # ---------------------------------------------------------
                    elif current_state == 'RETEST_TOUCHED':
                        target_level = radar_states[state_key]['level']
                        direction = radar_states[state_key]['dir']
                        
                        if direction == 'BULL':
                            if live_price > target_level + zone_buffer_in: # Kimmoke ylös
                                logger.info(f"🎯 [{symbol} {tf_name}] VAIHE 2 RETEST OK! Entry: {live_price:.5f}")
                                msg = (f"🔥 <b>Sniffer2: VAIHE 2 (RETEST VAHVISTETTU)</b> 🔥\n\n"
                                       f"<b>Pari:</b> {symbol} | <b>TF:</b> {tf_name}\n"
                                       f"📈 Ostajat puolustivat tasoa, Kimmoke havaittu!\n"
                                       f"<b>Entry (Kopioi):</b> <code>{live_price:.5f}</code>\n"
                                       f"<b>Tuki (SL alle):</b> <code>{target_level:.5f}</code>\n\n"
                                       f"<i>A+ Setup Valmis 1.5 Lot iskulle!</i>")
                                send_telegram_alert(msg)
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'time': 0}
                                
                            elif live_price < target_level - zone_buffer_out: # Myöhäinen Fake-out
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'time': 0}
                                
                        elif direction == 'BEAR':
                            if live_price < target_level - zone_buffer_in: # Kimmoke alas
                                logger.info(f"🎯 [{symbol} {tf_name}] VAIHE 2 RETEST OK! Entry: {live_price:.5f}")
                                msg = (f"🔥 <b>Sniffer2: VAIHE 2 (RETEST VAHVISTETTU)</b> 🔥\n\n"
                                       f"<b>Pari:</b> {symbol} | <b>TF:</b> {tf_name}\n"
                                       f"📉 Myyjät puolustivat tasoa, Kimmoke havaittu!\n"
                                       f"<b>Entry (Kopioi):</b> <code>{live_price:.5f}</code>\n"
                                       f"<b>Vastus (SL päälle):</b> <code>{target_level:.5f}</code>\n\n"
                                       f"<i>A+ Setup Valmis 1.5 Lot iskulle!</i>")
                                send_telegram_alert(msg)
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'time': 0}
                                
                            elif live_price > target_level + zone_buffer_out: # Myöhäinen Fake-out
                                radar_states[state_key] = {'state': 'IDLE', 'level': 0.0, 'dir': '', 'time': 0}

            # Nuku määritetty aika ennen seuraavaa skannausta
            time.sleep(SCAN_INTERVAL_SEC)
            
    except KeyboardInterrupt:
        logger.info("Radar-haistelija sammutettu käyttäjän toimesta (Ctrl+C).")
        send_telegram_alert("💤 <b>Sniffer2-tutka sammutettu taustalta.</b>")
    except Exception as e:
        logger.error(f"Kriittinen virhe pääsilmukassa: {e}", exc_info=True)
        send_telegram_alert(f"⚠️ <b>Sniffer2 Virhe:</b> Järjestelmä kaatui: {e}")
    finally:
        # Varmistetaan, että MT5-yhteys suljetaan siististi kaikissa tilanteissa
        mt5.shutdown()
        logger.info("MT5 yhteys suljettu turvallisesti.")

if __name__ == "__main__":
    run_radar_sniffer()