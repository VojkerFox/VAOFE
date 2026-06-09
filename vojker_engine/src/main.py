import sys
import os
import time
import requests
import psycopg2
from psycopg2 import pool
import MetaTrader5 as mt5
import jax.numpy as jnp
from dotenv import load_dotenv

# Varmistetaan projektin polku
load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.compute.delta_calc import DeltaEngine
from src.compute.liquidity_density import LiquidityDensityEngine
from src.compute.trend_engine import TrendEngine
from src.compute.wick_detector import WickScanner
from src.notify.telegram_bot import TelegramNotifier

# TUODAAN MOLEMMAT JAX-KINEETTISET MOOTTORIT (Inertia & Synteettinen DOM)
from flow_analytics import calculate_triad_inertia, calculate_synthetic_dom


# --- ASETUKSET ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP",
    "GOLD", "SILVER", "US_30"
]

ACTIVE_TIMEFRAME = mt5.TIMEFRAME_M30
CHECK_INTERVAL_SECONDS = 5 

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "vofe_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "salasana"),
    "host": "localhost",
    "port": "5432"
}

try:
    db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, **DB_CONFIG)
except Exception as e:
    print(f"❌ SQL Pool virhe: {e}")
    db_pool = None

def get_pip_size(pair):
    if "JPY" in pair: return 0.01
    if "GOLD" in pair or "SILVER" in pair or "XAU" in pair or "XAG" in pair: return 0.1
    if "US30" in pair or "SPX" in pair or "NAS" in pair or "US_30" in pair: return 1.0
    return 0.0001

def save_to_db(pair, predicted, actual, friction, loss, volume, macd_hist, wick_signal, wick_pct, dom_imbalance):
    if not db_pool: return
    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO ai_learning_logs 
            (pair, predicted_pips, actual_pips, friction_weight, loss, volume, macd_hist, wick_signal, wick_pct, dom_imbalance) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (pair, predicted, actual, friction, loss, volume, macd_hist, wick_signal, wick_pct, dom_imbalance)
        )
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"❌ TIETOKANTAVIRHE ({pair}): {e}") 
    finally:
        if conn: db_pool.putconn(conn)

def hae_order_flow_stats(pair):
    last_tick = mt5.symbol_info_tick(pair)
    if last_tick is None: return None
        
    start_time = int(last_tick.time) - 1800 
    end_time = int(last_tick.time) + 60     
    
    ticks = mt5.copy_ticks_range(pair, start_time, end_time, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) < 10: 
        return None
    
    volumes_raw = jnp.array(ticks['volume'], dtype=jnp.float32)
    tick_volumes = jnp.where(jnp.sum(volumes_raw) == 0, jnp.ones_like(volumes_raw), volumes_raw)
    
    volume = float(jnp.sum(tick_volumes))
    bid_vol = jnp.sum(jnp.where(jnp.diff(ticks['bid']) < 0, tick_volumes[1:], 0))
    ask_vol = jnp.sum(jnp.where(jnp.diff(ticks['ask']) > 0, tick_volumes[1:], 0))
    
    delta = float(DeltaEngine.calculate_delta(bid_vol, ask_vol))
    return {"volume": volume, "delta": delta}

def hae_l2_dom_imbalance(pair):
    """
    Hakee reaaliaikaisen Level 2 Depth of Market snapshotin ja laskee JAXilla tilauskirjan epätasapainon.
    Palauttaa arvon väliltä -1.0 (täysi myyntiseinä) ... +1.0 (täysi ostoseinä).
    """
    book = mt5.market_book_get(pair)
    if book is None or len(book) == 0:
        return None
        
    bids_vol = []
    asks_vol = []
    
    for item in book:
        # MT5 tyypit: 1 = BOOK_TYPE_SELL (Asks), 2 = BOOK_TYPE_BUY (Bids)
        if item.type == 1:
            asks_vol.append(float(item.volume))
        elif item.type == 2:
            bids_vol.append(float(item.volume))
            
    if not bids_vol or not asks_vol:
        return None
        
    # Muutetaan JAX-tensoreiksi ja ajetaan valmis L2-analyysi
    bids_arr = jnp.array(bids_vol, dtype=jnp.float32)
    asks_arr = jnp.array(asks_vol, dtype=jnp.float32)
    
    l2_score = float(LiquidityDensityEngine.analyze_l2_imbalance(bids_arr, asks_arr))
    return l2_score

def start_live_engine():
    print(f"🚀 VAOFE SciML Moottori käynnistyy (Hybrid JAX-DOM & M30 MACD Crossover)...")
    if not mt5.initialize():
        print("❌ MT5 alustus epäonnistui!")
        return

    # FAIL-SAFE: Varmistetaan että tietokanta tukee reaaliaikaista L2 DOM -tiedonkeruuta
    if db_pool:
        try:
            conn = db_pool.getconn()
            cur = conn.cursor()
            cur.execute("ALTER TABLE ai_learning_logs ADD COLUMN IF NOT EXISTS volume NUMERIC;")
            cur.execute("ALTER TABLE ai_learning_logs ADD COLUMN IF NOT EXISTS macd_hist NUMERIC;")
            cur.execute("ALTER TABLE ai_learning_logs ADD COLUMN IF NOT EXISTS wick_signal NUMERIC;")
            cur.execute("ALTER TABLE ai_learning_logs ADD COLUMN IF NOT EXISTS wick_pct NUMERIC;")
            cur.execute("ALTER TABLE ai_learning_logs ADD COLUMN IF NOT EXISTS dom_imbalance NUMERIC;")
            conn.commit()
            cur.close()
            db_pool.putconn(conn)
            print("✅ Tietokantarakenne ja L2-sarakkeet varmistettu onnistuneesti.")
        except Exception as e:
            print(f"⚠️ Huomautus tietokannan tarkistuksesta: {e}")

    # Rekisteröidään ja tilataan L2 tilauskirjavirta MT5:stä kaikille seuratuille pareille
    for pair in PAIRS:
        if mt5.market_book_add(pair):
            print(f"   📊 L2 DOM -tilauskirjavirta aktivoitu: {pair}")
        else:
            print(f"   ⚠️ Broker ei tarjoa aitoa DOM-dataa: {pair} -> Kytketään Kineettinen Synteettinen JAX-DOM (Kyle's Lambda).")

    ai_engines = {pair: LiquidityDensityEngine(learning_rate=0.05) for pair in PAIRS}
    ai_states = {pair: (ai_engines[pair].params, ai_engines[pair].opt_state) for pair in PAIRS}
    
    notifier = TelegramNotifier()
    last_wick_alert = {pair: 0 for pair in PAIRS}
    last_macd_alert = {pair: 0 for pair in PAIRS}

    print("✅ Moottori alustettu LIVENÄ. Seurataan aitoja muureja sekä M30 MACD-käännöksiä...")

    while True:
        try:
            for pair in PAIRS:
                if not mt5.symbol_select(pair, True):
                    continue
                
                rates = mt5.copy_rates_from_pos(pair, ACTIVE_TIMEFRAME, 0, 100)
                if rates is None or len(rates) < 50: 
                    continue
                
                of = hae_order_flow_stats(pair)
                if of:
                    pip_size = get_pip_size(pair)
                    actual_pips = (rates[-1]['close'] - rates[-1]['open']) / pip_size
                    
                    # 1. Ajetaan MACD-laskenta
                    closes = [float(r['close']) for r in rates]
                    _, _, histogram = TrendEngine.calculate_macd_all(closes)
                    latest_macd = float(histogram[-1])
                    
                    # 2. Ajetaan hylkäysskanneri SULKEUTUNEELLE kynttilälle (rates[-2])
                    closed_candle = rates[-2]
                    closed_bar_time = int(closed_candle['time'])
                    wick_signal, wick_pct = WickScanner.detect_rejection(
                        closed_candle['open'], closed_candle['high'], closed_candle['low'], closed_candle['close']
                    )
                    
                    # Varmistetaan MACD-risteys sulkeutuneista kynttilöistä (rates[-2] vs rates[-3])
                    closed_macd = float(histogram[-2])
                    prev_closed_macd = float(histogram[-3])
                    
                    macd_crossed_bullish = (prev_closed_macd < 0 and closed_macd >= 0)
                    macd_crossed_bearish = (prev_closed_macd > 0 and closed_macd <= 0)
                    
                    # 3. LUETAAN TAI LASKETAAN HYBRID-DOM (Aito Level 2 TAI Synteettinen Kyle's Lambda)
                    l2_imbalance = hae_l2_dom_imbalance(pair)
                    
                    if l2_imbalance is not None:
                        unified_dom_imbalance = float(l2_imbalance)
                        stiffness_val = 0.0 
                        l2_status_str = f"Aito L2 DOM: {unified_dom_imbalance:+.4f}"
                    else:
                        synthetic_imbalance, stiffness = calculate_synthetic_dom(
                            jnp.array(of["delta"], dtype=jnp.float32),
                            jnp.array(of["volume"], dtype=jnp.float32),
                            jnp.array(actual_pips, dtype=jnp.float32)
                        )
                        unified_dom_imbalance = float(synthetic_imbalance)
                        stiffness_val = float(stiffness)
                        l2_status_str = f"Synth DOM: {unified_dom_imbalance:+.4f} (Stiff: {stiffness_val:.1f})"
                    
                    imbalance = unified_dom_imbalance
                    
                    params, opt_state = ai_states[pair]
                    cost_per_pip = LiquidityDensityEngine.calculate_cost_per_pip(
                        jnp.array([rates[-1]['high']]), jnp.array([rates[-1]['low']]), 
                        jnp.array([of["volume"]]), pip_size
                    )[0]
                    
                    predicted = LiquidityDensityEngine.predict_movement(params, imbalance, cost_per_pip, of["volume"])
                    new_params, new_opt_state, loss = ai_engines[pair].update_learning(
                        params, opt_state, imbalance, cost_per_pip, of["volume"], actual_pips
                    )
                    ai_states[pair] = (new_params, new_opt_state)
                    friction_val = float(new_params['friction_weight'])
                    
                    db_pair_name = "US30" if pair == "US_30" else pair
                    
                    # TALLENNETAAN KAIKKI TIEDOT KANTAAN
                    save_to_db(
                        db_pair_name, float(predicted), float(actual_pips), float(friction_val), 
                        float(loss), float(of["volume"]), latest_macd, float(wick_signal), float(wick_pct),
                        unified_dom_imbalance
                    )
                    
                    inertia_val = float(calculate_triad_inertia(float(of["volume"]), latest_macd, friction_val))
                    
                    print(f"🧠 [{pair}] Live | {l2_status_str} | MACD: {latest_macd:.6f} | Wick: {wick_signal}")
                    
                    stiffness_text = f"\n🏋️ *Seinän kireys (Stiffness):* {stiffness_val:.1f}" if l2_imbalance is None else ""
                    
                    # --- HÄLYTYS A: WICK PYYHKÄISY ---
                    if wick_signal != 0.0 and last_wick_alert[pair] != closed_bar_time:
                        l2_confirmed = True
                        if wick_signal == -1.0 and unified_dom_imbalance > 0.4: l2_confirmed = False  
                        if wick_signal == 1.0 and unified_dom_imbalance < -0.4: l2_confirmed = False  
                        
                        if l2_confirmed:
                            last_wick_alert[pair] = closed_bar_time
                            suunta_emoji = "🟢 LONG" if wick_signal == 1.0 else "🔴 SHORT"
                            
                            alert_msg = (
                                f"🎯 *VOJKER L2 CONFLUENCE ALERT: {pair}*\n\n"
                                f"⚡ *Rakenne:* {suunta_emoji} (Hylkäys: {wick_pct:.1f}%)\n"
                                f"📊 *Tilauskirjan paine:* {unified_dom_imbalance:+.4f} {'(Synteettinen)' if l2_imbalance is None else '(Aito L2)'}{stiffness_text}\n"
                                f"🎛️ *Kitkakerroin:* {friction_val:.4f}\n"
                                f"⛓️ *Inertia-indeksi:* {inertia_val:.2f}\n"
                                f"📈 *MACD Histogram:* {latest_macd:.6f}\n\n"
                                f"💡 _Mekaaninen triangeli lukittu JA tilauskirjan seinillä vahvistettu. Ansat suodatettu._"
                            )
                            notifier.send_alert(alert_msg)
                        else:
                            print(f"🛡️ [VAOFE SUOJAUS] {pair} Wick-signaali peruttu! Stop-Hunt ansa havaittu.")
                            
                    # --- HÄLYTYS B: M30 MACD VÄRINVAIHTO ---
                    if (macd_crossed_bullish or macd_crossed_bearish) and last_macd_alert[pair] != closed_bar_time:
                        l2_confirmed = True
                        if macd_crossed_bearish and unified_dom_imbalance > 0.4: l2_confirmed = False
                        if macd_crossed_bullish and unified_dom_imbalance < -0.4: l2_confirmed = False
                        
                        if l2_confirmed:
                            last_macd_alert[pair] = closed_bar_time
                            suunta_emoji = "🟢 LONG (Vihreä)" if macd_crossed_bullish else "🔴 SHORT (Punainen)"
                            
                            alert_msg = (
                                f"📈 *VOJKER MACD M30 VÄRINVAIHTO: {pair}*\n\n"
                                f"⚡ *Momentum Kääntyi:* {suunta_emoji}\n"
                                f"📊 *Tilauskirjan paine:* {unified_dom_imbalance:+.4f} {'(Synteettinen)' if l2_imbalance is None else '(Aito L2)'}{stiffness_text}\n"
                                f"🎛️ *Kitkakerroin:* {friction_val:.4f}\n"
                                f"⛓️ *Inertia-indeksi:* {inertia_val:.2f}\n\n"
                                f"💡 _MACD (M30) on vaihtanut värin nollalinjan yli. Jos olet odottanut vahvistusta treidille, se on tässä._"
                            )
                            notifier.send_alert(alert_msg)
                        else:
                            print(f"🛡️ [VAOFE SUOJAUS] {pair} MACD-värivaihto peruttu! Tilauskirjan seinä ei tue käännöstä.")
            
            time.sleep(CHECK_INTERVAL_SECONDS)
        except Exception as e:
            print(f"❌ Virhe pääsilmukassa: {e}")
            time.sleep(10)

if __name__ == "__main__":
    start_live_engine()