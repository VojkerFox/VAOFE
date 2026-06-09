import sys
import os
import time
import requests
import psycopg2
from psycopg2 import pool
import MetaTrader5 as mt5
from datetime import datetime
import jax.numpy as jnp
from dotenv import load_dotenv

# Varmistetaan että projektin juuri on polussa
load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.compute.dynamic_threshold import DynamicThresholdCalculator
from src.compute.delta_calc import DeltaEngine
from src.compute.trend_engine import TrendEngine
from src.compute.wick_detector import WickScanner
from src.compute.liquidity_density import LiquidityDensityEngine

# --- ASETUKSET ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP"]
ACTIVE_TIMEFRAME = mt5.TIMEFRAME_M15
TF_STRING = "M15"
CHECK_INTERVAL_SECONDS = 5 

# Tietokantayhteys (varmista että .env tiedostossa on oikeat arvot)
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "vofe_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "salasana"),
    "host": "localhost",
    "port": "5432"
}

# Alustetaan yhteyspooli
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(1, 10, **DB_CONFIG)
except Exception as e:
    print(f"❌ SQL Pool virhe: {e}")
    db_pool = None

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram-virhe: {e}")

def save_to_db(pair, predicted, actual, friction, loss):
    if not db_pool: return
    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ai_learning_logs (pair, predicted_pips, actual_pips, friction_weight, loss) VALUES (%s, %s, %s, %s, %s)",
            (pair, predicted, actual, friction, loss)
        )
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"❌ Tietokantavirhe: {e}")
    finally:
        if conn: db_pool.putconn(conn)

def hae_order_flow_stats(pair):
    last_tick = mt5.symbol_info_tick(pair)
    if last_tick is None: return None
    ticks = mt5.copy_ticks_from(pair, int(last_tick.time), 1000, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) < 10: return None
    
    volumes = jnp.array(ticks['volume'], dtype=jnp.float32)
    history_volumes = volumes[:-5] if len(volumes) > 5 else volumes
    recent_ticks = ticks[-min(50, len(ticks)):]
    
    threshold = DynamicThresholdCalculator.calculate(history_volumes, factor=2.0)
    recent_volume_total = float(jnp.sum(jnp.array(recent_ticks['volume'], dtype=jnp.float32)))
    
    bid_vol, ask_vol = 0.0, 0.0
    for i in range(1, len(recent_ticks)):
        if recent_ticks[i]['ask'] > recent_ticks[i-1]['ask']: bid_vol += recent_ticks[i]['volume']
        elif recent_ticks[i]['bid'] < recent_ticks[i-1]['bid']: ask_vol += recent_ticks[i]['volume']
    
    delta = float(DeltaEngine.calculate_delta(jnp.array(bid_vol), jnp.array(ask_vol)))
    return {"volume": recent_volume_total, "threshold": float(threshold), "delta": delta}

def start_live_engine():
    print(f"🚀 VAOFE Engine ({TF_STRING}) käynnistyy...")
    if not mt5.initialize():
        print("❌ MT5 alustus epäonnistui!")
        return

    for pair in PAIRS:
        mt5.symbol_select(pair, True)
    
    ai_engines = {pair: LiquidityDensityEngine(learning_rate=0.05) for pair in PAIRS}
    ai_states = {pair: (ai_engines[pair].params, ai_engines[pair].opt_state) for pair in PAIRS}
    viimeisimmät_histogrammit = {pair: None for pair in PAIRS}
    viimeisimmät_kynttilä_ajat = {pair: 0 for pair in PAIRS}

    print("✅ Moottori alustettu. Siirrytään pääsilmukkaan.")
    
    while True:
        try:
            for pair in PAIRS:
                rates = mt5.copy_rates_from_pos(pair, ACTIVE_TIMEFRAME, 0, 100)
                if rates is None or len(rates) < 50: continue
                
                nykyinen_bar_aika = int(rates[-1]['time'])
                if nykyinen_bar_aika > viimeisimmät_kynttilä_ajat[pair]:
                    of = hae_order_flow_stats(pair)
                    if of:
                        open_p, high_p, low_p, close_p = rates[-1]['open'], rates[-1]['high'], rates[-1]['low'], rates[-1]['close']
                        pip_size = 0.01 if "JPY" in pair else 0.0001
                        actual_pips = (close_p - open_p) / pip_size
                        
                        params, opt_state = ai_states[pair]
                        cost_per_pip = LiquidityDensityEngine.calculate_cost_per_pip(jnp.array([high_p]), jnp.array([low_p]), jnp.array([of["volume"]]), pip_size)[0]
                        imbalance = of["delta"] / (of["volume"] + 1e-7)
                        
                        predicted = LiquidityDensityEngine.predict_movement(params, imbalance, cost_per_pip, of["volume"])
                        new_params, new_opt_state, loss = ai_engines[pair].update_learning(params, opt_state, imbalance, cost_per_pip, of["volume"], actual_pips)
                        ai_states[pair] = (new_params, new_opt_state)
                        
                        save_to_db(pair, float(predicted), actual_pips, float(new_params['friction_weight']), float(loss))
                        print(f"🧠 [AI-Oppiminen] {pair} Ennuste: {float(predicted):.1f}p | Todellinen: {actual_pips:.1f}p | Kitka: {float(new_params['friction_weight']):.4f}")
                    
                    viimeisimmät_kynttilä_ajat[pair] = nykyinen_bar_aika
            
            time.sleep(CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n🛑 VAOFE sammutetaan.")
            break
        except Exception as e:
            print(f"❌ Virhe pääsilmukassa: {e}")
            time.sleep(5)

    mt5.shutdown()
    if db_pool: db_pool.closeall()

if __name__ == "__main__":
    start_live_engine()