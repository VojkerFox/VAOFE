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

# --- ASETUKSET ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP",
    "GOLD", "SILVER", "US_30"
]

ACTIVE_TIMEFRAME = mt5.TIMEFRAME_M15
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

def save_to_db(pair, predicted, actual, friction, loss, volume, macd_hist, wick_signal, wick_pct):
    if not db_pool: return
    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO ai_learning_logs 
            (pair, predicted_pips, actual_pips, friction_weight, loss, volume, macd_hist, wick_signal, wick_pct) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (pair, predicted, actual, friction, loss, volume, macd_hist, wick_signal, wick_pct)
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

def start_live_engine():
    print(f"🚀 VAOFE SciML Moottori käynnistyy (Täysi konfluenssi-ajon alustus)...")
    if not mt5.initialize():
        print("❌ MT5 alustus epäonnistui!")
        return

    # FAIL-SAFE: Varmistetaan lennosta että tietokannassa on paikat uusille tiedoille
    if db_pool:
        try:
            conn = db_pool.getconn()
            cur = conn.cursor()
            cur.execute("ALTER TABLE ai_learning_logs ADD COLUMN IF NOT EXISTS volume NUMERIC;")
            cur.execute("ALTER TABLE ai_learning_logs ADD COLUMN IF NOT EXISTS macd_hist NUMERIC;")
            cur.execute("ALTER TABLE ai_learning_logs ADD COLUMN IF NOT EXISTS wick_signal NUMERIC;")
            cur.execute("ALTER TABLE ai_learning_logs ADD COLUMN IF NOT EXISTS wick_pct NUMERIC;")
            conn.commit()
            cur.close()
            db_pool.putconn(conn)
            print("✅ Tietokantarakenne varmistettu ja päivitetty onnistuneesti.")
        except Exception as e:
            print(f"⚠️ Huomautus tietokannan tarkistuksesta: {e}")

    ai_engines = {pair: LiquidityDensityEngine(learning_rate=0.05) for pair in PAIRS}
    ai_states = {pair: (ai_engines[pair].params, ai_engines[pair].opt_state) for pair in PAIRS}

    print("✅ Moottori alustettu livenä. Seurataan tilausvirtaa ja kynttilägeometriaa...")

    while True:
        try:
            for pair in PAIRS:
                if not mt5.symbol_select(pair, True):
                    continue
                
                # Haetaan riittävästi kynttilöitä (100 kpl) vakaata MACD-laskentaa varten
                rates = mt5.copy_rates_from_pos(pair, ACTIVE_TIMEFRAME, 0, 100)
                if rates is None or len(rates) < 50: 
                    continue
                
                of = hae_order_flow_stats(pair)
                if of:
                    pip_size = get_pip_size(pair)
                    actual_pips = (rates[-1]['close'] - rates[-1]['open']) / pip_size
                    
                    # 1. Ajetaan MACD-laskenta livenä
                    closes = [float(r['close']) for r in rates]
                    _, _, histogram = TrendEngine.calculate_macd_all(closes)
                    latest_macd = float(histogram[-1])
                    
                    # 2. Ajetaan institutionaalinen hännän hylkäysskanneri (Sweep)
                    open_p, high_p, low_p, close_p = rates[-1]['open'], rates[-1]['high'], rates[-1]['low'], rates[-1]['close']
                    wick_signal, wick_pct = WickScanner.detect_rejection(open_p, high_p, low_p, close_p)
                    
                    params, opt_state = ai_states[pair]
                    cost_per_pip = LiquidityDensityEngine.calculate_cost_per_pip(
                        jnp.array([rates[-1]['high']]), jnp.array([rates[-1]['low']]), 
                        jnp.array([of["volume"]]), pip_size
                    )[0]
                    
                    imbalance = of["delta"] / (of["volume"] + 1e-7)
                    
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
                        float(loss), float(of["volume"]), latest_macd, float(wick_signal), float(wick_pct)
                    )
                    print(f"🧠 [{pair}] Live | Kitka: {friction_val:.4f} | MACD Hist: {latest_macd:.6f} | Wick: {wick_signal} ({wick_pct:.1f}%)")
            
            time.sleep(CHECK_INTERVAL_SECONDS)
        except Exception as e:
            print(f"❌ Virhe pääsilmukassa: {e}")
            time.sleep(10)

if __name__ == "__main__":
    start_live_engine()