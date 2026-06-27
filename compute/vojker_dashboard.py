import streamlit as st
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import time
import requests
import ollama
import MetaTrader5 as mt5
import threading
import pandas as pd
import psycopg2
from psycopg2 import extras

# ==========================================
# 0. ASETUKSET & LIITÄNNÄT
# ==========================================
TELEGRAM_BOT_TOKEN = "8658806596:AAH3jFlP7LKuHY8wMXBt02kD9UMC9SacZRI"
TELEGRAM_CHAT_ID = "260783230"
OLLAMA_MODEL = "gemma2:2b"

# POSTGRESQL TIETOKANNASETUKSET (Päivitetty oikeilla tiedoilla)
DB_HOST = "localhost"
DB_NAME = "vofe_db"
DB_USER = "postgres"
DB_PASS = "password" 
DB_PORT = "5432"

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "XAUUSD"]
NUM_PAIRS = len(PAIRS)
BUFFER_SIZE = 500

st.set_page_config(page_title="VOJKER-One EXECUTIVE CONTROL", layout="wide", page_icon="⚡")

# ==========================================
# Tietokannan alustusfunktiot
# ==========================================
def init_db():
    """Alustaa PostgreSQL-taulun kaupoille jos sitä ei ole olemassa."""
    try:
        # TÄSSÄ KÄYTETÄÄN NYT YLHÄÄLLÄ MÄÄRITELTYJÄ MUUTTUJIA (DB_HOST, DB_NAME jne.)
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vojker_trades (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR(20),
                direction VARCHAR(20),
                entry_price NUMERIC(12,5),
                stop_loss NUMERIC(12,5),
                take_profit NUMERIC(12,5),
                energy NUMERIC(12,5)
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"PostgreSQL ei saavutettavissa, käytetään sisäistä muistia. Virhe: {e}")
        return False

def save_trade_to_db(pair, direction, price, sl, tp, energy):
    """Tallentaa kaupan PostgreSQL:ään tai session_stateen fallbackina."""
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO vojker_trades (symbol, direction, entry_price, stop_loss, take_profit, energy)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (pair, direction, price, sl, tp, energy))
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        # Fallback välimuistiin
        if "local_db_fallback" not in st.session_state:
            st.session_state.local_db_fallback = []
        st.session_state.local_db_fallback.append({
            "Aika": time.strftime('%Y-%m-%d %H:%M:%S'),
            "Pari": pair, "Suunta": direction, "Hinta": round(price, 5),
            "SL": round(sl, 5), "TP": round(tp, 5), "Kineettinen Energia": round(energy, 5)
        })

# ==========================================
# 1. JAX FYSIIKKA (Korjattu ja varmistettu rajasuodatin)
# ==========================================
def init_bombe_state(buffer_size=BUFFER_SIZE):
    return {
        "energy_buffer": jnp.zeros((NUM_PAIRS, buffer_size), dtype=jnp.float32),
        "price_buffer": jnp.zeros((NUM_PAIRS, buffer_size), dtype=jnp.float32),
        "buffer_index": jnp.int32(0),
        "prev_price": jnp.zeros(NUM_PAIRS, dtype=jnp.float32),
        "prev_time_msc": jnp.zeros(NUM_PAIRS, dtype=jnp.float32),
        "tick_counts": jnp.zeros(NUM_PAIRS, dtype=jnp.int32)
    }

@jax.jit
def calculate_tick_physics_batch(volumes, prices, flags, prev_prices, prev_times_msc, current_times_msc):
    dt = jnp.maximum((current_times_msc - prev_times_msc) / 1000.0, 1e-3)
    masses = volumes * flags
    velocities = (prices - prev_prices) / dt
    kinetic_energies = 0.5 * jnp.abs(masses) * (velocities ** 2)
    return masses, velocities, kinetic_energies

@jax.jit
def hardware_optimized_tqg_batch(energy_buffers, current_energies):
    top_energies, _ = jax.lax.top_k(energy_buffers, 4) 
    thresholds = top_energies[:, -1]
    return (current_energies >= thresholds) & (current_energies > 0.0)

@partial(jax.jit, static_argnums=(5,))
def bombe_step_batch(state, tick_prices, tick_vols, tick_flags, tick_times_msc, buffer_size=BUFFER_SIZE):
    masses, velocities, eks = calculate_tick_physics_batch(
        tick_vols, tick_prices, tick_flags, 
        state["prev_price"], state["prev_time_msc"], tick_times_msc
    )
    
    idx = state["buffer_index"]
    new_energy_buffer = state["energy_buffer"].at[:, idx].set(eks)
    new_price_buffer = state["price_buffer"].at[:, idx].set(tick_prices)
    new_idx = (idx + 1) % buffer_size
    new_counts = state["tick_counts"] + 1
    
    is_signal_core = hardware_optimized_tqg_batch(new_energy_buffer, eks)
    is_warmed_up = new_counts > buffer_size
    has_min_energy = eks > 0.005 
    
    # AVARUUDELLINEN SUODATIN (Estää haamusignaalit keskellä ei-mitään)
    local_high = jnp.max(new_price_buffer, axis=1)
    local_low = jnp.min(new_price_buffer, axis=1)
    price_range = jnp.maximum(local_high - local_low, 0.0001)
    
    is_near_ssl = tick_prices <= (local_low + 0.10 * price_range)
    is_near_bsl = tick_prices >= (local_high - 0.10 * price_range)
    
    is_bullish_reversal = (masses > 0.0) & (velocities > 0.0001) & is_near_ssl
    is_bearish_reversal = (masses < 0.0) & (velocities < -0.0001) & is_near_bsl
    
    is_reversal = (is_bullish_reversal | is_bearish_reversal) & is_signal_core & has_min_energy
    is_friction = (jnp.abs(velocities) < 0.0001) & (is_near_ssl | is_near_bsl) & is_signal_core & has_min_energy
    
    signals = jnp.where(is_warmed_up & is_reversal, 2, 
              jnp.where(is_warmed_up & is_friction, 1, 0))
    
    new_state = {
        "energy_buffer": new_energy_buffer, 
        "price_buffer": new_price_buffer,
        "buffer_index": new_idx,
        "prev_price": tick_prices, 
        "prev_time_msc": tick_times_msc,
        "tick_counts": new_counts
    }
    return new_state, signals, masses, velocities, eks

# ==========================================
# 2. MT5 AUTOMAATTINEN TOIMEKSIANTO & AI ARKKITEHTUURI
# ==========================================
def execute_mt5_order(pair, order_type, sl, tp):
    """Hakee oikean Ask/Bid hinnan ja neuvottelee välittäjän kanssa."""
    symbol_info = mt5.symbol_info(pair)
    if symbol_info is None:
        return False, "Symbolia ei löytynyt MT5:stä."
        
    tick = mt5.symbol_info_tick(pair)
    if tick is None:
        return False, "MT5 ei anna Tick-dataa suoritukseen."

    # 1. Tarkka markkinahinta (Ask = Ostetaan, Bid = Myydään)
    exec_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    # 2. Välittäjän Täyttöpolitiikan (Filling Mode) ratkaisu
    # Prop Firmit käyttävät yleensä IOC tai FOK. 
    filling_type = mt5.ORDER_FILLING_IOC
    if (symbol_info.filling_mode & 1): # Jos FOK sallittu
        filling_type = mt5.ORDER_FILLING_FOK
    elif (symbol_info.filling_mode & 2): # Jos IOC sallittu
        filling_type = mt5.ORDER_FILLING_IOC

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pair,
        "volume": 0.01,
        "type": order_type,
        "price": exec_price,
        "sl": sl,
        "tp": tp,
        "deviation": 20, # Sallitaan kullan volatiliteetti
        "magic": 202606,
        "comment": "VOJKER EXEC",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        return False, "MT5 ei vastannut (Yhteysvirhe)"
        
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        # Palautetaan tarkka virhekoodi (esim. 10015 = Invalid Price, 10016 = Invalid Stops)
        return False, f"Hylätty. Virhekoodi: {result.retcode} ({result.comment})"
        
    return True, f"HYVÄKSYTTY @ {result.price}"


def process_signal_agentic_workflow(pair, price, mass, velocity, energy):
    suunta = "OSTA (LONG)" if mass > 0 else "MYY (SHORT)"
    mt5_type = mt5.ORDER_TYPE_BUY if mass > 0 else mt5.ORDER_TYPE_SELL
    
    # 1. TARKKA RISKIMATEMATIIKKA
    if "JPY" in pair: pip_size = 0.01
    elif "XAU" in pair: pip_size = 0.10
    else: pip_size = 0.0001

    tp_taso = price + (5.0 * pip_size) if mass > 0 else price - (5.0 * pip_size)
    sl_taso = price - (2.0 * pip_size) if mass > 0 else price + (2.0 * pip_size)

    # 2. SUORITA AUTOMAATTINEN KAUPPA MT5:ssä
    success, exec_status = execute_mt5_order(pair, mt5_type, sl_taso, tp_taso)

    # 3. HISTORIOI POSTGRESQL (Tallenna vain jos meni läpi)
    if success:
        save_trade_to_db(pair, suunta, price, sl_taso, tp_taso, energy)

    # 4. TEKOÄLYTULKINTA JA TELEGRAM-REFLEKSI (Pakotettu rooli)
    try:
        profiler_sys = "Olet järjestelmäloki. Sinun on pakko antaa analyysi kineettisestä voimasta. Et saa pyytää lisätietoja. Vastaa yhdellä lauseella."
        profiler_user = f"Suunta {suunta}, massa {mass:.1f}, nopeus {velocity:.5f}. Tulkitse tämä kineettinen purkaus."
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'system', 'content': profiler_sys}, {'role': 'user', 'content': profiler_user}])
        ai_insight = response['message']['content'].strip()
    except Exception:
        ai_insight = "Fysiikka vahvistettu matemaattisen suodattimen kautta."

    tg_msg = f"⚡ *VOJKER EXECUTIVE STRIKE* ⚡\n\n"
    tg_msg += f"🎯 *KÄSKY:* 0.01 Lot {suunta}\n"
    tg_msg += f"⚙️ *MT5 STATUS:* {exec_status}\n\n" # TÄMÄ KERTOO JOS VÄLITTÄJÄ HYLKÄÄ!
    tg_msg += f"📍 *Pari:* {pair}\n"
    tg_msg += f"💲 *Hinta:* {price:.5f}\n"
    tg_msg += f"🛑 *Stop Loss:* {sl_taso:.5f}\n"
    tg_msg += f"✅ *Take Profit:* {tp_taso:.5f}\n\n"
    tg_msg += f"🧠 *AI Analyysi:*\n_{ai_insight}_"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": tg_msg, "parse_mode": "Markdown"}, timeout=5)
    except Exception: pass

# ==========================================
# 3. DATA & SALKKUTELEMETRIA PIPELINE
# ==========================================
def fetch_live_mt5_data(state_prices):
    prices, vols, flags, times = [], [], [], []
    for i, pair in enumerate(PAIRS):
        tick = mt5.symbol_info_tick(pair)
        if tick is None:
            prices.append(state_prices[i])
            vols.append(0.0)
            flags.append(1.0)
            times.append(time.time() * 1000)
        else:
            current_price = float(tick.last) if getattr(tick, 'last', 0.0) > 0.0 else (float(tick.bid) + float(tick.ask)) / 2.0
            prices.append(current_price)
            vol = getattr(tick, 'volume_real', getattr(tick, 'volume', 0))
            vols.append(float(vol) if vol > 0 else 1.0)
            flag = 1.0 if (tick.flags & mt5.TICK_FLAG_BUY) else (-1.0 if (tick.flags & mt5.TICK_FLAG_SELL) else (1.0 if current_price >= state_prices[i] else -1.0))
            flags.append(flag)
            times.append(float(tick.time_msc))
    return jnp.array(prices), jnp.array(vols), jnp.array(flags), jnp.array(times)

# ==========================================
# 4. EXECUTIVE CONTROL PANEL (Streamlit UI)
# ==========================================
def main():
    st.title("⚡ VOJKER-One EXECUTIVE CONTROL")
    
    # Tarkistetaan tietokantayhteys
    db_active = init_db()
    if db_active:
        st.sidebar.success("💻 PostgreSQL: YHDISTETTY")
    else:
        st.sidebar.warning("💻 PostgreSQL: OFFLINE (Käytetään välimuistia)")

    if not mt5.initialize():
        st.error("🔴 MT5 yhteyttä ei saatu!")
        return

    # Salkun tilan reaaliaikainen haku suoraan MT5-rajapinnasta
    acc_info = mt5.account_info()
    balance = acc_info.balance
    equity = acc_info.equity
    floating_pnl = acc_info.profit
    pnl_percent = (floating_pnl / balance) * 100

    # PNL JA SALKKUTELEMETRIA LAATIKOT
    st.subheader("Tilin reaaliaikainen telemetria")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tilin tase (Balance)", f"{balance:.2f} €")
    m2.metric("Oma pääoma (Equity)", f"{equity:.2f} €")
    
    # Väritetään PnL dynaamisesti tilanteen mukaan
    pnl_color = "inverse" if floating_pnl < 0 else "normal"
    m3.metric("Avoimet voitot/tappiot (PnL)", f"{floating_pnl:.2f} €", delta=None)
    m4.metric("PnL Prosentteina", f"{pnl_percent:.3f} %")

    # Pakotetaan parit päälle
    for pair in PAIRS:
        mt5.symbol_select(pair, True)

    if "state" not in st.session_state:
        st.session_state.state = init_bombe_state()
        st.session_state.prices = jnp.zeros(NUM_PAIRS, dtype=jnp.float32)
        st.session_state.cooldowns = {pair: 0 for pair in PAIRS}

    live_mode = st.sidebar.toggle("🔴 KÄYNNISTÄ AUTOMAATTINEN TREIDAUS", value=False)

    st.markdown("---")
    
    # Jakautuminen: Matriisi ja Tilastot allekkain
    master_frame = st.empty()
    
    st.markdown("---")
    st.subheader("Järjestelmän laukaisuhistoria ja aktiiviset positiot")
    
    pos_table = st.empty()
    history_table = st.empty()

    if live_mode:
        while True:
            # 1. Päivitetään hinta ja JAX fysiikka
            tick_prices, tick_vols, tick_flags, tick_times_msc = fetch_live_mt5_data(st.session_state.prices)
            st.session_state.prices = tick_prices
            
            st.session_state.state, signals, masses, velocities, eks = bombe_step_batch(
                st.session_state.state, tick_prices, tick_vols, tick_flags, tick_times_msc
            )

            counts = np.array(st.session_state.state["tick_counts"])
            
            # 2. Piirretään TQG Matrix reaaliajassa
            with master_frame.container():
                cols = st.columns(4)
                for i, pair in enumerate(PAIRS):
                    sig_val = int(signals[i])
                    m, v, ek, p = float(masses[i]), float(velocities[i]), float(eks[i]), float(tick_prices[i])
                    c = int(counts[i])

                    with cols[i % 4]:
                        st.markdown(f"### {pair}")
                        st.metric("Hinta (Mid)", f"{p:.5f}")
                        if c < BUFFER_SIZE:
                            st.info(f"⚙️ WARMUP ({c}/{BUFFER_SIZE})")
                        elif sig_val == 0:
                            st.error(f"🔴 HOLD | Ek: {ek:.4f}")
                        elif sig_val == 1:
                            st.warning(f"🟡 ABSORPTION | V: {v:.4f}")
                        else:
                            st.success(f"🟢 EXECUTE ORDER 0.01")

                    # 3. Toimeksiantoautomaatio asynkronisessa taustasäikeessä
                    current_time = time.time()
                    if sig_val == 2 and (current_time - st.session_state.cooldowns[pair] > 900):
                        st.session_state.cooldowns[pair] = current_time
                        
                        threading.Thread(
                            target=process_signal_agentic_workflow, 
                            args=(pair, p, m, v, ek),
                            daemon=True
                        ).start()

            # 4. Päivitetään aktiivisten positioiden taulukko suoraan MT5:stä
            open_positions = mt5.positions_get()
            if open_positions:
                df_pos = pd.DataFrame(list(open_positions), columns=open_positions[0]._asdict().keys())
                df_pos_clean = df_pos[['symbol', 'type', 'volume', 'price_open', 'price_current', 'sl', 'tp', 'profit']]
                df_pos_clean.columns = ['Pari', 'Tyyppi (0=Buy, 1=Sell)', 'Koko (Lot)', 'Avaushinta', 'Nykyhinta', 'SL', 'TP', 'PnL (€)']
                # PÄIVITETTY: Käytetään width="stretch" varoituksen vaimentamiseksi
                pos_table.dataframe(df_pos_clean, width="stretch")
            else:
                pos_table.info("Ei avoimia positioita markkinalla juuri nyt.")

            # 5. Päivitetään signaalihistoriataulukko (Puhdas PostgreSQL-haku, ei Pandas-varoituksia)
            if db_active:
                try:
                    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT)
                    cur = conn.cursor()
                    cur.execute("SELECT timestamp as \"Aika\", symbol as \"Pari\", direction as \"Suunta\", entry_price as \"Hinta\", stop_loss as \"SL\", take_profit as \"TP\", energy as \"Energia\" FROM vojker_trades ORDER BY id DESC LIMIT 10")
                    rows = cur.fetchall()
                    cols = [desc[0] for desc in cur.description]
                    df_hist = pd.DataFrame(rows, columns=cols)
                    cur.close()
                    conn.close()
                    
                    history_table.dataframe(df_hist, width="stretch")
                except Exception:
                    pass
            else:
                if "local_db_fallback" in st.session_state and st.session_state.local_db_fallback:
                    history_table.dataframe(pd.DataFrame(st.session_state.local_db_fallback).iloc[::-1].head(10), width="stretch")
                else:
                    history_table.info("Odotetaan ensimmäistä kineettistä signaalia historioitavaksi...")

            time.sleep(1.0)

if __name__ == "__main__":
    main()