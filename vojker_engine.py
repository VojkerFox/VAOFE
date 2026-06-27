import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import time
import MetaTrader5 as mt5
import psycopg2

# ==========================================
# 0. ASETUKSET
# ==========================================
DB_HOST = "localhost"
DB_NAME = "vofe_db"
DB_USER = "postgres"
DB_PASS = "password"
DB_PORT = "5432"

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "XAUUSD"]
NUM_PAIRS = len(PAIRS)
BUFFER_SIZE = 500

# ==========================================
# 1. TIETOKANNAN ALUSTUS (Keskushermosto)
# ==========================================
def init_db():
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT)
        cur = conn.cursor()
        # Luodaan taulu pelkille raaoille signaaleille, joita AI voi myöhemmin lukea
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vojker_signals (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR(20),
                direction VARCHAR(20),
                price NUMERIC(12,5),
                mass NUMERIC(12,5),
                velocity NUMERIC(12,5),
                energy NUMERIC(12,5),
                status VARCHAR(20) DEFAULT 'PENDING'
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("🟢 PostgreSQL 'vojker_signals' tietokanta yhdistetty ja valmiina.")
    except Exception as e:
        print(f"🔴 PostgreSQL Virhe: {e}")
        exit()

def log_signal_to_db(pair, direction, price, mass, velocity, energy):
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO vojker_signals (symbol, direction, price, mass, velocity, energy)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (pair, direction, float(price), float(mass), float(velocity), float(energy)))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Tietokantavirhe tallennuksessa: {e}")

# ==========================================
# 2. JAX FYSIIKKA (Täsmälleen sama kuin ennen)
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
        tick_vols, tick_prices, tick_flags, state["prev_price"], state["prev_time_msc"], tick_times_msc
    )
    
    idx = state["buffer_index"]
    new_energy_buffer = state["energy_buffer"].at[:, idx].set(eks)
    new_price_buffer = state["price_buffer"].at[:, idx].set(tick_prices)
    new_idx = (idx + 1) % buffer_size
    new_counts = state["tick_counts"] + 1
    
    is_signal_core = hardware_optimized_tqg_batch(new_energy_buffer, eks)
    is_warmed_up = new_counts > buffer_size
    has_min_energy = eks > 0.005 
    
    local_high = jnp.max(new_price_buffer, axis=1)
    local_low = jnp.min(new_price_buffer, axis=1)
    price_range = jnp.maximum(local_high - local_low, 0.0001)
    
    is_near_ssl = tick_prices <= (local_low + 0.10 * price_range)
    is_near_bsl = tick_prices >= (local_high - 0.10 * price_range)
    
    is_bullish_reversal = (masses > 0.0) & (velocities > 0.0001) & is_near_ssl
    is_bearish_reversal = (masses < 0.0) & (velocities < -0.0001) & is_near_bsl
    
    is_reversal = (is_bullish_reversal | is_bearish_reversal) & is_signal_core & has_min_energy
    
    # 0 = Hold, 2 = Execute
    signals = jnp.where(is_warmed_up & is_reversal, 2, 0)
    
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
# 3. MT5 TIEDONKERUU
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
# 4. PÄÄMOOTTORIN LUPPI (Headless)
# ==========================================
def main():
    print("\n⚡ VOJKER-One KINEETTINEN MOOTTORI (HEADLESS) ⚡")
    print("Käynnistetään JAX XLA -kääntäjä...\n")
    
    if not mt5.initialize():
        print("🔴 MT5 Yhteyttä ei saatu!")
        return
    print(f"🟢 MT5 Yhdistetty: {mt5.account_info().company}")

    for pair in PAIRS:
        mt5.symbol_select(pair, True)

    init_db()
    
    state = init_bombe_state()
    prices = jnp.zeros(NUM_PAIRS, dtype=jnp.float32)
    cooldowns = {pair: 0 for pair in PAIRS}

    print("\nSkannaus käynnissä. Paina Ctrl+C lopettaaksesi.\n")
    
    try:
        while True:
            tick_prices, tick_vols, tick_flags, tick_times_msc = fetch_live_mt5_data(prices)
            prices = tick_prices
            
            state, signals, masses, velocities, eks = bombe_step_batch(
                state, tick_prices, tick_vols, tick_flags, tick_times_msc
            )

            counts = np.array(state["tick_counts"])
            
            for i, pair in enumerate(PAIRS):
                sig_val = int(signals[i])
                c = int(counts[i])
                
                # Tulostetaan 100 tickin välein varmistus, että moottori käy
                if c % 100 == 0 and c <= BUFFER_SIZE:
                    print(f"[{pair}] Warmup: {c}/{BUFFER_SIZE}")

                if sig_val == 2:
                    current_time = time.time()
                    if current_time - cooldowns[pair] > 900:  # 15 min cooldown
                        cooldowns[pair] = current_time
                        m, v, ek, p = float(masses[i]), float(velocities[i]), float(eks[i]), float(tick_prices[i])
                        direction = "OSTA" if m > 0 else "MYY"
                        
                        print(f"🎯 ANOMALIA HAVAITTU: {pair} | {direction} | P: {p:.5f} | Ek: {ek:.4f}")
                        
                        # Kirjoitetaan tietokantaan, josta Moduuli B (Tekoäly) löytää sen
                        log_signal_to_db(pair, direction, p, m, v, ek)

            time.sleep(1.0) # Moottori hengittää

    except KeyboardInterrupt:
        print("\n🛑 Moottori sammutettu turvallisesti.")
        mt5.shutdown()

if __name__ == "__main__":
    main()