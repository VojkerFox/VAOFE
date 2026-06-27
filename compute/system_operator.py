import time
import MetaTrader5 as mt5
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import ollama

# --- 0. UI ASETUKSET ---
st.set_page_config(page_title="QUANT CORE", layout="wide")

# --- 1. ALUSTUS ---
if 'symbol' not in st.session_state: st.session_state.symbol = "EURUSD"
if 'last_analysis' not in st.session_state: st.session_state.last_analysis = ""

# MT5 ALUSTUS
if not mt5.initialize():
    st.error("MT5 Offline")
    st.stop()

def get_market_data(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    ticks = mt5.copy_ticks_range(symbol, int(time.time()) - 60, int(time.time()), mt5.COPY_TICKS_ALL)
    if rates is None or ticks is None: return None
    
    # MATRIISILASKENTA (BSL/SSL)
    highs = np.array([r['high'] for r in rates])
    lows = np.array([r['low'] for r in rates])
    bsl, ssl = np.max(highs), np.min(lows)
    
    # KINETIIKKA
    vols = ticks['volume']
    dp = np.diff(ticks['bid']) * 100000
    kinetic = np.mean(0.5 * vols[1:] * (dp / 0.1)**2) * 50
    
    kqi = int(max(0, 100 - (kinetic / 500)))
    action = "SELL" if ticks['bid'][-1] > np.median(highs) else "BUY"
    
    return {"price": ticks['bid'][-1], "bsl": bsl, "ssl": ssl, "kinetic": kinetic, "kqi": kqi, "action": action, "rates": rates}

def get_quant_briefing(data, symbol):
    prompt = f"""
    Role: Professional Quant Trader.
    Market: {symbol}. Price: {data['price']}.
    Liquidity: BSL at {data['bsl']:.5f}, SSL at {data['ssl']:.5f}.
    Action: {data['action']}. KQI: {data['kqi']}.
    
    Write 3 lines in English:
    1. Liquidity Bias: Where is the price currently drawing liquidity?
    2. Strategy: How to play the {data['action']} signal relative to BSL/SSL?
    3. Execution: Give one specific stop-loss price (based on liquidity levels).
    """
    try:
        res = ollama.chat(model='gemma:2b', messages=[{'role': 'user', 'content': prompt}])
        return res['message']['content']
    except: return "Engine Offline."

# --- 2. UI LOGIIKKA ---
st.title("🧠 QUANT CORE: MATRIX ENGINE")

new_symbol = st.selectbox("Asset:", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"], index=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"].index(st.session_state.symbol))
if new_symbol != st.session_state.symbol:
    st.session_state.symbol = new_symbol
    st.rerun()

data = get_market_data(st.session_state.symbol)

if data:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("KQI Index", f"{data['kqi']}/100")
        st.metric("Liquidity BSL", f"{data['bsl']:.5f}")
        st.metric("Liquidity SSL", f"{data['ssl']:.5f}")
        
        brief = get_quant_briefing(data, st.session_state.symbol)
        st.markdown(f"### 🛡️ Tactical Briefing\n{brief}")

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot([r['close'] for r in data['rates']], color='#38bdf8', label="Price")
        ax.axhline(data['bsl'], color='red', linestyle='--', label="BSL")
        ax.axhline(data['ssl'], color='green', linestyle='--', label="SSL")
        ax.legend()
        st.pyplot(fig)

    time.sleep(5)
    st.rerun()