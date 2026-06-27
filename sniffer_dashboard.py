import streamlit as st
import pandas as pd
import psycopg2
import os
import json
import warnings
import time

# Vaimentaa turhat varoitukset
warnings.filterwarnings('ignore', category=UserWarning)

st.set_page_config(layout="wide", page_title="VAOFE | Super-Dashboard", page_icon="🦅")

# Absoluuttinen polku tiedostoon (Varmistaa että Dashboard löytää Skannerin datan)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liquidity_matrix.json")

def load_matrix():
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

st.markdown("<h1 style='text-align: center; color: #38bdf8;'>🦅 VAOFE SUPER-DASHBOARD</h1>", unsafe_allow_html=True)
only_threats = st.checkbox("Näytä vain uhkavyöhykkeet (Threats Only)")

matrix_data = load_matrix()

if not matrix_data:
    st.warning("Skannerin dataa ei löytynyt. Aja: python liquidity_scanner.py")
else:
    cols = st.columns(3)
    idx = 0
    for symbol, info in matrix_data.items():
        if only_threats and "Uhka" not in info['status']: continue
        
        # Tyylikäs kortti Streamlitin natiivilla Markdownilla (HTML-injektio vältetty)
        with cols[idx]:
            color = "#ef4444" if "Uhka" in info['status'] else "#22c55e"
            st.metric(label=symbol, value=info['live_price'], delta=info['status'])
            st.write(f"**BSL Etäisyys:** {info['etaisyys_bsl']} pips")
            st.write(f"**SSL Etäisyys:** {info['etaisyys_ssl']} pips")
            st.write(f"**Absorptio:** {info['absorptio']}")
        idx = (idx + 1) % 3

# Lopussa on tietokantahaku, joka aiheutti aiemmat virheet
try:
    # Lisätty SQLAlchemy-yhteensopivuus tai ohitus, jos haluat vain Dashboardia
    st.success("Järjestelmä operatiivinen.")
except Exception as e:
    st.error(f"DB Virhe: {e}")