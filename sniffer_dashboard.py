import streamlit as st
import pandas as pd
import psycopg2
import os
import time
import requests
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Ladataan ympäristömuuttujat
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "vofe_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": "5432"
}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Alustetaan MT5 Dashboardia varten (read-only hinnan hakua varten)
if not mt5.initialize():
    st.error("MT5 alustus epäonnistui Dashboardissa. Varmista että MT5 on auki.")

# Sivun asetukset
st.set_page_config(layout="wide", page_title="VAOFE | Sniffer Performance")

# Tyylittely HTML-injektiolla
st.html("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h1 { color: #38bdf8 !important; font-weight: 800 !important; }
    .metric-card { background: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .metric-value { font-size: 48px; font-weight: bold; color: #f8fafc; margin: 10px 0; }
    .metric-label { font-size: 16px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .bull-text { color: #22c55e; font-weight: bold; }
    .bear-text { color: #ef4444; font-weight: bold; }
    </style>
""")

def varmista_tietokanta_rakenne():
    """Lisätään max_pips sarake tietokantaan, jos sitä ei vielä ole."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("ALTER TABLE sniffer_history ADD COLUMN IF NOT EXISTS max_pips NUMERIC DEFAULT 0.0;")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        pass # Rakenne on luultavasti jo kunnossa

varmista_tietokanta_rakenne()

def get_pip_size(pair: str) -> float:
    if "JPY" in pair: return 0.01
    if "GOLD" in pair or "XAU" in pair: return 0.1
    if "BTC" in pair: return 1.0
    if "US30" in pair or "SPX" in pair: return 1.0
    return 0.0001

def paivita_avoimet_treidit():
    """Käy läpi 'OPEN' tilassa olevat signaalit ja päivittää niiden PnL:n ja SL-tilan MT5:stä."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT id, symbol, direction, entry_price, sl_price, max_pips FROM sniffer_history WHERE status = 'OPEN'")
        avoimet = cur.fetchall()
        
        for trade in avoimet:
            t_id, symbol, direction, entry, sl, max_p = trade
            tick = mt5.symbol_info_tick(symbol)
            if not tick: continue
            
            # Valitaan hinta suunnan mukaan (karkeasti last, bid tai ask)
            current_price = tick.bid if direction == "BULL" else tick.ask
            pip_size = get_pip_size(symbol)
            
            # Lasketaan nykyinen pips-liike entrystä
            if direction == "BULL":
                current_pips = (current_price - float(entry)) / pip_size
                is_sl = current_price <= float(sl)
            else:
                current_pips = (float(entry) - current_price) / pip_size
                is_sl = current_price >= float(sl)
                
            # Päivitetään maksimipipsit (ei anneta sen pienentyä)
            current_max = float(max_p) if max_p is not None else 0.0
            new_max = max(current_max, current_pips)
            
            # Jos hinta osui stop lossiin, vaihdetaan tila
            new_status = 'SL HIT' if is_sl else 'OPEN'
            
            cur.execute("UPDATE sniffer_history SET max_pips = %s, status = %s WHERE id = %s", (new_max, new_status, t_id))
            
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Virhe hintojen päivityksessä: {e}")

def laheta_60min_raportti():
    """Kerää viimeisen 60 minuutin signaalit ja lähettää PnL-yhteenvedon Telegramiin."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        # Haetaan viimeisen 60 min aikana tulleet signaalit
        query = "SELECT max_pips, status FROM sniffer_history WHERE timestamp >= NOW() - INTERVAL '1 hour'"
        df_stats = pd.read_sql(query, conn)
        conn.close()
        
        if df_stats.empty:
            return # Ei lähetetä tyhjää raporttia, jos ei ole ollut signaaleja
            
        total = len(df_stats)
        sl_hits = len(df_stats[df_stats['status'] == 'SL HIT'])
        yli_3 = len(df_stats[df_stats['max_pips'] >= 3.0])
        yli_10 = len(df_stats[df_stats['max_pips'] >= 10.0])
        yli_15 = len(df_stats[df_stats['max_pips'] >= 15.0])
        
        msg = (
            f"📊 *VAOFE Tunnin PnL-Katsaus (Paper Trading)* 📊\n\n"
            f"Tunnin sisään uusia signaaleja: {total} kpl\n"
            f"-----------------------------------\n"
            f"📈 Liikkui > +3 pipsiä: {yli_3} kpl\n"
            f"🔥 Liikkui > +10 pipsiä: {yli_10} kpl\n"
            f"🚀 Liikkui > +15 pipsiä: {yli_15} kpl\n"
            f"🛑 Osuivat Stop Lossiin: {sl_hits} kpl\n"
            f"-----------------------------------\n"
            f"🦅 _Järjestelmä on Forward Test -tilassa._"
        )
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except Exception as e:
        pass

def hae_historia():
    """Hakee signaalihistorian tietokannasta."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = "SELECT timestamp, symbol, timeframe, direction, entry_price, sl_price, max_pips, status FROM sniffer_history ORDER BY timestamp DESC LIMIT 100"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Tietokantavirhe: {e}")
        return pd.DataFrame()

st.title("🎯 VAOFE Sniffer | Live Performance UX")
st.markdown("Reaaliaikainen seuranta ja Forward Testing -statistiikka.")

metrics_placeholder = st.empty()
table_placeholder = st.empty()

# Ajastin Telegram-raportille (Alustetaan nykyhetkeen)
last_report_time = time.time()
# Vaihda tämä 3600:aan kun haluat tunnin raportin (nyt 3600 = 60 min)
REPORT_INTERVAL_SECONDS = 3600 

while True:
    # 1. Haetaan MT5 livenä hinnat ja päivitetään tietokantaa
    paivita_avoimet_treidit()
    
    # 2. Tarkistetaan onko 60 minuuttia kulunut raportin lähetyksestä
    current_time = time.time()
    if current_time - last_report_time >= REPORT_INTERVAL_SECONDS:
        laheta_60min_raportti()
        last_report_time = current_time

    # 3. Haetaan päivitetty data näyttöä varten
    df = hae_historia()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        nyt = pd.Timestamp.now()
        tunti_sitten = nyt - pd.Timedelta(minutes=60)
        paiva_sitten = nyt - pd.Timedelta(hours=24)
        
        viim_60_min = df[df['timestamp'] >= tunti_sitten]
        viim_24_h = df[df['timestamp'] >= paiva_sitten]
        
        lkm_60min = len(viim_60_min)
        lkm_24h = len(viim_24_h)
        
        # UI Muotoilu
        df_display = df.copy()
        df_display['Aika'] = df_display['timestamp'].dt.strftime('%H:%M:%S')
        df_display['Pari'] = df_display['symbol'] + " (" + df_display['timeframe'] + ")"
        df_display['Suunta'] = df_display['direction'].apply(lambda x: "🟢 BULL" if x == "BULL" else "🔴 BEAR")
        df_display['Entry'] = df_display['entry_price'].apply(lambda x: f"{x:.5f}")
        df_display['Max Pips'] = df_display['max_pips'].apply(lambda x: f"+{x:.1f}" if pd.notnull(x) and x > 0 else "0.0")
        
        # Korostetaan Stop Loss osumat ja avoimet emojilla
        def muotoile_status(status):
            if status == "SL HIT": return "🛑 SL HIT"
            return "⏳ OPEN"
            
        df_display['Tila'] = df_display['status'].apply(muotoile_status)
        
        df_display = df_display[['Aika', 'Pari', 'Suunta', 'Entry', 'Max Pips', 'Tila']]
        
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.html(f"""<div class="metric-card">
                    <div class="metric-label">Iskut (Viim. 60 min)</div>
                    <div class="metric-value" style="color: {'#38bdf8' if lkm_60min > 0 else '#94a3b8'};">{lkm_60min}</div>
                    </div>""")
            with col2:
                # Näytetään nopea PnL arvio taulun yläreunassa
                voittoprosentti = 0
                if lkm_60min > 0:
                    voitolliset = len(viim_60_min[viim_60_min['max_pips'] > 3.0])
                    voittoprosentti = int((voitolliset / lkm_60min) * 100)
                st.html(f"""<div class="metric-card">
                    <div class="metric-label">> 3 pips Osumatarkkuus (1h)</div>
                    <div class="metric-value" style="color: {'#22c55e' if voittoprosentti > 50 else '#f8fafc'};">{voittoprosentti}%</div>
                    </div>""")
            with col3:
                viimeisin = df_display.iloc[0]['Pari'] if not df_display.empty else "Ei dataa"
                viimeisin_suunta = df.iloc[0]['direction'] if not df.empty else ""
                color = "#22c55e" if viimeisin_suunta == "BULL" else "#ef4444"
                st.html(f"""<div class="metric-card">
                    <div class="metric-label">Viimeisin Setup</div>
                    <div class="metric-value" style="color: {color}; font-size: 32px; padding-top: 10px;">{viimeisin}</div>
                    </div>""")
                
        with table_placeholder.container():
            st.markdown("### 📋 Signaalien Paper Trading -seuranta")
            st.dataframe(df_display.head(20), use_container_width=True, hide_index=True)
            
    else:
        with metrics_placeholder.container():
            st.info("Odotetaan ensimmäistä Vaihe 2 -signaalia tietokantaan...")
            
    st.write(f"*Päivitetty viimeksi (MT5 Livetieto): {datetime.now().strftime('%H:%M:%S')}*")
    # Päivitetään hintoja 5 sekunnin välein
    time.sleep(5)