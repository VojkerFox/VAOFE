import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
import time
from dotenv import load_dotenv

load_dotenv()

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP", "GOLD", "SILVER", "US30"]
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@localhost:5432/{os.getenv('DB_NAME', 'vofe_db')}"
engine = create_engine(DB_URL)

st.set_page_config(layout="wide", page_title="VOJKER | Global Liquidity State Machine")

# Puhdas globaali tyylitys suoraan natiivilla HTML-injektiolla
st.html("""
    <style>
    .stApp { background-color: #0d1321; color: #f8fafc; font-size: 16px; }
    h1 { font-size: 36px !important; font-weight: 800 !important; color: #f8fafc !important; }
    h2 { font-size: 24px !important; font-weight: 700 !important; margin-bottom: 15px !important; }
    </style>
""")

def hae_tuoreimmat():
    try:
        query = "SELECT * FROM ai_learning_logs ORDER BY id DESC"
        df = pd.read_sql(query, engine)
        if df.empty: return pd.DataFrame()
        # Varmistetaan uniikit rivit parikohtaisesti (Cpk 3.0 -suodatus)
        return df.drop_duplicates(subset=['pair'], keep='first')
    except:
        return pd.DataFrame()

st.title("🌐 VOJKER | Full Confluence State Machine")
grid_placeholder = st.empty()

while True:
    df_db = hae_tuoreimmat()
    
    df_display = pd.DataFrame({'pair': PAIRS})
    if not df_db.empty:
        df_display = df_display.merge(df_db, on='pair', how='left')
    else:
        df_display['friction_weight'] = None
        df_display['predicted_pips'] = None
        df_display['volume'] = None
        df_display['macd_hist'] = None
        df_display['wick_signal'] = None
        df_display['wick_pct'] = None
        df_display['dom_imbalance'] = None
        
    with grid_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.html("<h2 style='text-align:center; color:#38bdf8; background:#0c4a6e; padding:8px; border-radius:6px; margin:0;'>1. Vapaa Impulssi 🏎️</h2>")
        with col2:
            st.html("<h2 style='text-align:center; color:#f97316; background:#7c2d12; padding:8px; border-radius:6px; margin:0;'>2. Jarrut Päälle 🛑</h2>")
        with col3:
            st.html("<h2 style='text-align:center; color:#f43f5e; background:#881337; padding:8px; border-radius:6px; margin:0;'>3. Kasauma/Seinä 🧱</h2>")
        with col4:
            st.html("<h2 style='text-align:center; color:#22c55e; background:#14532d; padding:8px; border-radius:6px; border: 2px solid #22c55e; margin:0;'>4. KALA ISKEE 🔥</h2>")

        for i, row in df_display.iterrows():
            friction = row.get('friction_weight')
            predicted = row.get('predicted_pips')
            volume = row.get('volume')
            macd = row.get('macd_hist')
            w_sig = row.get('wick_signal')
            w_pct = row.get('wick_pct')
            l2_dom = row.get('dom_imbalance')
            is_valid = pd.notnull(friction)
            
            val = float(friction) if is_valid else 0.0
            pred_val = float(predicted) if pd.notnull(predicted) else 0.0
            vol_val = float(volume) if pd.notnull(volume) else 0.0
            macd_val = float(macd) if pd.notnull(macd) else 0.0
            wick_sig_val = float(w_sig) if pd.notnull(w_sig) else 0.0
            wick_pct_val = float(w_pct) if pd.notnull(w_pct) else 0.0
            l2_dom_val = float(l2_dom) if pd.notnull(l2_dom) else 0.0
            
            # Lasketaan JAX-Inertia
            inertia_val = (vol_val / (abs(macd_val) + 1e-7)) * val if is_valid else 0.0
            
            # Tilakone-luokittelu (Phase 1 - 4)
            if not is_valid:
                vaihe = 1
                status = "NODATA"
                color = "#64748b"
            elif wick_sig_val != 0.0:
                vaihe = 4  
                status = "🎯 A+ SETUP"
                color = "#22c55e"
            elif val >= 1.5:
                vaihe = 3  
                status = "EXHAUSTED"
                color = "#ef4444"
            elif val >= 0.5:
                vaihe = 2  
                status = "EXIT SOON"
                color = "#f97316"
            else:
                vaihe = 1  
                status = "HOLD / FLOW"
                color = "#38bdf8"
            
            # Suuntatekstit isona
            if not is_valid:
                suunta_teksti = "ODOTTAA DATAA"
                suunta_color = "#64748b"
                kasauma_status = ""
            elif pred_val > 0.05:
                suunta_teksti = f"🟢 OSTO (LONG) +{pred_val:.1f}p"
                suunta_color = "#38bdf8"
                kasauma_status = "⚠️ OSTO-KASAUMA ESTÄÄ LASKUN" if val > 1.1 else ""
            elif pred_val < -0.05:
                suunta_teksti = f"🔴 MYYNTI (SHORT) {pred_val:.1f}p"
                suunta_color = "#f43f5e"
                kasauma_status = "⚠️ MYYNTI-KASAUMA ESTÄÄ NOUSUN" if val > 1.1 else ""
            else:
                suunta_teksti = "⚪ NEUTRAALI VIRTA"
                suunta_color = "#94a3b8"
                kasauma_status = ""
            
            if not is_valid:
                macd_html = '<span style="color:#64748b; font-size:14px;">MACD: ---</span>'
            elif macd_val > 0:
                macd_html = f'<span style="color:#22c55e; font-weight:bold; font-size:14px;">📈 MACD OSTAJILLA ({macd_val:.5f})</span>'
            else:
                macd_html = f'<span style="color:#ef4444; font-weight:bold; font-size:14px;">📉 MACD MYYJILLÄ ({macd_val:.5f})</span>'
                
            if not is_valid or wick_sig_val == 0.0:
                wick_html = '<div style="font-size:13px; color:#64748b; margin-top:6px;">Häntähylkäys: Ei signaalia</div>'
            elif wick_sig_val == 1.0:
                wick_html = f'<div style="font-size:13px; color:#38bdf8; font-weight:bold; background:#0c4a6e; padding:5px; border-radius:4px; margin-top:6px; border: 1px solid #38bdf8;">🎯 PYYHKÄISY (BULLISH {wick_pct_val:.1f}%)</div>'
            elif wick_sig_val == -1.0:
                wick_html = f'<div style="font-size:13px; color:#f43f5e; font-weight:bold; background:#881337; padding:5px; border-radius:4px; margin-top:6px; border: 1px solid #f43f5e;">🎯 PYYHKÄISY (BEARISH {wick_pct_val:.1f}%)</div>'
            
            friction_display = f"{val:.4f}" if is_valid else "---"
            vol_display = f"{int(vol_val):,}" if is_valid else "---"
            inertia_display = f"{inertia_val:,.1f}" if is_valid else "---"
            
            # Muotoillaan Level 2 DOM Imbalance dynaamisesti väreillä (Ostovoima sininen, myyntivoima punainen)
            if pd.isnull(l2_dom):
                l2_display = "---"
                l2_color = "#94a3b8"
            else:
                l2_display = f"{l2_dom_val:+.4f}"
                l2_color = "#38bdf8" if l2_dom_val >= 0 else "#f43f5e"
            
            html_kasauma = f'<div style="font-size:12px; font-weight:bold; color:#f8fafc; background:#b91c1c; padding:5px; border-radius:4px; margin-bottom:8px; text-align:center;">{kasauma_status}</div>' if kasauma_status else ''
            
            # Lopullinen Cpk 3.0 -korttipohja ilman sisäisiä sisennyksiä
            card_html = f"""<div style="background:#1e293b; padding:18px; border-radius:12px; border:1px solid #334155; margin-bottom:15px; text-align:center; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
<h3 style="margin:0 0 8px 0; color:#f8fafc; font-size:26px; font-weight:800; letter-spacing: 0.5px;">{row['pair']}</h3>
<div style="text-align:left; background:#111827; padding:10px; border-radius:8px; margin-bottom:10px; border:1px solid #1f2937;">
<p style="font-size:14px; color:#94a3b8; margin:3px 0;">Kontraktit: <b style="color:#f8fafc; font-size:15px;">{vol_display}</b></p>
<p style="font-size:14px; color:#94a3b8; margin:3px 0;">Kitkakerroin: <b style="color:#f8fafc; font-size:15px;">{friction_display}</b></p>
<p style="font-size:14px; color:#94a3b8; margin:3px 0;">📊 L2 DOM Paine: <b style="color:{l2_color}; font-size:15px;">{l2_display}</b></p>
<p style="font-size:14px; color:#94a3b8; margin:3px 0;">⛓️ Inertia-indeksi: <b style="color:#60a5fa; font-size:15px;">{inertia_display}</b></p>
</div>
<div style="margin:8px 0;">
{macd_html}
</div>
<div style="background:#0f172a; padding:8px; border-radius:6px; margin:10px 0; border:1px solid #334155;">
<span style="font-size:16px; font-weight:bold; color:{suunta_color};">{suunta_teksti}</span>
</div>
{wick_html}
{html_kasauma}
<p style="font-size:22px; font-weight:bold; color:{color}; margin-top:12px; margin-bottom:0; letter-spacing: 1px;">{status}</p>
</div>"""
            
            # Ohjataan kortit st.html-komennolla oikeaan sarakkeeseen ilman välikäsiä
            if vaihe == 1:
                with col1: st.html(card_html)
            elif vaihe == 2:
                with col2: st.html(card_html)
            elif vaihe == 3:
                with col3: st.html(card_html)
            elif vaihe == 4:
                with col4: st.html(card_html)
                
        st.write(f"--- Päivitetty: {time.strftime('%H:%M:%S')} | JAX-Kineettinen L2-tilakone toiminnassa (Cpk 3.0) ---")
    
    time.sleep(2)