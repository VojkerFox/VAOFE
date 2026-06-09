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

st.set_page_config(layout="wide", page_title="VOJKER | Global Liquidity Watch")

st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    </style>
""", unsafe_allow_html=True)

def hae_tuoreimmat():
    try:
        query = "SELECT * FROM ai_learning_logs ORDER BY id DESC"
        df = pd.read_sql(query, engine)
        if df.empty: return pd.DataFrame()
        return df.drop_duplicates(subset=['pair'], keep='first')
    except:
        return pd.DataFrame()

st.title("🌐 VOJKER | Full Confluence Trade Cockpit")
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
        
    with grid_placeholder.container():
        cols = st.columns(4)
        for i, row in df_display.iterrows():
            friction = row.get('friction_weight')
            predicted = row.get('predicted_pips')
            volume = row.get('volume')
            macd = row.get('macd_hist')
            w_sig = row.get('wick_signal')
            w_pct = row.get('wick_pct')
            is_valid = pd.notnull(friction)
            
            # 1. Kitka-status
            val = float(friction) if is_valid else 0.0
            status = "EXHAUSTED" if is_valid and val > 0.9 else ("EXIT SOON" if is_valid and val > 0.6 else ("HOLD" if is_valid else "NODATA"))
            color = "#ef4444" if status == "EXHAUSTED" else ("#f97316" if status == "EXIT SOON" else ("#22c55e" if status == "HOLD" else "#64748b"))
            
            # 2. Tilausvirran pips-paine
            pred_val = float(predicted) if pd.notnull(predicted) else 0.0
            vol_val = float(volume) if pd.notnull(volume) else 0.0
            macd_val = float(macd) if pd.notnull(macd) else 0.0
            wick_sig_val = float(w_sig) if pd.notnull(w_sig) else 0.0
            wick_pct_val = float(w_pct) if pd.notnull(w_pct) else 0.0
            
            # Suuntateksti tilausvirrasta
            if not is_valid:
                suunta_teksti = "ODOTTAA DATAA"
                suunta_color = "#64748b"
                kasauma_status = ""
            elif pred_val > 0.05:
                suunta_teksti = f"🟢 OSTO (LONG) +{pred_val:.1f}p"
                suunta_color = "#38bdf8"
                kasauma_status = "⚠️ OSTO-KASAUMA ESTÄÄ LASKUN" if val > 0.75 else ""
            elif pred_val < -0.05:
                suunta_teksti = f"🔴 MYYNTI (SHORT) {pred_val:.1f}p"
                suunta_color = "#f43f5e"
                kasauma_status = "⚠️ MYYNTI-KASAUMA ESTÄÄ NOUSUN" if val > 0.75 else ""
            else:
                suunta_teksti = "⚪ NEUTRAALI VIRTA"
                suunta_color = "#94a3b8"
                kasauma_status = ""
            
            # 3. MACD Trendi-indikaattorin visualisointi (Värinvaihto)
            if not is_valid:
                macd_html = '<span style="color:#64748b;">MACD: ---</span>'
            elif macd_val > 0:
                macd_html = f'<span style="color:#22c55e; font-weight:bold;">📈 MACD OSTAJILLA ({macd_val:.5f})</span>'
            else:
                macd_html = f'<span style="color:#ef4444; font-weight:bold;">📉 MACD MYYJILLÄ ({macd_val:.5f})</span>'
                
            # 4. Hylkäyshäntäskanneri (Sweep)
            if not is_valid or wick_sig_val == 0.0:
                wick_html = '<div style="font-size:11px; color:#64748b; margin-top:4px;">Häntähylkäys: Ei signaalia</div>'
            elif wick_sig_val == 1.0:
                wick_html = f'<div style="font-size:11px; color:#38bdf8; font-weight:bold; background:#0c4a6e; padding:3px; border-radius:4px; margin-top:4px;">🎯 LIKVIDITEETTI PYYHKÄISTY (BULLISH {wick_pct_val:.1f}%)</div>'
            elif wick_sig_val == -1.0:
                wick_html = f'<div style="font-size:11px; color:#f43f5e; font-weight:bold; background:#881337; padding:3px; border-radius:4px; margin-top:4px;">🎯 LIKVIDITEETTI PYYHKÄISTY (BEARISH {wick_pct_val:.1f}%)</div>'
            
            friction_display = f"{val:.4f}" if is_valid else "---"
            vol_display = f"{int(vol_val)}" if is_valid else "---"
            
            html_kasauma = f'<div style="font-size:11px; font-weight:bold; color:#f8fafc; background:#b91c1c; padding:4px; border-radius:4px; margin-bottom:8px;">{kasauma_status}</div>' if kasauma_status else ''
            
            with cols[i % 4]:
                st.markdown(f"""
<div style="background:#1e293b; padding:15px; border-radius:10px; border:1px solid #334155; margin-bottom:10px; text-align:center;">
<h3 style="margin:0; color:#f8fafc;">{row['pair']}</h3>
<p style="font-size:12px; color:#94a3b8; margin:2px 0;">Kontraktit (30m vol): <b>{vol_display}</b></p>
<p style="font-size:12px; color:#94a3b8; margin:2px 0;">Kitkakerroin: <b>{friction_display}</b></p>

<div style="margin:5px 0; font-size:12px;">
{macd_html}
</div>

<div style="background:#0f172a; padding:6px; border-radius:6px; margin:8px 0; border:1px solid #1e293b;">
<span style="font-size:14px; font-weight:bold; color:{suunta_color};">{suunta_teksti}</span>
</div>

{wick_html}

{html_kasauma}
<p style="font-size:18px; font-weight:bold; color:{color}; margin-top:8px; margin-bottom:0;">{status}</p>
</div>
                """, unsafe_allow_html=True)
                
        st.write(f"--- Päivitetty: {time.strftime('%H:%M:%S')} | Seurannassa: 11 instrumenttia ---")
    
    time.sleep(2)