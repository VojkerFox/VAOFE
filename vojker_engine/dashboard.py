import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go
import plotly.express as px

# Streamlit sivun asetukset
st.set_page_config(layout="wide", page_title="VOJKER | VAOFE System Monitor")

# CSS-tyylittely (Tailwind-inspiroitu Dark Mode)
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .metric-card { 
        background: #1e293b; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #334155; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: #0284c7; 
        padding: 15px; 
        border-radius: 8px; 
        color: white;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Tietokantayhteys
DB_CONFIG = "dbname=vofe_db user=postgres password=password host=localhost port=5432"

@st.cache_data(ttl=5) # Automaattinen päivitys 5 sekunnin välein
def hae_tuorein_data():
    try:
        conn = psycopg2.connect(DB_CONFIG)
        query = "SELECT * FROM ai_learning_logs ORDER BY id DESC LIMIT 100"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Tietokantavirhe: {e}")
        return pd.DataFrame()

st.title("⚡ VOJKER | SciML Live Engine")

df = hae_tuorein_data()

if not df.empty:
    # --- VISKOSITEETTIMITTARI ---
    col1, col2, col3 = st.columns(3)
    latest_friction = df['friction_weight'].iloc[0]
    avg_loss = df['loss'].mean()
    
    col1.markdown(f'<div class="metric-card"><h3>Markkinan Viskositeetti (Kitka)</h3><p style="font-size:30px;">{latest_friction:.4f}</p></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><h3>Mallin Luottamus (Loss)</h3><p style="font-size:30px;">{avg_loss:.4f}</p></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><h3>Analysoituja Tapahtumia</h3><p style="font-size:30px;">{len(df)}</p></div>', unsafe_allow_html=True)

    # --- SCI-ML VISUALISOINTI ---
    st.subheader("SciML-ennuste: Ennustettu vs. Toteutunut liike (Pips)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['predicted_pips'], name='Ennuste', line=dict(color='#38bdf8')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['actual_pips'], name='Toteuma', line=dict(color='#fbbf24')))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # --- FSM TILAKONEEN TILA ---
    st.subheader("FSM Tilakoneen Tila")
    st.info(f"Status: ACTIVE | Seurataan {df['pair'].iloc[0]} paria")

    st.markdown(f"""
    <div class="info-box">
        <strong>Analyysi:</strong> SciML-mallin kitkakerroin on {latest_friction:.4f}. 
        Mallin gradientti-indikaattorit vahvistavat determinististä suuntaa.
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("Tietokannasta ei löydy vielä dataa. Odota, että botti tallentaa ensimmäiset rivit.")