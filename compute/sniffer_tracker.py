import os
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Ladataan .env tiedosto
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "vofe_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": "5432"
}

def init_db():
    """Luo seurantataulun, jos sitä ei vielä ole olemassa."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sniffer_history (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR(10),
                timeframe VARCHAR(10),
                direction VARCHAR(10),
                entry_price NUMERIC,
                sl_price NUMERIC,
                status VARCHAR(20) DEFAULT 'OPEN'
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ Tietokannan alustusvirhe (Tracker): {e}")

# Kutsutaan alustusta heti kun moduuli ladataan
init_db()

def log_trade_signal(symbol: str, timeframe: str, direction: str, entry: float, sl: float):
    """
    Tämä funktio ottaa signaalin vastaan ja tallentaa sen kantaan.
    Tämä on suojattu try-except -lohkolla: jos DB kaatuu, Sniffer jatkaa toimintaansa!
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO sniffer_history (symbol, timeframe, direction, entry_price, sl_price)
            VALUES (%s, %s, %s, %s, %s)
        """, (symbol, timeframe, direction, entry, sl))
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"💾 [Tracker] Signaali tallennettu tietokantaan: {symbol} {direction}")
    except Exception as e:
        print(f"⚠️ [Tracker] Virhe signaalin tallennuksessa: {e}")