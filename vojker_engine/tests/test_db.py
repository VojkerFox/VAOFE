import psycopg2
import sys
import os

# Lisätään projektin juuri sys.path:iin, jotta config löytyy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DB_CONN

try:
    conn = psycopg2.connect(DB_CONN)
    print("Tietokantayhteys OK!")
    conn.close()
except Exception as e:
    print(f"Yhteysvirhe: {e}")