import os
from dotenv import load_dotenv

# Ladataan .env tiedosto projektin juuresta
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# MT5 asetukset
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")

# Postgres asetukset - Rakennetaan yhteysmerkkijono
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")

# Tämä on se muuttuja, jota test_db.py etsii
DB_CONN = f"dbname={DB_NAME} user={DB_USER} password={DB_PASS} host={DB_HOST}"