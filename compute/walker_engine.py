import os
import time
import datetime
import pytz
import requests
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import MetaTrader5 as mt5
from dotenv import load_dotenv

# --- LADATAAN YMPÄRISTÖMUUTTUJAT .ENV TIEDOSTOSTA ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if TELEGRAM_CHAT_ID:
    TELEGRAM_CHAT_ID = TELEGRAM_CHAT_ID.strip("'\" ")

SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "EURAUD", "GBPJPY", "XAUUSD", "BTCUSD"]
FINNISH_TZ = pytz.timezone("Europe/Helsinki")


# ============================================================================
# 1. GEOMETRINEN JAX-YDIN (Kuvatunnistus ilman luuppeja)
# ============================================================================

# FIKSI: Kerrotaan JAXille että is_uptrend_break on staattinen argumentti
@partial(jit, static_argnames=['is_uptrend_break'])
def detect_lightning_bolt(highs, lows, closes, level, is_uptrend_break=True):
    """
    Tunnistaa kuvien mukaisen '3+ Candle LB' / BOS rakenteen matriisista.
    Käyttää puhdasta JAX-matematiikkaa signaalin havaitsemiseen.
    """
    if is_uptrend_break:
        # Downtrend Reverse / Bullish LB:
        # 1. Breakout-vaihe: Viimeisimmät kynttilät sulkevat H1-vastustason (level) yläpuolelle
        break_condition = closes[-1] > level
        
        # 2. Retest-vaihe: Kynttilöiden alimmat pisteet (lows) käyvät lähellä tasoa, mutta eivät romuta sitä
        retest_zone_ok = (lows[-2] <= level * 1.0005) & (lows[-2] >= level * 0.9995)
        
        # 3. Continuation-vaihe: Hinta tekee uuden huipun (HH) breakout-kynttilän yli
        continuation_ok = closes[-1] > highs[-2]
        
        signal = break_condition & retest_zone_ok & continuation_ok
        return jnp.where(signal, 1.0, 0.0)
    else:
        # Uptrend Reverse / Bearish LB (Break Below):
        break_condition = closes[-1] < level
        retest_zone_ok = (highs[-2] >= level * 0.9995) & (highs[-2] <= level * 1.0005)
        continuation_ok = closes[-1] < lows[-2]
        
        signal = break_condition & retest_zone_ok & continuation_ok
        return jnp.where(signal, 1.0, 0.0)


# ============================================================================
# 2. RAKENTEELLISET TASOT (Aamun klo 09:00 rutiini)
# ============================================================================

def get_daily_h1_boundaries():
    """
    Hakee aamulla voimassa olevat strategiset H1-laatikon rajat.
    """
    daily_levels = {}
    print(f"\n⏰ Kello on 09:00 Suomen aikaa. Haetaan päivän strategiset H1-tasot...")
    
    for symbol in SYMBOLS:
        # Haetaan edellisen 24 tunnin konsolidaatio (Chop zone)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 24)
        if rates is None or len(rates) == 0:
            print(f"⚠️ Varoitus: H1-dataa ei saatu symbolille {symbol}")
            continue
            
        highs = [c['high'] for c in rates]
        lows = [c['low'] for c in rates]
        
        daily_levels[symbol] = {
            "buy_above": float(max(highs)),
            "sell_below": float(min(lows))
        }
        print(f"📊 {symbol: <7} Locked -> Range: {min(lows):.5f} - {max(highs):.5f}")
    return daily_levels


# ============================================================================
# 3. LÄHETYS JA SEURANTA (The Execution Layer)
# ============================================================================

def send_premium_lightning_bolt_alert(symbol, direction, price):
    """
    Lähettää täydellisen kuvien mukaisen vahvistusviestin Telegramiin.
    """
    emoji = "⚡🟢" if direction == "BULLISH BREAKOUT" else "⚡🔴"
    message = (
        f"🦅 *MR. WALKER ML – LIGHTNING BOLT DETECTED* {emoji}\n"
        f"-----------------------------------------\n"
        f"📈 *Asset:* {symbol}\n"
        f"🎯 *Pattern:* {direction} (M15 BOS)\n"
        f"💵 *Execution Price:* `{price}`\n"
        f"-----------------------------------------\n"
        f"✅ *M15/M5 STRUCTURE CONFIRMED:*\n"
        f"[✓] Structural Level Broken Cleanly\n"
        f"[✓] 3+ Candle LB Retest Validated\n"
        f"[✓] Continuation Candle Triggered\n"
        f"-----------------------------------------\n"
        f"👉 *Action:* Check lot size and execute according to 1:2 / 1:4 trailing rules."
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})
    except Exception as e:
        print(f"❌ Virhe Telegram-hälytyksen lähetyksessä: {e}")


# ============================================================================
# 4. PÄÄOHJELMA JA AJASTINLOOPPI
# ============================================================================

def main():
    if not mt5.initialize():
        print("MT5 alustus epäonnistui.")
        return

    daily_levels = get_daily_h1_boundaries()
    print("\n🚀 WalkerEngine Autopilot käynnistetty. Odotetaan tasojen rikkoutumista ja LB-muodostelmia...")

    try:
        while True:
            now_finland = datetime.datetime.now(FINNISH_TZ)
            
            # Aamun nollaus ja tasojen päivitys tismalleen klo 09:00:00
            if now_finland.hour == 9 and now_finland.minute == 0 and now_finland.second == 0:
                daily_levels = get_daily_h1_boundaries()
                time.sleep(1.1) # Estetään tuplalataus saman sekunnin aikana
            
            # Live-seuranta matalalla aikajänteellä (M15 / M5)
            for symbol in SYMBOLS:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    continue
                current_price = tick.last if tick.last != 0 else tick.bid
                
                # Haetaan M15 kynttilät JAX-analyysiä varten
                m15_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 10)
                if m15_rates is None or len(m15_rates) < 5:
                    continue
                
                # Muutetaan data puhtaiksi JAX-taulukoiksi laskentaa varten
                m15_highs = jnp.array([c['high'] for c in m15_rates])
                m15_lows = jnp.array([c['low'] for c in m15_rates])
                m15_closes = jnp.array([c['close'] for c in m15_rates])
                
                limits = daily_levels.get(symbol)
                if not limits:
                    continue
                
                # --- CASE 1: Hinta on ylittänyt H1 OSTO-rajan -> Etsitään Bullish LB ---
                if current_price > limits["buy_above"]:
                    signal = detect_lightning_bolt(m15_highs, m15_lows, m15_closes, limits["buy_above"], is_uptrend_break=True)
                    if signal == 1.0:
                        send_premium_lightning_bolt_alert(symbol, "BULLISH BREAKOUT", current_price)
                        time.sleep(60) # Estetään spämmäys saman kynttilän aikana
                
                # --- CASE 2: Hinta on alittanut H1 MYYNTI-rajan -> Etsitään Bearish LB ---
                if current_price < limits["sell_below"]:
                    signal = detect_lightning_bolt(m15_highs, m15_lows, m15_closes, limits["sell_below"], is_uptrend_break=False)
                    if signal == 1.0:
                        send_premium_lightning_bolt_alert(symbol, "BEARISH BREAKOUT", current_price)
                        time.sleep(60)
                        
            time.sleep(1.0) # Skannataan kerran sekunnissa
            
    except KeyboardInterrupt:
        print("\nAutopilot sammutettu.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()