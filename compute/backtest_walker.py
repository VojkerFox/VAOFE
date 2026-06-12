import os
import sys
import MetaTrader5 as mt5
import jax.numpy as jnp
from datetime import datetime, timedelta

# Lisätään polku juureen, jotta importit walker_enginesta toimivat
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from walker_engine import detect_lightning_bolt, SYMBOLS

def run_multi_asset_backtest(days_to_test=7):
    """
    Simuloi kaikkien 14 instrumentin live-toimintaa menneisyydessä.
    Laskee jokaiselle päivälle aidot H1-rajat ja skannaa M15 salamat JAXilla.
    """
    if not mt5.initialize():
        print("❌ MT5 alustus epäonnistui. Varmista, että ohjelma on auki taustalla.")
        return

    print(f"\n🦅 VOJKER AUTOMATED BACKTESTER – MULTI-ASSET TRACKING 🦅")
    print(f"🔄 Analysoidaan {days_to_test} edellistä markkinapäivää dynaamisilla H1-tasoilla...")
    print("=" * 105)
    print(f"{'AJANKOHTA':<17} | {'PARI':<10} | {'SUUNTA':<16} | {'HINTA':<10} | {'RIKKOTTU TASO':<14} | {'STATUS'}")
    print("=" * 105)

    total_signals = 0
    all_events = []
    now = datetime.now()
    
    # Käydään päivät läpi järjestyksessä menneisyydestä tähän päivään
    for d in range(days_to_test, -1, -1):
        # Määritellään testattavan päivän aloitushetki (klo 09:00 vastaava hetki menneisyydessä)
        test_day_start = datetime(now.year, now.month, now.day) - timedelta(days=d)
        test_day_end = test_day_start + timedelta(days=1)
        
        for symbol in SYMBOLS:
            # 1. Haetaan H1-tasot testipäivää edeltäneeltä 24 tunnilta (Aamun klo 09:00 rutiini)
            h1_rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, test_day_start, 24)
            if h1_rates is None or len(h1_rates) == 0:
                continue
            
            h1_highs = [c['high'] for c in h1_rates]
            h1_lows = [c['low'] for c in h1_rates]
            buy_above_level = float(max(h1_highs))
            sell_below_level = float(min(h1_lows))

            # 2. Haetaan saman vuorokauden M15 kynttilät live-seurantaa varten
            m15_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, test_day_start, test_day_end)
            if m15_rates is None or len(m15_rates) < 10:
                continue

            # 3. Liukuva ikkuna kynttilöiden yli (tismalleen kuten walker_engine.py:ssä)
            for i in range(10, len(m15_rates)):
                window = m15_rates[i-10:i]
                
                m15_highs = jnp.array([c['high'] for c in window])
                m15_lows = jnp.array([c['low'] for c in window])
                m15_closes = jnp.array([c['close'] for c in window])
                
                current_price = float(m15_closes[-1])
                event_time = datetime.fromtimestamp(int(window[-1]['time']))
                time_str = event_time.strftime('%Y-%m-%d %H:%M')

                # --- CASE A: OSTO-Murtuma (Bullish) ---
                if current_price > buy_above_level:
                    signal = detect_lightning_bolt(m15_highs, m15_lows, m15_closes, buy_above_level, is_uptrend_break=True)
                    if signal == 1.0:
                        all_events.append({
                            'time_raw': event_time,
                            'output': f"{time_str:<17} | {symbol:<10} | {'BULLISH BREAK':<16} | {current_price:<10.5f} | {buy_above_level:<14.5f} | ⚡ VALID LB"
                        })
                        total_signals += 1

                # --- CASE B: MYYNTI-Murtuma (Bearish) ---
                elif current_price < sell_below_level:
                    signal = detect_lightning_bolt(m15_highs, m15_lows, m15_closes, sell_below_level, is_uptrend_break=False)
                    if signal == 1.0:
                        all_events.append({
                            'time_raw': event_time,
                            'output': f"{time_str:<17} | {symbol:<10} | {'BEARISH BREAK':<16} | {current_price:<10.5f} | {sell_below_level:<14.5f} | ⚡ VALID LB"
                        })
                        total_signals += 1

    # Järjestetään kaikki havaitut tapahtumat globaaliin aikajärjestykseen yli kaikkien parien
    all_events.sort(key=lambda x: x['time_raw'])
    for event in all_events:
        print(event['output'])

    print("=" * 105)
    print(f"✨ MULTI-ASSET BACKTEST VALMIS. Löydetty yhteensä {total_signals} mekaanisesti vahvistettua rakennetta.")
    print("=" * 105 + "\n")
    
    mt5.shutdown()

if __name__ == "__main__":
    # Testataan oletuksena viimeistä 7 markkinapäivää dynaamisesti
    run_multi_asset_backtest(days_to_test=7)