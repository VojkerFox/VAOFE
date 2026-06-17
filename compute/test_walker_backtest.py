import os
import MetaTrader5 as mt5
import jax.numpy as jnp
from dotenv import load_dotenv
from walker_engine import detect_lightning_bolt  # Tuodaan JAX-ydin suoraan moottoristasi

load_dotenv()

def run_historical_analysis(symbol, lookback_m15_candles=500):
    if not mt5.initialize():
        print("MT5 alustus epäonnistui.")
        return

    # Haetaan suuri määrä historiallista M15-dataa testipenkkiin
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, lookback_m15_candles)
    if rates is None or len(rates) < 50:
        print("Dataa ei saatu.")
        return

    print(f"\n⚡ KÄYNNISTETÄÄN HISTORIALLINEN JAX-ANALYYSI: {symbol} ({len(rates)} kynttilää) ⚡")
    print("-" * 60)

    # Simuloidaan liukuvaa markkinatilannetta historiassa
    # Otetaan kiinteä testiskenario (esim. jokin kuvitteellinen H1-taso)
    mock_h1_resistance = float(rates[10]['high']) 
    print(f"Testitasoksi lukittu historiallisen kynttilän huippu: {mock_h1_resistance:.5f}")

    signals_found = 0
    for i in range(20, len(rates)):
        # Luodaan 10 kynttilän liukuva ikkuna, aivan kuten live-moottorissa
        window = rates[i-10:i]
        
        highs = jnp.array([c['high'] for c in window])
        lows = jnp.array([c['low'] for c in window])
        closes = jnp.array([c['close'] for c in window])
        
        # Ajetaan JAX-tunnistus
        signal = detect_lightning_bolt(highs, lows, closes, mock_h1_resistance, is_uptrend_break=True)
        
        if signal == 1.0:
            signals_found += 1
            print(f"🎯 Löydetty 'Lightning Bolt' indeksissä {i} | Aika: {window[-1]['time']} | Hinta: {window[-1]['close']:.5f}")

    print("-" * 60)
    print(f"✅ Analyysi valmis. Löydetty yhteensä {signals_found} vahvistettua murtorakennetta.")
    mt5.shutdown()

if __name__ == "__main__":
    # Ajetaan testi esimerkiksi punnan tai euron historialla
    run_historical_analysis("EURUSD", lookback_m15_candles=1000)