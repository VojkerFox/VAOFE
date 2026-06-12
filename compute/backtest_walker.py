import os
import sys
import MetaTrader5 as mt5
import jax.numpy as jnp
from datetime import datetime

# Lisätään suora polku, jotta voimme importata JAX-ytimen suoraan livenä pyörivästä moottoristasi
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from walker_engine import detect_lightning_bolt

def run_walker_backtest(symbol="EURUSD", lookback_candles=1000):
    """
    Kelaa historiallisen datan läpi ja testaa JAX-salamatunnistimen toiminnan.
    """
    if not mt5.initialize():
        print("❌ MT5 alustus epäonnistui. Varmista että terminaali on auki taustalla.")
        return

    print(f"\n⚡ KÄYNNISTETÄÄN HISTORIALLINEN JAX-BACKTEST: {symbol} ⚡")
    print(f"📂 Ladataan {lookback_candles} kynttilää M15-aikajänteeltä historiasta...")
    print("-" * 70)

    # Haetaan M15-kynttilähistoria testipenkkiin
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, lookback_candles)
    if rates is None or len(rates) < 50:
        print("❌ Datan saanti epäonnistui. Varmista, että symboli on Market Watchissa.")
        mt5.shutdown()
        return

    # Simuloidaan H1-tasot: Lukitaan testitasoksi jokin historiallisen kynttilän merkittävä huippu/pohja
    # (Käytetään esimerkkinä kynttilän 10 huippua, jotta saamme kiinteän tason breakout-testille)
    mock_h1_resistance = float(rates[10]['high'])
    print(f"🎯 Testitasoksi simuloitu historiallisen rakenteen huippu: {mock_h1_resistance:.5f}")
    print("🚀 Skannataan kynttilämatriisia...")
    print("-" * 70)

    signals_found = 0

    # Liukuva ikkuna kynttilähistorian yli (aivan kuten livenä pyörivässä moottorissa)
    for i in range(10, len(rates)):
        # Napataan 10 kynttilän siivu JAX-analyysiin
        window = rates[i-10:i]
        
        m15_highs = jnp.array([c['high'] for c in window])
        m15_lows = jnp.array([c['low'] for c in window])
        m15_closes = jnp.array([c['close'] for c in window])
        
        # Testataan nykyisen ikkunan viimeisintä hintaa (closes[-1]) suhteessa tasoon
        current_price = float(m15_closes[-1])
        
        if current_price > mock_h1_resistance:
            # Ajetaan aito JAX-tunnistin suoraan livenä käyttämästäsi walker_engine.py tiedostosta!
            signal = detect_lightning_bolt(m15_highs, m15_lows, m15_closes, mock_h1_resistance, is_uptrend_break=True)
            
            if signal == 1.0:
                signals_found += 1
                candle_time = datetime.fromtimestamp(int(window[-1]['time'])).strftime('%Y-%m-%d %H:%M')
                print(f"🔥 [OSUMA {signals_found}] -> Lightning Bolt havaittu!")
                print(f"   📅 Aika: {candle_time} | 💵 Sulkuhinta: {current_price:.5f} (BOS tasosta {mock_h1_resistance:.5f})")
                print(f"   📝 Geometria: Breakout + Retest liukuvassa 3+ ikkunassa varmistettu.")
                print("-" * 50)

    print("-" * 70)
    print(f"✨ BACKTEST SUORITETTU ONNISTUNEESTI.")
    print(f"📊 Läpikäydyt kynttilät: {len(rates)} kpl")
    print(f"🎯 Löydetyt validit '3+ Candle LB' -muodostelmat: {signals_found} kpl")
    print("-" * 70 + "\n")
    
    mt5.shutdown()

if __name__ == "__main__":
    # Voit vapaasti vaihtaa tähän minkä tahansa parin (esim. "GBPJPY" tai "BTCUSD") ja katsoa miten se suoriutuu
    run_walker_backtest(symbol="EURUSD", lookback_candles=1000)