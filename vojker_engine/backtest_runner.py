import sys
import os

# Lisätään polku juureen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.compute.absorption import AbsorptionEngine
from src.compute.delta_calc import DeltaEngine
from src.compute.scanner import DirectionScanner

def run_simulation(data_stream):
    engine = AbsorptionEngine(threshold=20000)
    delta_calc = DeltaEngine()
    
    print(f"{'AIKA':<10} | {'VOLUME':<10} | {'DELTA':<10} | {'SIGNAALI'}")
    print("-" * 50)
    
    for tick in data_stream:
        is_spike = engine.detect_spike(tick['volume'])
        delta = delta_calc.calculate_delta(tick['bid_vol'], tick['ask_vol'])
        signal = DirectionScanner.analyze(is_spike, delta)
        
        if signal != 0:
            print(f"{tick['time']:<10} | {tick['volume']:<10} | {delta:<10} | {'BULLISH' if signal == 1.0 else 'BEARISH'}")

# Simuloidaan markkinapäivää
if __name__ == "__main__":
    # Tähän voit myöhemmin ladata oikean CSV-tiedoston
    mock_market_day = [
        {'time': '09:00', 'volume': 25000, 'bid_vol': 10000, 'ask_vol': 15000}, # Bullish
        {'time': '10:30', 'volume': 5000, 'bid_vol': 2000, 'ask_vol': 3000},   # Ei signaalia
        {'time': '12:15', 'volume': 30000, 'bid_vol': 20000, 'ask_vol': 10000},# Bearish
        {'time': '14:45', 'volume': 22000, 'bid_vol': 8000, 'ask_vol': 14000}, # Bullish
    ]
    
    run_simulation(mock_market_day)