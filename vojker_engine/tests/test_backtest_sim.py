import unittest
from src.adapter.data_normalizer import DataNormalizer
from src.compute.absorption import AbsorptionEngine
from src.compute.delta_calc import DeltaEngine

class TestBacktestSim(unittest.TestCase):
    def test_historical_day_simulation(self):
        """Simuloi päivän dataa ja tarkistaa monta piikkiä löytyy."""
        engine = AbsorptionEngine(threshold=22000)
        delta_calc = DeltaEngine()
        
        # Simuloidaan 3 tickiä (yksi piikki, kaksi tavallista)
        mock_data = [
            {'volume': 25000, 'bid_vol': 15000, 'ask_vol': 10000},
            {'volume': 5000, 'bid_vol': 2000, 'ask_vol': 3000},
            {'volume': 30000, 'bid_vol': 10000, 'ask_vol': 20000}
        ]
        
        detected_spikes = 0
        total_delta = 0
        
        for tick in mock_data:
            if engine.detect_spike(tick['volume']):
                detected_spikes += 1
                total_delta += delta_calc.calculate_delta(tick['bid_vol'], tick['ask_vol'])
        
        self.assertEqual(detected_spikes, 2)
        self.assertEqual(total_delta, -5000) # (5000) + (-10000)

if __name__ == "__main__":
    unittest.main()