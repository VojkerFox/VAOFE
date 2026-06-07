import unittest
import jax.numpy as jnp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compute.dynamic_threshold import DynamicThresholdCalculator

class TestDynamicThreshold(unittest.TestCase):
    
    def test_standard_deviation_scaling(self):
        """1. Testi: Normaali markkinajakauma."""
        volumes = jnp.array([1000.0, 1100.0, 900.0, 1050.0, 950.0])
        threshold = DynamicThresholdCalculator.calculate(volumes, factor=3.0)
        self.assertTrue(1000.0 < threshold < 1500.0)

    def test_flat_market(self):
        """2. Testi: Täysin vakaa markkina (std = 0)."""
        volumes = jnp.array([1000.0, 1000.0, 1000.0])
        threshold = DynamicThresholdCalculator.calculate(volumes, factor=3.0)
        self.assertEqual(threshold, 1000.0)

    def test_single_value_input(self):
        """3. Testi: Minimisyöte (yksi arvo)."""
        volumes = jnp.array([5000.0])
        threshold = DynamicThresholdCalculator.calculate(volumes, factor=3.0)
        self.assertEqual(threshold, 5000.0)

    def test_high_volatility(self):
        """4. Testi: Voimakas volatiliteetti nostaa kynnystä."""
        low_vol = jnp.array([100.0, 100.0])
        high_vol = jnp.array([100.0, 10000.0])
        t1 = DynamicThresholdCalculator.calculate(low_vol)
        t2 = DynamicThresholdCalculator.calculate(high_vol)
        self.assertGreater(t2, t1)

    def test_negative_factor(self):
        """5. Testi: Epälooginen kerroin (Cpk-stressitesti)."""
        volumes = jnp.array([1000.0, 2000.0])
        # Jos faktori on negatiivinen, kynnyksen pitäisi laskea keskiarvon alle
        threshold = DynamicThresholdCalculator.calculate(volumes, factor=-1.0)
        self.assertLess(threshold, 1500.0)

if __name__ == "__main__":
    unittest.main()