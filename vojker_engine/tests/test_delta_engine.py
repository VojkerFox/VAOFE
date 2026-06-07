import unittest
import jax.numpy as jnp
import sys
import os

# Polun korjaus
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compute.delta_calc import DeltaEngine

class TestDeltaEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DeltaEngine()

    def test_positive_delta(self):
        """1. Testi: Puhtaasti ostovoimainen markkina."""
        self.assertEqual(self.engine.calculate_delta(1000, 200), 800)

    def test_negative_delta(self):
        """2. Testi: Puhtaasti myyntivoimainen markkina."""
        self.assertEqual(self.engine.calculate_delta(200, 1000), -800)

    def test_neutral_delta(self):
        """3. Testi: Absorptio/Tasapaino."""
        self.assertEqual(self.engine.calculate_delta(500, 500), 0)

    def test_jax_batch_consistency(self):
        """4. Testi: Varmistetaan JAX-vektorisoinnin toimivuus."""
        bids = jnp.array([1000, 2000])
        asks = jnp.array([500, 2500])
        results = self.engine.calculate_delta(bids, asks)
        expected = jnp.array([500, -500])
        self.assertTrue(jnp.array_equal(results, expected))

    def test_zero_input(self):
        """5. Testi: Tyhjän volyymin käsittely."""
        self.assertEqual(self.engine.calculate_delta(0, 0), 0)

if __name__ == "__main__":
    unittest.main()