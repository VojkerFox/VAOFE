import sys
import os
import unittest
import jax.numpy as jnp
from unittest.mock import MagicMock

# Lisätään polku, jotta src löytyy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compute.absorption import AbsorptionEngine

class TestAbsorptionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AbsorptionEngine(threshold=22000)

    def test_spike_detection(self):
        """Testataan kynnysarvon ylitys ja alitus."""
        self.assertEqual(self.engine.detect_spike(25000), 1.0)
        self.assertEqual(self.engine.detect_spike(10000), 0.0)

    def test_threshold_boundary(self):
        """Testataan tasan kynnysarvolla."""
        self.assertEqual(self.engine.detect_spike(22000), 1.0)

    def test_batch_processing(self):
        """Testataan JAX-batch-käsittely usealla arvolla."""
        volumes = jnp.array([10000, 22000, 30000, 5000])
        results = self.engine.batch_process(volumes)
        expected = jnp.array([0.0, 1.0, 1.0, 0.0])
        self.assertTrue(jnp.array_equal(results, expected))

    def test_negative_values(self):
        """Varmistetaan, että negatiiviset arvot eivät sekoita."""
        self.assertEqual(self.engine.detect_spike(-100), 0.0)

if __name__ == "__main__":
    unittest.main()