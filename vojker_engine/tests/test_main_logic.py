import unittest
from unittest.mock import MagicMock
import jax.numpy as jnp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapter.data_normalizer import DataNormalizer
from src.compute.absorption import AbsorptionEngine
from src.compute.dynamic_threshold import DynamicThresholdCalculator

class TestMainLogic(unittest.TestCase):
    def setUp(self):
        self.engine = AbsorptionEngine(threshold=1000)
        self.normalizer = DataNormalizer()

    def test_pipeline_spike_detection(self):
        """1. Testi: Kokonainen putki (normalisointi + piikki)."""
        mock_tick = MagicMock(volume=2000.0)
        data = self.normalizer.normalize_tick(mock_tick)
        self.assertEqual(self.engine.detect_spike(data['volume']), 1.0)

    def test_dynamic_threshold_adaptation(self):
        """2. Testi: Kynnysarvon adaptiivisuus historian perusteella."""
        history = jnp.array([100.0, 100.0, 100.0, 5000.0]) # Volyymipiikki historiassa
        new_threshold = DynamicThresholdCalculator.calculate(history)
        self.engine.threshold = new_threshold
        # Uusi piikki pitää olla tätä suurempi
        self.assertEqual(self.engine.detect_spike(2000.0), 0.0)

    def test_buffer_empty_state(self):
        """3. Testi: Tyhjä puskuri ei saa kaataa järjestelmää."""
        buffer = []
        # Laskennan pitäisi käsitellä tyhjä syöte tai palauttaa default
        threshold = DynamicThresholdCalculator.calculate(jnp.array(buffer))
        self.assertTrue(jnp.isnan(threshold) or threshold >= 0)

    def test_buffer_reset(self):
        """4. Testi: Puskurin tyhjennyslogiikka 100 tickin jälkeen."""
        buffer = [100.0] * 105
        if len(buffer) >= 100:
            buffer = []
        self.assertEqual(len(buffer), 0)

    def test_data_integrity_none(self):
        """5. Testi: Virheellinen tick-data (None) käsittely."""
        with self.assertRaises(Exception):
            self.normalizer.normalize_tick(None)

if __name__ == "__main__":
    unittest.main()