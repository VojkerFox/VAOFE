import sys
import os
import unittest
from unittest.mock import MagicMock
import pandas as pd

# Lisätään polku, jotta src löytyy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapter.data_normalizer import DataNormalizer

class TestDataNormalizer(unittest.TestCase):
    def setUp(self):
        self.mock_tick = MagicMock()
        self.mock_tick.time_msc = 1780760000000
        self.mock_tick.bid = 1.15186
        self.mock_tick.ask = 1.15193
        self.mock_tick.last = 1.15190
        self.mock_tick.volume = 10.0

    def test_normalization_structure(self):
        result = DataNormalizer.normalize_tick(self.mock_tick)
        self.assertIn("bid", result)
        self.assertEqual(result["bid"], 1.15186)

    def test_types(self):
        result = DataNormalizer.normalize_tick(self.mock_tick)
        self.assertIsInstance(result["bid"], float)

    def test_null_data(self):
        with self.assertRaises(ValueError):
            DataNormalizer.normalize_tick(None)

    def test_zero_volume(self):
        self.mock_tick.volume = 0.0
        result = DataNormalizer.normalize_tick(self.mock_tick)
        self.assertEqual(result["volume"], 0.0)

if __name__ == "__main__":
    unittest.main()