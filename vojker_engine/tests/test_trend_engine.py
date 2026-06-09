import unittest
import jax.numpy as jnp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.compute.trend_engine import TrendEngine

class TestTrendEngine(unittest.TestCase):
    def test_bullish_trend(self):
        # Selkeä tasainen nousu
        prices = jnp.linspace(1.0, 1.5, 30)
        trend = TrendEngine.calculate_macd(prices)
        self.assertEqual(trend, 1.0)

    def test_bearish_trend(self):
        # Selkeä tasainen lasku
        prices = jnp.linspace(1.5, 1.0, 30)
        trend = TrendEngine.calculate_macd(prices)
        self.assertEqual(trend, -1.0)

if __name__ == "__main__":
    unittest.main()