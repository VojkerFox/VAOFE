import unittest
import sys
import os

# Lisätään projektin juuri sys.path-listaan
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compute.scanner import DirectionScanner

class TestScanner(unittest.TestCase):
    def test_bullish_signal(self):
        """1. Absorptio + Negatiivinen Delta = Bullish."""
        self.assertEqual(DirectionScanner.analyze(1.0, -5000), 1.0)

    def test_bearish_signal(self):
        """2. Absorptio + Positiivinen Delta = Bearish."""
        self.assertEqual(DirectionScanner.analyze(1.0, 5000), -1.0)

    def test_no_spike_no_signal(self):
        """3. Ei piikkiä = Ei signaalia."""
        self.assertEqual(DirectionScanner.analyze(0.0, -5000), 0.0)

    def test_neutral_delta(self):
        """4. Absoluuttinen nollatilanne (harvinainen)."""
        self.assertEqual(DirectionScanner.analyze(1.0, 0), -1.0)

if __name__ == "__main__":
    unittest.main()