import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compute.scanner import DirectionScanner
from src.notify.telegram_bot import TelegramNotifier

class TestFullIntegration(unittest.TestCase):

    def test_bullish_signal_flow(self):
        """1. Testi: Bullish-signaalin kulku."""
        mock_notifier = MagicMock()
        signal = DirectionScanner.analyze(1.0, -5000)
        if signal != 0: mock_notifier.send_alert("Bullish")
        mock_notifier.send_alert.assert_called_with("Bullish")

    def test_bearish_signal_flow(self):
        """2. Testi: Bearish-signaalin kulku."""
        mock_notifier = MagicMock()
        signal = DirectionScanner.analyze(1.0, 5000)
        if signal != 0: mock_notifier.send_alert("Bearish")
        mock_notifier.send_alert.assert_called_with("Bearish")

    def test_no_spike_no_notification(self):
        """3. Testi: Ei piikkiä, ei ilmoitusta."""
        mock_notifier = MagicMock()
        signal = DirectionScanner.analyze(0.0, -5000)
        if signal != 0: mock_notifier.send_alert("Signal")
        mock_notifier.send_alert.assert_not_called()

    def test_notifier_network_failure(self):
        """4. Testi: Notifierin vikasietoisuus (verkkohäiriö)."""
        notifier = TelegramNotifier()
        # Simuloidaan virheellinen URL, jotta saadaan poikkeus
        with patch("requests.get", side_effect=Exception("Network Error")):
            success = notifier.send_alert("Test message")
            self.assertFalse(success)

    def test_signal_value_integrity(self):
        """5. Testi: Scannerin palautusarvojen validointi."""
        self.assertEqual(DirectionScanner.analyze(1.0, -100), 1.0)
        self.assertEqual(DirectionScanner.analyze(1.0, 100), -1.0)

if __name__ == "__main__":
    unittest.main()