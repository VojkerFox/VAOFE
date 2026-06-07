import sys
import os
import unittest
# Lisätään polku, jotta löydämme src/adapter-kansion
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapter.mt5_adapter import initialize_mt5, get_tick_data

class TestMT5Adapter(unittest.TestCase):
    
    def test_connection(self):
        """Testataan, että MT5-yhteys aukeaa."""
        result = initialize_mt5()
        self.assertTrue(result, "MT5-yhteyden pitäisi aueta onnistuneesti.")

    def test_get_data(self):
        """Testataan, että saamme dataa EURUSD-symbolista."""
        initialize_mt5()
        data = get_tick_data("EURUSD")
        self.assertIsNotNone(data, "Datan pitäisi palautua EURUSD-symbolista.")
        print(f"\nTesti haettu data: Bid={data.bid}")

if __name__ == "__main__":
    unittest.main()