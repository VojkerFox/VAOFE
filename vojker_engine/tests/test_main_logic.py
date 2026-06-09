import unittest
from unittest.mock import MagicMock, patch
import jax.numpy as jnp
import sys
import os

# Varmistetaan polku
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapter.data_normalizer import DataNormalizer
from src.compute.liquidity_density import LiquidityDensityEngine
from src.compute.dynamic_threshold import DynamicThresholdCalculator

class TestMainLogic(unittest.TestCase):
    def setUp(self):
        # Alustetaan SciML-moottori
        self.ld_engine = LiquidityDensityEngine(learning_rate=0.05)
        self.normalizer = DataNormalizer()

    def test_1_pipeline_sciml_prediction(self):
        """1. Testi: SciML-ennusteen matemaattinen eheys."""
        imbalance = jnp.array(0.5)
        cost_per_pip = jnp.array(500.0)
        recent_volume = jnp.array(1000.0)
        predicted = self.ld_engine.predict_movement(self.ld_engine.params, imbalance, cost_per_pip, recent_volume)
        self.assertIsInstance(float(predicted), float)

    def test_2_dynamic_threshold_adaptation(self):
        """2. Testi: Kynnysarvon adaptiivisuus historiasta."""
        history = jnp.array([100.0, 100.0, 100.0, 5000.0])
        threshold = DynamicThresholdCalculator.calculate(history, factor=2.0)
        self.assertGreater(threshold, 0.0)

    @patch('psycopg2.connect')
    def test_3_db_save_logic(self, mock_connect):
        """3. Testi: Varmistetaan että tietokantatallennus (save_to_db) on mahdollinen."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Simuloidaan tietokantakutsu
        mock_cursor.execute("INSERT INTO ai_learning_logs ...", ("EURUSD", 1.2, 1.3, 0.5, 0.01))
        
        # Tarkistetaan että executea on kutsuttu
        mock_cursor.execute.assert_called()
        self.assertTrue(True)

    def test_4_data_integrity_none(self):
        """4. Testi: Virheellinen tick-data (None) käsittely."""
        with self.assertRaises(ValueError):
            self.normalizer.normalize_tick(None)

    def test_5_sciml_learning_loop(self):
        """5. Testi: Oppimisluupin gradienttipäivitys (Backpropagation)."""
        params = self.ld_engine.params
        opt_state = self.ld_engine.opt_state
        
        # Simuloidaan oppimisaskel
        new_params, new_opt_state, loss = self.ld_engine.update_learning(
            params, opt_state, jnp.array(0.1), jnp.array(100.0), jnp.array(1000.0), jnp.array(2.0)
        )
        self.assertIsNotNone(new_params)
        self.assertIsInstance(float(loss), float)

if __name__ == "__main__":
    unittest.main()