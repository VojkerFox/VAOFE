import unittest
import jax.numpy as jnp
import sys
import os

# Varmistetaan polku
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compute.liquidity_density import LiquidityDensityEngine

class TestLiquidityDensity(unittest.TestCase):
    def setUp(self):
        self.engine = LiquidityDensityEngine(learning_rate=0.1)
        self.params = self.engine.params
        self.opt_state = self.engine.opt_state

    def test_calculate_cost_per_pip(self):
        """1. Testi: Normaali cost-per-pip laskenta."""
        # High = 1.1020, Low = 1.1010 -> Liike 10 pipsiä. Volyymi 5000.
        highs = jnp.array([1.1020])
        lows = jnp.array([1.1010])
        volumes = jnp.array([5000.0])
        
        cpp = LiquidityDensityEngine.calculate_cost_per_pip(highs, lows, volumes, pip_size=0.0001)
        # 5000 / 10 = 500
        self.assertAlmostEqual(float(cpp[0]), 500.0, places=1)

    def test_zero_price_movement(self):
        """2. Testi: Nollalla jakamisen esto (Epsilon-suojaus), jos hinta ei liiku."""
        highs = jnp.array([1.1010])
        lows = jnp.array([1.1010])
        volumes = jnp.array([1000.0])
        
        cpp = LiquidityDensityEngine.calculate_cost_per_pip(highs, lows, volumes)
        self.assertFalse(jnp.isnan(cpp[0])) # Ei saa olla NaN

    def test_analyze_l2_imbalance(self):
        """3. Testi: L2-epätasapainon rajat (-1.0 ... 1.0)."""
        bids = jnp.array([1000.0, 500.0])
        asks = jnp.array([0.0, 0.0])
        
        imbalance = LiquidityDensityEngine.analyze_l2_imbalance(bids, asks)
        self.assertAlmostEqual(float(imbalance), 1.0, places=4) # Täysi ostopaine

    def test_predict_movement(self):
        """4. Testi: Forward-pass ennuste palauttaa oikean suuntaisen luvun."""
        imbalance = jnp.array(0.5) # Nouseva
        cost_per_pip = jnp.array(500.0)
        recent_volume = jnp.array(1000.0)
        
        predicted = LiquidityDensityEngine.predict_movement(self.params, imbalance, cost_per_pip, recent_volume)
        self.assertGreater(float(predicted), 0.0) # Pitäisi ennustaa positiivista pips-liikettä

    def test_update_learning(self):
        """5. Testi: Optax päivittää kitkapainoa (Backpropagation)."""
        imbalance = jnp.array(1.0)
        cost_per_pip = jnp.array(100.0)
        recent_volume = jnp.array(1000.0)
        actual_pips = jnp.array(5.0) # Oikea liike olikin vain 5 pipsiä
        
        # Alkuperäinen ennuste (paino = 1.0): 1000 / 100 = 10 pipsiä.
        # Markkina liikkuikin vain 5, eli kitkaa on odotettua MIKÄ. Painon pitäisi pienentyä.
        new_params, new_state, loss = self.engine.update_learning(
            self.params, self.opt_state, imbalance, cost_per_pip, recent_volume, actual_pips
        )
        
        self.assertLess(float(new_params["friction_weight"]), float(self.params["friction_weight"]))
        self.assertGreater(float(loss), 0.0)

if __name__ == "__main__":
    unittest.main()