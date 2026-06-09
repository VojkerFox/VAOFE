import jax
import jax.numpy as jnp
import optax
from functools import partial

class LiquidityDensityEngine:
    def __init__(self, learning_rate=0.01):
        """
        Alustaa oppivan Orderbook-haistelijan.
        """
        self.optimizer = optax.adam(learning_rate)
        
        # Alkuperäinen "kitkapaino" (Liquidity resistance weight)
        self.params = {
            "friction_weight": jnp.array(1.0, dtype=jnp.float32)
        }
        self.opt_state = self.optimizer.init(self.params)

    @staticmethod
    @jax.jit
    def calculate_cost_per_pip(highs, lows, volumes, pip_size=0.0001):
        """
        Laskee, kuinka paljon volyymia vaadittiin yhden pipsin siirtämiseen.
        """
        price_movement_pips = jnp.maximum((highs - lows) / pip_size, 1e-7)
        cost_per_pip = volumes / price_movement_pips
        return cost_per_pip

    @staticmethod
    @jax.jit
    def analyze_l2_imbalance(bids_volume, asks_volume):
        """
        Laskee L2-tason (Depth of Market) epätasapainon.
        """
        total_bid = jnp.sum(bids_volume)
        total_ask = jnp.sum(asks_volume)
        total_volume = total_bid + total_ask + 1e-7
        
        imbalance = (total_bid - total_ask) / total_volume
        return imbalance

    @staticmethod
    @jax.jit
    def predict_movement(params, imbalance, cost_per_pip, recent_volume):
        """
        Ennustaa hinnan liikkeen.
        """
        force = imbalance * recent_volume
        expected_pips = (force / (cost_per_pip + 1e-7)) * params["friction_weight"]
        return expected_pips

    @staticmethod
    @jax.jit
    def _loss_fn(params, imbalance, cost_per_pip, recent_volume, actual_pips_moved):
        """
        Tappiofunktio (Mean Squared Error).
        """
        predicted_pips = LiquidityDensityEngine.predict_movement(params, imbalance, cost_per_pip, recent_volume)
        return jnp.square(predicted_pips - actual_pips_moved)

    # KORJAUS: Kerrotaan JAXille, että argumentti 0 (self) on staattinen olio, ei matriisi!
    @partial(jax.jit, static_argnums=(0,))
    def update_learning(self, params, opt_state, imbalance, cost_per_pip, recent_volume, actual_pips_moved):
        """
        Päivittää kitkapainot Optaxilla backpropagationin kautta.
        """
        loss, grads = jax.value_and_grad(self._loss_fn)(
            params, imbalance, cost_per_pip, recent_volume, actual_pips_moved
        )
        
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss