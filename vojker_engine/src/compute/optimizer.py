import jax
import jax.numpy as jnp
import optax

class AdaptiveOptimizer:
    def __init__(self, initial_threshold=22000.0):
        self.params = jnp.array([initial_threshold])
        self.optimizer = optax.adam(learning_rate=0.01)
        self.opt_state = self.optimizer.init(self.params)

    def update_threshold(self, loss):
        """Päivittää kynnysarvon gradientin perusteella."""
        grads = jax.grad(lambda p: p * loss)(self.params) # Yksinkertaistettu häviöfunktio
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return self.params[0]