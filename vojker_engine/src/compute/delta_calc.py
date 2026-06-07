import jax
import jax.numpy as jnp

class DeltaEngine:
    @staticmethod
    @jax.jit
    def calculate_delta(bid_volume, ask_volume):
        """Laskee erotuksen (Aggressiiviset ostot - Aggressiiviset myynnit)."""
        return bid_volume - ask_volume