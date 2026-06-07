import jax.numpy as jnp

class DirectionScanner:
    @staticmethod
    def analyze(is_spike, delta):
        """
        Päättelee suunnan:
        - 1.0 (Bullish reversal): Absorptio + Negatiivinen Delta
        - -1.0 (Bearish reversal): Absorptio + Positiivinen Delta
        - 0.0: Ei signaalia
        """
        if not is_spike:
            return 0.0
        
        # Jos ostajien volyymi imeytyy (Negatiivinen delta), hinta nousee
        return jnp.where(delta < 0, 1.0, -1.0)