import jax
import jax.numpy as jnp
import os

class AbsorptionEngine:
    def __init__(self, threshold=None):
        # Jos thresholdia ei anneta, luetaan se ympäristömuuttujasta (tai asetetaan oletus)
        self.threshold = threshold or float(os.getenv("ABSORPTION_THRESHOLD", 22000))

    @staticmethod
    @jax.jit
    def _jax_detect(volumes, threshold):
        return jnp.where(volumes >= threshold, 1.0, 0.0)

    def detect_spike(self, volume):
        return self._jax_detect(jnp.array([volume]), self.threshold)[0]