import jax.numpy as jnp

class DynamicThresholdCalculator:
    @staticmethod
    def calculate(volumes, factor=3.0):
        """
        Laskee dynaamisen kynnysarvon: keskiarvo + factor * keskihajonta.
        Tämä adaptoituu markkinan volyymitason muutoksiin.
        """
        mean = jnp.mean(volumes)
        std = jnp.std(volumes)
        return mean + (factor * std)