import unittest
import jax
import jax.numpy as jnp
from src.compute.absorption import AbsorptionEngine

class TestSensitivity(unittest.TestCase):
    def test_jacobian_sensitivity(self):
        """Testataan, miten järjestelmä reagoi kynnysarvon muutoksiin."""
        engine = AbsorptionEngine(threshold=22000)
        
        # Luodaan funktio, joka laskee havainnon suhteessa volyymiin
        def f(vol):
            return engine.detect_spike(vol)
        
        # Lasketaan jakobiaani (tässä tapauksessa derivaatta)
        jacobian_fn = jax.jacobian(f)
        
        # Testataan herkkyys kohdassa 22000
        # (JAX palauttaa 0, koska funktio on porrasfunktio)
        sensitivity = jacobian_fn(22000.0)
        self.assertEqual(sensitivity, 0.0)
        print(f"Järjestelmän paikallinen herkkyys (Jakobiaani): {sensitivity}")

if __name__ == "__main__":
    unittest.main()