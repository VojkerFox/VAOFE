import jax
import jax.numpy as jnp

@jax.jit
def calculate_triad_inertia(volume, macd_hist, friction_weight):
    """
    Laskee puhtaan tilausvirran Inertiaindeksin.
    Korkea inertia = Markkina on "raskas", seinää puretaan kovalla volyymilla mutta momentum ei kasva.
    Matala inertia = Markkina on "kevyt", hinta liukuu helposti.
    """
    abs_macd = jnp.abs(macd_hist) + 1e-7
    inertia_index = (volume / abs_macd) * friction_weight
    return inertia_index

@jax.jit
def calculate_synthetic_dom(tick_delta, tick_volume, price_change_pips):
    """
    Kvantitatiivinen Kyle's Lambda / Price Impact -malli.
    Laskee reaaliaikaisen synteettisen tilauskirjan epätasapainon (Synthetic DOM).
    
    Palauttaa:
    - synthetic_imbalance: -1.0 (Näkymätön myyntimuuri) ... +1.0 (Näkymätön ostoseinä)
    - liquidity_stiffness: Mitä suurempi luku, sitä paksumpi piiloseinä on imemässä voimaa.
    """
    # Estetään nollalla jakaminen dynaamisella epsilonilla
    abs_price_move = jnp.abs(price_change_pips) + 1e-4
    
    # Markkinan kireys (Stiffness): Kuinka paljon volyymia transaktoitiin per liikutettu pipsi
    liquidity_stiffness = tick_volume / abs_price_move
    
    # Lasketaan raaka dynaaminen paine
    raw_pressure = tick_delta / (tick_volume + 1e-7)
    
    # Jos kireys (stiffness) on valtava (eli volyymi hakkaa tyhjää ilman hinnanmuutosta), 
    # se vahvistaa piilossa olevan absorptioseinän läsnäolon.
    stiffness_factor = jnp.clip(jnp.log1p(liquidity_stiffness) / 8.0, 0.1, 2.0)
    synthetic_imbalance = jnp.clip(raw_pressure * stiffness_factor, -1.0, 1.0)
    
    return synthetic_imbalance, liquidity_stiffness