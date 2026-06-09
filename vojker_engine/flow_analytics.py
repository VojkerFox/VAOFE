import jax
import jax.numpy as jnp

@jax.jit
def calculate_vacuum_burst(history_matrix):
    """
    Puhdas, JIT-optimoitu JAX-funktio yhdelle tai useammalle parille.
    history_matrix muotoa: (history_len, 3) -> Sarakkeet: [kitka, volyymi, momentum]
    """
    # Erotetaan komponentit tensorista
    friction = history_matrix[:, 0]
    volume = history_matrix[:, 1]
    momentum = history_matrix[:, 2]

    # Lasketaan ensimmäiset ja toiset derivaatat (Kiihtyvyys ja Voima)
    dfriction = jnp.gradient(friction)
    dmomentum = jnp.gradient(momentum)
    dvolume = jnp.gradient(volume)

    # VOIMAKAS TYHJIÖ-ANALYYSI (The Vacuum Logic):
    # Jos momentum on kova, volyymi kasvaa, mutta kitka (vastus) romahtaa jyrkästi alas,
    # kyseessä on likviditeettiseinän murtuminen -> hinta slipaa tyhjiöön.
    
    current_friction = friction[-1]
    current_momentum = momentum[-1]
    
    # Kitkan muutosnopeus (jos negatiivinen, seinä sulaa alta)
    friction_velocity = dfriction[-1]
    
    # Momentumin kiihtyvyys
    momentum_acceleration = jnp.gradient(dmomentum)[-1]

    # Lasketaan indikaattori tyhjiölle (VBI - Vacuum Burst Index)
    # Suuri arvo tarkoittaa, että jarrut on vapautettu ja momentum vetää takaa
    vbi = (jnp.abs(current_momentum) * dvolume[-1]) / (jnp.abs(current_friction) + 0.1)
    
    # Signaalilogiikka: Pitääkö iskeä sisään?
    # Aktivoituu, kun kitka ohenee vauhdilla (friction_velocity < -0.5) ja momentum vahvistuu
    trigger_short = (current_momentum < 0) & (friction_velocity < -0.2) & (vbi > 1.5)
    trigger_long = (current_momentum > 0) & (friction_velocity < -0.2) & (vbi > 1.5)

    direction = jnp.where(trigger_long, 1.0, jnp.where(trigger_short, -1.0, 0.0))
    confidence = jnp.clip(vbi * 10.0, 0.0, 100.0)

    return direction, confidence, vbi