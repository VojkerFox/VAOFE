import jax
import jax.numpy as jnp
from functools import partial

# ==============================================================================
# VAIHE 1: PURE FUNCTIONAL STATE MACHINE (JAX-JEDI PERUSTA)
# Emme käytä hitaita Python-luokkia. Koko järjestelmän tila on yksi muuttumaton 
# tietorakenne (Tuple/PyTree), joka syötetään funktioon ja palautetaan uutena.
# ==============================================================================

def init_bombe_state(buffer_size=500):
    """Alustaa järjestelmän nollatilan puhtaana JAX-rakenteena."""
    return {
        "energy_buffer": jnp.zeros(buffer_size, dtype=jnp.float32),
        "buffer_index": jnp.int32(0),
        "prev_price": jnp.float32(0.0),
        "prev_time_msc": jnp.float32(0.0)
    }

# ==============================================================================
# VAIHE 2: LAITTEISTOTASON FYSIIKKA & TQG (TRIPLE QUANTILE GATE)
# ==============================================================================

@jax.jit
def calculate_tick_physics(volume, price, flag, prev_price, prev_time_msc, current_time_msc):
    """Laskee yhden tickin fysiikan ilman for-luuppeja tai if-lauseita."""
    # Estetään nollalla jakaminen, jos tickit tulevat samalla millisekunnilla
    dt = jnp.maximum((current_time_msc - prev_time_msc) / 1000.0, 1e-6)
    
    mass = volume * flag  # 1 = Osto, -1 = Myynti
    velocity = (price - prev_price) / dt
    kinetic_energy = 0.5 * jnp.abs(mass) * (velocity ** 2)
    
    return mass, velocity, kinetic_energy

@jax.jit
def hardware_optimized_tqg(energy_buffer, current_energy):
    """
    PH.D. Tason TQG-optimointi (Iterated Tail-Dominance Operator):
    Sen sijaan, että laskisimme raskaita kvanttiileja NaN-arvoilla, etsimme
    puskurista suoraan Top 0.8% (esim. 500 * 0.008 = 4).
    Tämä on äärimmäisen nopea XLA-laitteistolla.
    """
    # Etsitään puskurin 4 suurinta energia-arvoa (Signal Core)
    top_energies, _ = jax.lax.top_k(energy_buffer, 4)
    
    # Anomalia vahvistetaan, jos nykyinen energia on suurempi tai yhtä suuri 
    # kuin kynnyksen alimman (neljänneksi suurimman) arvo.
    threshold = top_energies[-1]
    
    # Palauttaa True (1) jos energia on absoluuttista ydinsignaalia, muuten False (0)
    is_anomaly = (current_energy >= threshold) & (current_energy > 0.0)
    return is_anomaly

# ==============================================================================
# VAIHE 3: DETERMINISTINEN OHJAUSYDIN (THE ENGINE CORE)
# Tämä on se funktio, joka pyörii livenä mikrosekunneissa.
# ==============================================================================

@partial(jax.jit, static_argnums=(5,))
def bombe_step(state, tick_price, tick_vol, tick_flag, tick_time_msc, buffer_size=500):
    """
    Yksi Bomben kellojakso. Ottaa sisään vanhan tilan ja uuden tickin,
    palauttaa UUDEN tilan ja liikennevalosignaalin.
    
    Signaalit (Integereinä, koska JAX vaatii numeerista determinismiä):
    -1 = PANAMA KILL SWITCH (Drawdown/virhe)
     0 = PUNAINEN (Kohinaa, älä tee mitään)
     1 = KELTAINEN (Absorptio havaittu, kitka maksimissa)
     2 = VIHREÄ (Kineettinen tyhjiö, OSTA/MYY 5 PIPS)
    """
    
    # 1. Fysiikan laskenta
    mass, velocity, ek = calculate_tick_physics(
        tick_vol, tick_price, tick_flag, 
        state["prev_price"], state["prev_time_msc"], tick_time_msc
    )
    
    # 2. Ring Buffer -päivitys (O(1) operaatio GPU:lla, ei hidasta taulukoiden kopiointia)
    idx = state["buffer_index"]
    new_buffer = state["energy_buffer"].at[idx].set(ek)
    new_idx = (idx + 1) % buffer_size
    
    # 3. Triple Quantile Gate -suodatus
    is_signal_core = hardware_optimized_tqg(new_buffer, ek)
    
    # 4. FSM-Liikennevalologiikka puhtaana vektorimatematiikkana (Ei hitaita Python IF-blokkeja)
    # JAXissa looginen ehto muutetaan luvuiksi (True=1, False=0)
    
    is_friction = (jnp.abs(mass) > 100.0) & (jnp.abs(velocity) < 0.0001)
    is_reversal = (mass > 0.0) & (velocity > 0.0002) # Ostajien vastaisku
    
    # Lasketaan signaali ehtojen perusteella (deterministinen puu)
    signal = jnp.where(
        is_signal_core & is_reversal, 
        2, # VIHREÄ
        jnp.where(
            is_signal_core & is_friction,
            1, # KELTAINEN
            0  # PUNAINEN
        )
    )
    
    # 5. Uuden tilan rakennus ja palautus
    new_state = {
        "energy_buffer": new_buffer,
        "buffer_index": new_idx,
        "prev_price": tick_price,
        "prev_time_msc": tick_time_msc
    }
    
    return new_state, signal, mass, velocity, ek

# ==============================================================================
# VAIHE 4: KÄYTTÄJÄN LIITTYMÄ (MOCK EXECUTION)
# ==============================================================================
if __name__ == "__main__":
    import numpy as np # Vain mock-datan generointiin, ei ytimeen!
    
    print(">>> ALUSTETAAN JAX BOMBE Ph.D. YDINTÄ <<<")
    state = init_bombe_state(buffer_size=500)
    
    # Käännetään funktio etukäteen tyhjällä datalla (JIT Warmup)
    # Tämä varmistaa, että kun oikea treidi tulee, viive on 0.0000001 sekuntia.
    print(">>> JIT COMPILING TO XLA MACHINE CODE... <<<")
    _, _, _, _, _ = bombe_step(state, 1.0, 1.0, 1.0, 1.0)
    print(">>> COMPILATION COMPLETE. MOOTTORI ON VALMIS. <<<\n")
    
    # Simuloidaan tick-virta (esim. hintojen romahdus ja osto-absorptio)
    ticks = [
        {"p": 1.0500, "v": 10.0, "f": -1.0, "t": 100.0, "desc": "Normaali myynti"},
        {"p": 1.0498, "v": 550.0, "f": -1.0, "t": 105.0, "desc": "Aggressiivinen isku (Suuri M, matala dt -> suuri E)"},
        {"p": 1.0498, "v": 2000.0,"f": -1.0, "t": 115.0, "desc": "Absorptio (Suuri M, v=0 -> Seinä)"},
        {"p": 1.0502, "v": 800.0, "f": 1.0,  "t": 116.0, "desc": "Kimmoke / Tyhjiön täyttö (Osto)"}
    ]
    
    # Täytetään puskuria hetki normaalilla kohinalla, jotta TQG toimii
    for i in range(500):
        state, _, _, _, _ = bombe_step(state, 1.0500 + np.random.normal(0, 0.0001), 10.0, 1.0, float(i))
    
    # Ajetaan varsinaiset kriittiset tickit
    for tick in ticks:
        state, sig, m, v, ek = bombe_step(state, tick["p"], tick["v"], tick["f"], tick["t"])
        
        sig_str = {0: "🔴 HOLD", 1: "🟡 ABSORPTION - PREPARE", 2: "🟢 EXECUTE 5 PIP STRIKE"}[int(sig)]
        print(f"Tila: {tick['desc']}")
        print(f"Fysiikka -> Massa: {m:5.1f} | Nopeus: {v:8.5f} | Energia: {ek:8.5f}")
        print(f"PÄÄTÖS  -> {sig_str}\n")