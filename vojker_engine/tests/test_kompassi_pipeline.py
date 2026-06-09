import unittest
import jax.numpy as jnp
import sys
import os

# Lisätään juuri polkuun, jotta omat moduulisi löytyvät
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compute.dynamic_threshold import DynamicThresholdCalculator
from src.compute.delta_calc import DeltaEngine
from src.compute.scanner import DirectionScanner

class TestKompassiPipeline(unittest.TestCase):
    
    def test_1_bullish_reversal_pipeline(self):
        """Testi 1: Täydellinen ostosignaali (Bullish). 
        Volyymi ylittää kynnyksen (spike) ja delta on vahvasti negatiivinen (myyjät imettiin)."""
        history_vol = jnp.array([100.0, 110.0, 90.0, 100.0, 105.0]) # Normaali volyymi ~100
        recent_vol = 500.0 # Iso piikki
        
        threshold = DynamicThresholdCalculator.calculate(history_vol, factor=3.0)
        is_spike = 1.0 if recent_vol > threshold else 0.0
        
        delta = DeltaEngine.calculate_delta(100.0, 400.0) # 100 ostoa, 400 myyntiä = -300 delta
        
        signal = DirectionScanner.analyze(is_spike, delta)
        self.assertEqual(signal, 1.0, "Pitäisi antaa 1.0 (OSTA) signaali")

    def test_2_bearish_reversal_pipeline(self):
        """Testi 2: Täydellinen myyntisignaali (Bearish). 
        Volyymi ylittää kynnyksen ja delta on vahvasti positiivinen (ostajat pysäytettiin seinään)."""
        history_vol = jnp.array([50.0, 60.0, 55.0, 45.0, 50.0]) # Normaali volyymi ~50
        recent_vol = 300.0 # Iso piikki
        
        threshold = DynamicThresholdCalculator.calculate(history_vol, factor=3.0)
        is_spike = 1.0 if recent_vol > threshold else 0.0
        
        delta = DeltaEngine.calculate_delta(250.0, 50.0) # 250 ostoa, 50 myyntiä = +200 delta
        
        signal = DirectionScanner.analyze(is_spike, delta)
        self.assertEqual(signal, -1.0, "Pitäisi antaa -1.0 (MYY) signaali")

    def test_3_no_spike_ignored(self):
        """Testi 3: Normaali markkina (Ei signaalia).
        Vaikka delta olisi epätasapainossa, jos volyymipiikkiä ei ole, ei tehdä mitään."""
        history_vol = jnp.array([200.0, 210.0, 190.0, 205.0]) # Normaali volyymi ~200
        recent_vol = 220.0 # Ei ylitä 3x keskihajonnan rajaa
        
        threshold = DynamicThresholdCalculator.calculate(history_vol, factor=3.0)
        is_spike = 1.0 if recent_vol > threshold else 0.0
        
        delta = DeltaEngine.calculate_delta(220.0, 0.0) # Täysin yksisuuntainen delta!
        
        signal = DirectionScanner.analyze(is_spike, delta)
        self.assertEqual(signal, 0.0, "Ei piikkiä = pitäisi antaa 0.0 signaali")

    def test_4_extreme_volatility_adaptation(self):
        """Testi 4: Järjestelmä mukautuu uutisten aikaan (NFP / FED).
        Historiassa on jo massiivisia piikkejä. Kynnyksen pitää nousta niin korkealle, 
        että normaali iso volyymi ei enää laukaise vääriä signaaleja."""
        history_vol = jnp.array([1000.0, 5000.0, 2000.0, 8000.0]) # Markkina käy todella kuumana
        recent_vol = 3000.0 # Tavallisesti tämä olisi piikki, mutta ei nyt!
        
        threshold = DynamicThresholdCalculator.calculate(history_vol, factor=3.0)
        is_spike = 1.0 if recent_vol > threshold else 0.0
        delta = DeltaEngine.calculate_delta(0.0, 3000.0)
        
        signal = DirectionScanner.analyze(is_spike, delta)
        self.assertEqual(signal, 0.0, "Uutistilanteessa kynnyksen pitää suojata vääristä signaaleista (0.0)")

    def test_5_dead_market_zero_division_safety(self):
        """Testi 5: Kuollut markkina (viikonloppu tai pyhä).
        Volyymit ovat nollassa. Varmistetaan, ettei JAX tai matematiikka kaadu nollalla jakamiseen."""
        history_vol = jnp.array([0.0, 0.0, 0.0, 0.0])
        recent_vol = 0.0
        
        threshold = DynamicThresholdCalculator.calculate(history_vol, factor=3.0)
        is_spike = 1.0 if recent_vol > threshold else 0.0
        delta = DeltaEngine.calculate_delta(0.0, 0.0)
        
        signal = DirectionScanner.analyze(is_spike, delta)
        self.assertEqual(signal, 0.0, "Kuollut markkina pitää palauttaa turvallisesti 0.0")
        
    def test_6_tick_estimation_logic(self):
        """Testi 6: Estimoitu Delta laskenta (Simuloidaan MT5 dataa).
        Varmistetaan, että hinnanmuutoslaskenta osaa erotella ostot ja myynnit."""
        # Simuloidaan 4 peräkkäistä tickiä (ask-hinta nousee -> osto, bid-hinta laskee -> myynti)
        simulated_ticks = [
            {'ask': 1.1000, 'bid': 1.0998, 'volume': 10}, # Aloitustick
            {'ask': 1.1002, 'bid': 1.1000, 'volume': 50}, # Hinta nousi -> Osto (+50)
            {'ask': 1.1002, 'bid': 1.0995, 'volume': 80}, # Bid laski -> Myynti (+80)
            {'ask': 1.1004, 'bid': 1.0997, 'volume': 20}  # Hinta nousi -> Osto (+20)
        ]
        
        bid_vol = 0.0
        ask_vol = 0.0
        for i in range(1, len(simulated_ticks)):
            if simulated_ticks[i]['ask'] > simulated_ticks[i-1]['ask']:
                bid_vol += simulated_ticks[i]['volume']
            elif simulated_ticks[i]['bid'] < simulated_ticks[i-1]['bid']:
                ask_vol += simulated_ticks[i]['volume']
                
        # Odotamme: Ostoja 50 + 20 = 70. Myyntejä 80.
        self.assertEqual(bid_vol, 70.0, "Ostovolyymin pitäisi olla 70")
        self.assertEqual(ask_vol, 80.0, "Myyntivolyymin pitäisi olla 80")
        self.assertEqual(DeltaEngine.calculate_delta(bid_vol, ask_vol), -10.0, "Deltan pitäisi olla -10")

if __name__ == "__main__":
    unittest.main(verbosity=2)