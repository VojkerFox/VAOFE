class WickScanner:
    @staticmethod
    def detect_rejection(open_p, high_p, low_p, close_p):
        """
        Tunnistaa institutionaalisen hylkäyshännän (Liquidity Sweep / Pin Bar).
        
        Säännöt:
        1. Hännän pitää olla vähintään 2 kertaa isompi kuin kynttilän rungon.
        2. Hännän pitää muodostaa vähintään 50 % koko kynttilän pituudesta.
        
        Palauttaa: 
        (Signaali, Hännän osuus prosentteina)
        Signaali: 1.0 (Bullish), -1.0 (Bearish), 0.0 (Ei signaalia)
        """
        total_range = high_p - low_p
        if total_range == 0:
            return 0.0, 0.0

        body = abs(open_p - close_p)
        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p

        # Bearish Hylkäys (Hinta kävi ylhäällä, mutta myytiin rajusti alas)
        if upper_wick >= (2 * body) and upper_wick >= (0.5 * total_range):
            wick_percentage = (upper_wick / total_range) * 100
            return -1.0, wick_percentage
            
        # Bullish Hylkäys (Hinta kävi alhaalla, mutta ostettiin rajusti ylös)
        elif lower_wick >= (2 * body) and lower_wick >= (0.5 * total_range):
            wick_percentage = (lower_wick / total_range) * 100
            return 1.0, wick_percentage
            
        return 0.0, 0.0