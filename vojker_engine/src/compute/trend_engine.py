class TrendEngine:
    @staticmethod
    def calculate_macd_all(prices, fast=12, slow=26, signal=9):
        """
        Laskee MACD:n 64-bittisellä tarkkuudella (välttää JAX float32 kohinan).
        Palauttaa: macd_line, signal_line, histogram
        """
        # Muutetaan puhtaiksi 64-bit Python floateiksi
        prices_list = [float(p) for p in prices]
        
        # 1. Nopea EMA (12)
        alpha_fast = 2.0 / (fast + 1.0)
        ema_fast = []
        current_fast = prices_list[0]
        for p in prices_list:
            current_fast = (p - current_fast) * alpha_fast + current_fast
            ema_fast.append(current_fast)
            
        # 2. Hidas EMA (26)
        alpha_slow = 2.0 / (slow + 1.0)
        ema_slow = []
        current_slow = prices_list[0]
        for p in prices_list:
            current_slow = (p - current_slow) * alpha_slow + current_slow
            ema_slow.append(current_slow)
            
        # 3. MACD Line
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        
        # 4. Signal Line (9)
        alpha_signal = 2.0 / (signal + 1.0)
        signal_line = []
        current_signal = macd_line[0]
        for m in macd_line:
            current_signal = (m - current_signal) * alpha_signal + current_signal
            signal_line.append(current_signal)
            
        # 5. Histogram (MACD - Signal)
        histogram = [m - s for m, s in zip(macd_line, signal_line)]
        
        return macd_line, signal_line, histogram