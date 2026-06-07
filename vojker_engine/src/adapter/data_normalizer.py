import pandas as pd

class DataNormalizer:
    @staticmethod
    def normalize_tick(tick_data):
        if tick_data is None:
            raise ValueError("Data puuttuu")
        
        return {
            "timestamp": pd.to_datetime(tick_data.time_msc, unit='ms'),
            "bid": float(tick_data.bid),
            "ask": float(tick_data.ask),
            "last": float(tick_data.last),
            "volume": float(tick_data.volume)
        }