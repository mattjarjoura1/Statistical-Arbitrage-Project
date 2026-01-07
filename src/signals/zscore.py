import pandas as pd
import numpy as np

class ZScoreGenerator:
    """
    Responsibility: Signal Conditioning.
    Normalizes the spread into a Z-Score so we can use standard thresholds.
    """
    
    def __init__(self, window=30):
        self.window = window # Lookback period (e.g., 30 bars)

    def compute(self, spread_series: pd.Series):
        """
        Input: A pandas Series of the raw spread (Price_A - Beta * Price_B)
        Output: A Series of Z-Scores
        """
        # 1. Rolling Mean (Moving Average)
        rolling_mean = spread_series.rolling(window=self.window).mean()
        
        # 2. Rolling Standard Deviation (Volatility)
        rolling_std = spread_series.rolling(window=self.window).std()
        
        # 3. Z-Score Calculation: (Value - Mean) / StdDev
        z_score = (spread_series - rolling_mean) / rolling_std
        
        return z_score