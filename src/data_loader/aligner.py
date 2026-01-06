import pandas as pd

class DataAligner:
    """
    Responsibility: Take two raw DataFrames with different timestamps and 
    align them to a single frequency for statistical analysis.
    """
    
    def __init__(self):
        pass

    def align_series(self, series_a: pd.Series, series_b: pd.Series, method='ffill'):
        """
        Aligns two series to the union of their timestamps.
        
        Args:
            series_a: Time-series data for Asset A (e.g., BTC)
            series_b: Time-series data for Asset B (e.g., ETH)
            method: 'ffill' (Forward Fill) propagates last valid observation forward.
        
        Returns:
            pd.DataFrame: A single dataframe with columns ['asset_a', 'asset_b']
                          cleaned and aligned.
        """
        # 1. Create a DataFrame with the two inputs
        # The 'outer' join creates the Union of all timestamps
        df = pd.DataFrame({series_a.name: series_a, series_b.name: series_b})
        
        # 2. Sort index to ensure time flows linearly (Critical for backtesting)
        df = df.sort_index()
        
        # 3. Handle Missing Data (Signal Conditioning)
        # If 'method' is ffill, we assume the price hasn't changed since the last tick.
        if method == 'ffill':
            df = df.ffill()
            
        # 4. Drop initial NaNs (e.g., if Asset A started trading in 2015 but B in 2017)
        df = df.dropna()
        
        return df

    def calculate_spread(self, df: pd.DataFrame, hedge_ratio=1.0):
        """
        Calculates the raw spread: Price_A - (Beta * Price_B)
        """
        df['spread'] = df['asset_a'] - (hedge_ratio * df['asset_b'])
        return df