import yfinance as yf
import pandas as pd

class YahooConnector:
    """
    Responsibility: Interface with external API (Yahoo Finance) to fetch raw market data.
    Input: Ticker string, start_date, end_date
    Output: pd.Series of the 'Close' price
    """
    def __init__(self):
        pass

    def fetch_ticker(self, ticker, start_date, end_date, interval="1d"):
        print(f"[SENSOR] Fetching data for: {ticker}...")
        
        # 1. Hit the API
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        # 2. Basic Validation (Did we get data?)
        if data.empty:
            raise ValueError(f"No data found for {ticker}. Check ticker or internet connection.")
            
        # 3. Return only the signal we care about (The Close Price)
        # Note: yfinance returns a DataFrame, we extract the Series
        return data['Close'].squeeze()  # Converts single-column DataFrame to Series