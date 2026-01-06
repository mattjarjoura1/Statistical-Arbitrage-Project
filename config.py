# config.py

# 1. The Assets to Trade (Yahoo Finance Tickers)
PAIRS = [
    {
        'id': 'BTC_ETH_ARB',      # Internal name
        'asset_a': 'BTC-USD',     # Ticker A (The "Lead")
        'asset_b': 'ETH-USD',     # Ticker B (The "Lag")
        'hedge_ratio': 1.0,       # Initial guess (will be dynamic later)
    }
]

# 2. Data Settings
START_DATE = "2024-06-01"
END_DATE = "2025-01-01"
INTERVAL = "1h"  # "1d" for daily, "1h" for hourly

# 3. Strategy Parameters
Z_SCORE_WINDOW = 30   # Lookback period for moving average
ENTRY_THRESHOLD = 2.0 # Enter trade when Z-score > 2
EXIT_THRESHOLD = 0.0  # Exit trade when Z-score returns to 0