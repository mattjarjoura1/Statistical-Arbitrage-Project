from src.data_loader.connector import YahooConnector
from src.data_loader.aligner import DataAligner
import config

def run_pipeline():
    # 1. Initialize Objects
    connector = YahooConnector()
    aligner = DataAligner()
    
    # 2. Load Mission Parameters
    pair_config = config.PAIRS[0] # Get the first pair (BTC vs ETH)
    print(f"--- Starting Analysis for {pair_config['id']} ---")

    # 3. FETCH (The Sensors)
    # We fetch them separately to simulate independent data sources
    series_a = connector.fetch_ticker(
        pair_config['asset_a'], config.START_DATE, config.END_DATE, config.INTERVAL
    )
    series_b = connector.fetch_ticker(
        pair_config['asset_b'], config.START_DATE, config.END_DATE, config.INTERVAL
    )
    
    # 4. ALIGN (The Fusion)
    # This fixes the timestamps and joins them into one DataFrame
    df_aligned = aligner.align_series(series_a, series_b)
    
    # 5. VERIFY (Print first 5 rows)
    print("\n[SYSTEM] Data Alignment Complete. Preview:")
    print(df_aligned.head())
    print(f"[SYSTEM] Total aligned rows: {len(df_aligned)}")
    
def testing():
    
    connector = YahooConnector()
    pair_config = config.PAIRS[0] # Get the first pair (BTC vs ETH)
    print(f"--- Starting Analysis for {pair_config['id']} ---")

    # 3. FETCH (The Sensors)
    # We fetch them separately to simulate independent data sources
    series_a = connector.fetch_ticker(
        pair_config['asset_a'], config.START_DATE, config.END_DATE, config.INTERVAL
    )


if __name__ == "__main__":
    run_pipeline()
    # testing()
