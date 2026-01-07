import pandas as pd
import matplotlib.pyplot as plt

# Import our custom modules
from src.data_loader.connector import YahooConnector
from src.data_loader.aligner import DataAligner
from src.signals.cointegration import CointegrationTests
from src.signals.zscore import ZScoreGenerator
from src.signals.generator import SignalGenerator
from src.backtester.engine import BacktestEngine
import config

def run_system():
    print("--- 1. INITIALIZATION ---")
    connector = YahooConnector()
    aligner = DataAligner()
    coint_engine = CointegrationTests()
    
    # Configuration for the "Experiment"
    pair = config.PAIRS[0]
    train_start = "2023-01-01"
    train_end   = "2023-06-01" # Calibrate on first half
    test_start  = "2023-06-02"
    test_end    = "2024-01-01" # Trade on second half (Out-of-Sample)

    print(f"[SYSTEM] Asset A: {pair['asset_a']}")
    print(f"[SYSTEM] Asset B: {pair['asset_b']}")

    # --- 2. DATA INGESTION (The Sensors) ---
    print("\n--- 2. DATA INGESTION ---")
    # Fetch ample data to cover both periods
    raw_a = connector.fetch_ticker(pair['asset_a'], train_start, test_end)
    raw_b = connector.fetch_ticker(pair['asset_b'], train_start, test_end)
    
    # Align timestamps
    df_aligned = aligner.align_series(raw_a, raw_b)
    print(f"[DATA] Aligned {len(df_aligned)} rows.")

    # --- 3. RESEARCH PHASE (The Calibration) ---
    print("\n--- 3. CALIBRATION (In-Sample) ---")
    # Slice the data to 'Training' period only
    df_train = df_aligned.loc[train_start:train_end]
    
    # Calculate Beta on Training Data ONLY
    beta = coint_engine.calculate_hedge_ratio(df_train['asset_a'], df_train['asset_b'])
    print(f"[MATH] Calibrated Hedge Ratio (Beta): {beta:.4f} (using data up to {train_end})")

    # --- 4. EXECUTION PHASE (The Live Simulation) ---
    print("\n--- 4. SIMULATION (Out-of-Sample) ---")
    # Slice data to 'Testing' period
    df_test = df_aligned.loc[test_start:test_end].copy()
    
    # A. Calculate Spread using the TRAINING Beta (No cheating!)
    df_test['spread'] = df_test['asset_a'] - (beta * df_test['asset_b'])
    
    # B. Signal Conditioning (Z-Score)
    z_gen = ZScoreGenerator(window=30)
    df_test['z_score'] = z_gen.compute(df_test['spread'])
    
    # C. Controller Logic (Generate Signals)
    sig_gen = SignalGenerator(entry_threshold=2.0, exit_threshold=0.0)
    df_test['signal'] = sig_gen.generate_signals(df_test['z_score'])
    
    # D. Actuator (Backtest Engine)
    engine = BacktestEngine(initial_cash=10_000)
    df_results = engine.run_backtest(df_test)
    
    # --- 5. REPORTING ---
    final_equity = df_results['portfolio_value'].iloc[-1]
    pnl = final_equity - 10000
    return_pct = (pnl / 10000) * 100
    
    print("\n--- 5. PERFORMANCE REPORT ---")
    print(f"Trading Period: {test_start} to {test_end}")
    print(f"Final Equity:   ${final_equity:,.2f}")
    print(f"Net Profit:     ${pnl:,.2f}")
    print(f"Return:         {return_pct:.2f}%")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: The Spread and Z-Score Thresholds
    color = 'tab:blue'
    ax1.set_title(f"Spread Dynamics (Beta={beta:.2f})")
    ax1.set_ylabel('Spread Price', color=color)
    ax1.plot(df_results.index, df_results['spread'], color=color, alpha=0.6)
    
    # Plot 2: Portfolio Value (The Equity Curve)
    ax2.set_title("Portfolio Equity Curve")
    ax2.set_ylabel("Account Value ($)")
    ax2.plot(df_results.index, df_results['portfolio_value'], color='green', linewidth=1.5)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_system()