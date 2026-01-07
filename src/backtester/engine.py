import pandas as pd
import numpy as np

class BacktestEngine:
    """
    Responsibility: The 'Plant Model'. 
    Simulates the evolution of Cash and Inventory based on signals.
    """
    
    def __init__(self, initial_cash=10000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0  # +1 (Long Spread), -1 (Short Spread), 0 (Flat)
        
        # We track the "Equity Curve" (Portfolio Value over time)
        self.portfolio_history = [] 

    def run_backtest(self, df: pd.DataFrame):
        """
        Iterates through the DataFrame (Time) and executes trades.
        Assumptions:
        - We trade '1 Unit' of the Spread.
        - No transaction costs (yet).
        - Execution happens at the CLOSE price (Simplified for V1).
        """
        print("[SIMULATION] Starting Backtest...")
        
        for i in range(len(df)):
            row = df.iloc[i]
            signal = row['signal']  # We need to generate this column first
            spread_price = row['spread']
            
            # --- EXECUTION LOGIC (The Actuator) ---
            
            # Signal says BUY and we are not Long
            if signal == 1 and self.position != 1:
                # If we were Short (-1), we Buy to Close (0) then Buy to Open (+1)
                # Net effect: Buy 2 units if Short, Buy 1 unit if Flat
                trades_needed = 1 - self.position
                self.cash -= (trades_needed * spread_price)
                self.position = 1
                # print(f"[{row.name.date()}] BUY SPREAD @ {spread_price:.2f}")

            # Signal says SELL and we are not Short
            elif signal == -1 and self.position != -1:
                trades_needed = self.position - (-1) # e.g. 1 - (-1) = 2
                self.cash += (trades_needed * spread_price)
                self.position = -1
                # print(f"[{row.name.date()}] SELL SPREAD @ {spread_price:.2f}")

            # Signal says EXIT (0)
            elif signal == 0 and self.position != 0:
                # Close whatever position we have
                if self.position == 1: # Sell to close
                    self.cash += spread_price
                elif self.position == -1: # Buy to close
                    self.cash -= spread_price
                self.position = 0
                # print(f"[{row.name.date()}] CLOSE POSITION @ {spread_price:.2f}")

            # --- MARK TO MARKET (Portfolio Value) ---
            # Value = Cash + (Inventory * Current Price)
            current_equity = self.cash + (self.position * spread_price)
            self.portfolio_history.append(current_equity)
            
        # Attach history to DF for plotting
        df['portfolio_value'] = self.portfolio_history
        return df