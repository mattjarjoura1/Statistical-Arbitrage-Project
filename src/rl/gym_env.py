import gymnasium as gym
import numpy as np
import pandas as pd
from datetime import datetime, timezone

class TradingEnv(gym.Env):
    """
    A custom OpenAI Gym environment for Statistical Arbitrage.
    Simulates a 'Game' where the agent trades the spread based on math features.
    """
    
    def __init__(self, csv_path, initial_balance=10000.0, transaction_fee=0.0005, skip_rows=100):
        super(TradingEnv, self).__init__()
        
        # --- 1. CONFIGURATION ---
        self.initial_balance = initial_balance
        # self.fee = transaction_fee
        #just for testing, set fee to 0
        self.fee = 0.00
        
        # --- 2. LOAD DATA ---
        self.raw_data = pd.read_csv(csv_path)
        
        # --- SANITIZATION (The Fix) ---
        # 1. Replace "Infinite" values with NaN
        self.raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 2. Forward Fill (If a value is missing, use the previous second's value)
        self.raw_data.ffill(inplace=True)
        
        # 3. Drop any remaining NaNs (e.g., if the very first row is bad)
        self.raw_data.dropna(inplace=True)
        
        # 4. Clean Garbage (Warm-up Phase)
        if len(self.raw_data) > skip_rows:
            self.raw_data = self.raw_data.iloc[skip_rows:].reset_index(drop=True)
            
        # Safety Check: If data is empty after cleaning, stop immediately
        if len(self.raw_data) < 10:
            raise ValueError("Data is empty or too short after cleaning NaNs. Check your CSV!")

        # --- 3. FEATURE ENGINEERING (Vectorized) ---
        # (This part remains the same)
        timestamps = pd.to_datetime(self.raw_data['timestamp'], unit='s', utc=True)
        
        self.hour_sin = np.sin(2 * np.pi * timestamps.dt.hour / 24.0).values
        self.hour_cos = np.cos(2 * np.pi * timestamps.dt.hour / 24.0).values
        self.day_sin = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7.0).values
        self.day_cos = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7.0).values
        
        self.z_scores = self.raw_data['z_score'].values
        self.spreads = self.raw_data['spread'].values
        self.thetas = self.raw_data['theta'].values
        self.vols = self.raw_data['volatility'].values
        self.prices = self.raw_data['spread'].values
        
        self.n_steps = len(self.raw_data)
        
        # --- 4. SPACES ---
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        self.current_step = 0
        self.position = 0
        self.cash = self.initial_balance
        self.portfolio_value = self.initial_balance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset State
        self.current_step = 0
        self.position = 0
        self.cash = self.initial_balance
        self.portfolio_value = self.initial_balance
        
        return self._get_observation(), {}

    def step(self, action):
        """
        The Core Loop: 
        1. Execute Action
        2. Calculate PnL
        3. Advance Time
        """
        prev_portfolio_value = self.portfolio_value
        current_price = self.prices[self.current_step]
        
        # --- 1. EXECUTE ACTION ---
        # Map Discrete Actions (0,1,2) to Target Position (0, 1, -1)
        target_position = 0
        if action == 1: target_position = 1  # Long
        if action == 2: target_position = -1 # Short
        
        # Did we change position? (Trades incur fees)
        if target_position != self.position:
            # Fee is calculated on the NOTIONAL value of the trade.
            # Simplified: We assume 1 Unit of Spread ~ Spread Price (Rough approximation)
            # A rigorous version would track Price_A + Beta*Price_B separately.
            cost = abs(current_price) * self.fee 
            self.cash -= cost
        
        self.position = target_position
        
        # --- 2. ADVANCE TIME ---
        self.current_step += 1
        terminated = (self.current_step >= self.n_steps - 1)
        
        if terminated:
            return self._get_observation(), 0, True, False, {}
            
        # --- 3. CALCULATE REWARD ---
        # Mark to Market PnL
        new_price = self.prices[self.current_step]
        
        # Value = Cash + (Position * Current_Spread_Price)
        # Note: This PnL logic mimics "Trading the Spread" directly.
        self.portfolio_value = self.cash + (self.position * new_price)
        
        # Reward = Change in Portfolio Value
        step_reward = self.portfolio_value - prev_portfolio_value
        
        # Optional: Add small penalty for holding risk?
        # step_reward -= 0.01 if self.position != 0 else 0
        
        return self._get_observation(), step_reward, terminated, False, {}

    def _get_observation(self):
        """
        Constructs the observation vector efficiently using the pre-calculated arrays.
        """
        # Look up values from Numpy arrays (Fast)
        idx = self.current_step
        
        obs = np.array([
            self.z_scores[idx],
            self.thetas[idx],
            self.vols[idx],
            float(self.position), # Current inventory
            self.hour_sin[idx],
            self.hour_cos[idx],
            self.day_sin[idx],
            self.day_cos[idx]
        ], dtype=np.float32)
        
        return obs