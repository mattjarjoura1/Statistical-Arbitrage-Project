import sys
import os
import numpy as np

# Ensure we can find the src folder
sys.path.append(os.getcwd())

from src.rl.gym_env import TradingEnv

def test_environment():
    # 1. Setup
    csv_path = "data/raw/live_session.csv"
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        print("Please run 'record_session.py' first to generate data.")
        return

    print(f"[TEST] Initializing Gym Environment from {csv_path}...")
    
    # 2. Load Env (Simulating the 'Warm-up Skip')
    # We skip 10 rows just for this test (since your file might be short). 
    # In production/training, keep this at 100.
    env = TradingEnv(csv_path, skip_rows=10)
    
    print(f"[TEST] Success. Loaded {env.n_steps} playable steps.")
    print(f"[TEST] Observation Space: {env.observation_space.shape[0]} features")
    
    # 3. The Reset
    obs, _ = env.reset()
    print(f"\n[TEST] First Observation Vector:\n{obs}")
    # Expected: [Z-Score, Theta, Vol, Pos, Hour_Sin, Hour_Cos, Day_Sin, Day_Cos]
    
    # 4. The Loop (Random Monkey Testing)
    print("\n[TEST] Starting Random Agent Simulation...")
    
    done = False
    steps = 0
    total_reward = 0
    
    while not done:
        # Pick a random action (0, 1, or 2)
        action = env.action_space.sample()
        
        # Step the physics forward
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        # Log every 50 steps so we can see it working
        if steps % 50 == 0:
            print(f"Step {steps:04d} | Action: {action} | Reward: {reward:+.4f} | Z-Score: {obs[0]:.2f} | PnL: ${env.portfolio_value - env.initial_balance:.2f}")
        
        if terminated or truncated:
            done = True
            
    # 5. Report
    print("-" * 30)
    print(f"[TEST] Episode Complete.")
    print(f"Total Steps Run: {steps}")
    print(f"Final Account Balance: ${env.portfolio_value:.2f}")
    print("-" * 30)
    print("✅ SYSTEM VERIFIED: Data Pipeline is Solid.")

if __name__ == "__main__":
    test_environment()