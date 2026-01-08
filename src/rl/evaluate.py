import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure we can find the src folder
sys.path.append(os.getcwd())

from src.rl.gym_env import TradingEnv

def evaluate_agent():
    # 1. Configuration
    TEST_FILE = "data/raw/live_session.csv"
    MODEL_PATH = "models/ppo_stat_arb_v1"
    STATS_PATH = "models/vec_normalize.pkl"
    
    # 2. Recreate Environment
    # CRITICAL: Must match training config exactly
    env = DummyVecEnv([lambda: TradingEnv(TEST_FILE, skip_rows=100)])
    env = VecNormalize.load(STATS_PATH, env)
    env.training = False
    env.norm_reward = False 
    
    # 3. Load Brain
    print(f"[EVAL] Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    
    # 4. Simulation Loop
    obs = env.reset()
    done = False
    
    # Access inner env for raw data
    raw_env = env.envs[0]
    
    history = {
        'step': [],
        'price': [],
        'z_score': [],
        'action': [],
        'portfolio': [],
        'position': []
    }
    
    print("[EVAL] Running simulation...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # Log State BEFORE step
        history['step'].append(raw_env.current_step)
        history['price'].append(raw_env.prices[raw_env.current_step]) # Spread Price
        history['z_score'].append(raw_env.z_scores[raw_env.current_step])
        history['portfolio'].append(raw_env.portfolio_value)
        history['position'].append(raw_env.position) # Previous position
        history['action'].append(int(action[0]))     # Target position
        
        obs, reward, done, _ = env.step(action)

    # 5. Generate Text Report
    analyze_performance(history, raw_env)

    # 6. Plotting (Optional - keeps the visual)
    plot_results(history)

def analyze_performance(history, env_instance):
    """
    Reconstructs trades and prints a detailed forensic report.
    """
    df = pd.DataFrame(history)
    df['prev_action'] = df['action'].shift(1).fillna(0)
    
    # Detect Trade Changes
    # A "Trade" happens when the Target Action changes (e.g., 0 -> 1, or 1 -> -1)
    trades = []
    current_trade = {}
    
    # Fee estimator (approximate based on env settings)
    # We use a fixed proxy for the 'Asset Price' since we don't have it in history dict easily
    # Ideally, you'd log 'price_a' in history too, but we can estimate.
    est_fee_per_trade = 1.50 # $3000 * 0.05%
    
    for i, row in df.iterrows():
        if i == 0: continue
        
        # If Action changed, we closed the old position and/or opened a new one
        if row['action'] != row['prev_action']:
            
            # 1. Close Previous Trade (if we weren't flat)
            if current_trade:
                current_trade['exit_step'] = row['step']
                current_trade['exit_price'] = row['price']
                current_trade['duration'] = row['step'] - current_trade['entry_step']
                
                # Raw PnL = Position * (Exit - Entry)
                price_delta = row['price'] - current_trade['entry_price']
                gross_pnl = current_trade['direction'] * price_delta
                
                current_trade['gross_pnl'] = gross_pnl
                current_trade['fee'] = est_fee_per_trade # Entry fee
                current_trade['net_pnl'] = gross_pnl - (est_fee_per_trade * 2) # Entry + Exit fees
                
                trades.append(current_trade)
                current_trade = {}
            
            # 2. Open New Trade (if target is not flat)
            target = row['action']
            direction = 0
            if target == 1: direction = 1
            if target == 2: direction = -1
            
            if direction != 0:
                current_trade = {
                    'entry_step': row['step'],
                    'entry_price': row['price'],
                    'direction': direction, # 1 or -1
                    'type': "LONG" if direction == 1 else "SHORT"
                }

    # METRICS CALCULATION
    n_trades = len(trades)
    final_balance = df['portfolio'].iloc[-1]
    total_pnl = final_balance - env_instance.initial_balance
    
    print("\n" + "="*40)
    print(f"       PERFORMANCE AUTOPSY")
    print("="*40)
    print(f"Initial Balance:   ${env_instance.initial_balance:,.2f}")
    print(f"Final Balance:     ${final_balance:,.2f}")
    print(f"Total Net PnL:     ${total_pnl:,.2f}  ({(total_pnl/env_instance.initial_balance)*100:.2f}%)")
    print(f"Total Trades:      {n_trades}")
    
    if n_trades > 0:
        # Trade Stats
        wins = [t for t in trades if t['net_pnl'] > 0]
        losses = [t for t in trades if t['net_pnl'] <= 0]
        win_rate = len(wins) / n_trades * 100
        
        avg_win = np.mean([t['net_pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['net_pnl'] for t in losses]) if losses else 0
        
        fees_paid = n_trades * (est_fee_per_trade * 2)
        
        print(f"-"*40)
        print(f"Win Rate:          {win_rate:.1f}%  ({len(wins)} W / {len(losses)} L)")
        print(f"Avg Win:           ${avg_win:.2f}")
        print(f"Avg Loss:          ${avg_loss:.2f}")
        print(f"Est Fees Paid:     ${fees_paid:.2f}")
        print(f"-"*40)
        
        # Duration Stats
        avg_duration = np.mean([t['duration'] for t in trades])
        print(f"Avg Hold Time:     {avg_duration:.1f} seconds")
        print(f"Longest Trade:     {max([t['duration'] for t in trades])} seconds")
        
        # Drawdown
        equity_curve = df['portfolio']
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max)
        max_dd = drawdown.min()
        print(f"Max Drawdown:      ${max_dd:.2f}")
        
    else:
        print("\n[WARN] No trades were executed.")
        print("Possible causes:")
        print("1. Transaction Fee is too high -> Agent is scared.")
        print("2. Z-Score never crossed threshold -> Market too calm.")
        print("3. Neural Net converged to 'Always Hold' -> Needs more training.")

    print("="*40 + "\n")

def plot_results(history):
    # (Same visualization logic as before)
    spread = np.array(history['price'])
    actions = np.array(history['action'])
    portfolio = np.array(history['portfolio'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(spread, label='Spread', color='gray', alpha=0.5)
    
    # Map 0, 1, 2 actions to visual markers
    # We want to see where it was LONG (1) vs SHORT (2)
    # This is a 'State' plot, not just entry points
    is_long = (actions == 1)
    is_short = (actions == 2)
    
    ax1.fill_between(range(len(spread)), spread, min(spread), where=is_long, color='green', alpha=0.1, label='Long Zone')
    ax1.fill_between(range(len(spread)), spread, min(spread), where=is_short, color='red', alpha=0.1, label='Short Zone')
    
    ax1.set_title("Agent Behavior (Green=Long, Red=Short)")
    ax1.set_ylabel("Spread ($)")
    ax1.legend()
    
    ax2.plot(portfolio, color='blue')
    ax2.set_title("Equity Curve")
    ax2.set_ylabel("Account Balance ($)")
    
    plt.tight_layout()
    plt.savefig("evaluation_report.png")
    print("[EVAL] Visual saved to evaluation_report.png")

if __name__ == "__main__":
    evaluate_agent()