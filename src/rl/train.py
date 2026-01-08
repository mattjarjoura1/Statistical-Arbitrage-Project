import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# Ensure we can find the src folder
sys.path.append(os.getcwd())

from src.rl.gym_env import TradingEnv

def train_agent():
    # 1. Configuration
    TRAIN_FILE = "data/raw/live_session.csv"
    LOG_DIR = "logs/"
    MODEL_DIR = "models/"
    TIMESTEPS = 100_000 # How many "seconds" of trading to simulate
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 2. Setup the Environment
    # We wrap it in a 'DummyVecEnv' because SB3 expects vectorized environments
    # (This allows running 4 simulations in parallel on 4 CPU cores later if we want)
    env = DummyVecEnv([lambda: TradingEnv(TRAIN_FILE, skip_rows=100)])
    
    # 3. Normalize Rewards
    # CRITICAL: This fixes the "Balance didn't change" issue.
    # It scales the tiny $0.00001 rewards so the Neural Net can actually see them.
    # It also clips huge outliers so gradients don't explode.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 4. Define the Agent (PPO)
    model = PPO(
        "MlpPolicy",        # Standard Multi-Layer Perceptron (Simple Neural Net)
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,       # Update the brain every 2048 ticks
        batch_size=64,      # Train on chunks of 64 ticks
        gamma=0.99,         # Discount factor (Future rewards are slightly less valuable)
        tensorboard_log=LOG_DIR
    )
    
    print(f"[TRAIN] Starting PPO training on {TRAIN_FILE}...")
    print(f"[TRAIN] Target Steps: {TIMESTEPS}")
    
    # 5. The Training Loop
    try:
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted! Saving current progress...")
    
    # 6. Save the Brain
    model_path = os.path.join(MODEL_DIR, "ppo_stat_arb_v1")
    model.save(model_path)
    
    # IMPORTANT: We must also save the 'normalization' stats.
    # If we don't, the agent won't know how to scale inputs during live trading.
    env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    
    print(f"[TRAIN] Model saved to {model_path}.zip")
    print(f"[TRAIN] Normalizer saved to {MODEL_DIR}vec_normalize.pkl")
    print("Done.")

if __name__ == "__main__":
    train_agent()