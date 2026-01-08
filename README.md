# Crypto Statistical Arbitrage Bot (Async/HFT Architecture)

## Intent
This project is an exploration into building a high-frequency trading system from first principles. The goal wasn't just to "make a trading bot," but to engineer a robust, event-driven chassis that can handle real-time market data without blocking or lagging.

I wanted to move away from basic sequential scripts to a proper asynchronous architecture (Producer-Consumer pattern) that mimics how professional systems handle data ingestion and state estimation.

## Current State (The "Chassis")
We currently have a fully functional "Perception Layer" running in real-time.

* **Ingestion Engine:** A websocket client that connects to Binance, filters for trade events, and writes to a thread-safe shared memory block (`Blackboard`). It uses `asyncio` to ensure we never drop a packet while waiting for locks.
* **Math Engine:** A dedicated processor that calculates the hidden state of the market on every tick.
    * **Kalman Filter:** Estimates the dynamic Beta (Hedge Ratio) between two assets (e.g., ETH/BTC). It adapts to regime changes automatically.
    * **Ornstein-Uhlenbeck Process:** Models the spread as a mean-reverting rubber band. It calculates Theta (mean reversion speed) and Sigma (volatility).
    * **Z-Score Normalization:** Outputs a standardized signal that indicates how statistically "stretched" the spread is.
* **Architecture:** The system is split into independent async tasks (Ingest, Math, Monitor) that communicate via a locked State Object. This prevents the heavy math from slowing down the data feed.

## Next Steps

### 1. The Decision Brain (RL Agent)
The chassis is ready, but there is no driver yet. The next major step is implementing a Reinforcement Learning agent (PPO) that:
* Observes the State Vector (Z-Score, Theta, Beta, Volatility).
* Learns to make trade decisions (Buy, Sell, Hold) based on probability rather than hard-coded `if` statements.
* Manages inventory risk (e.g., learning not to buy if we already hold too much).

### 2. Execution Layer
Currently, we just log the signals. We need to build the `ExecutionEngine` to actually send orders to the exchange, handling slippage and fee calculations.

### 3. Future Improvements
* **Multi-Asset Correlations:** Move beyond single pairs to track a basket of assets (e.g., "L1s" vs "L2s").
* **Regime Filtering:** Add a global volatility filter (like VIX or BTC Volatility) to shut down trading during crashes.
* **Backtesting Pipeline:** Build a replay system to train the RL agent on historical data before letting it run live.