import asyncio
import os
import sys

# Ensure python can find your src modules
sys.path.append(os.getcwd())

from src.shared.state import Blackboard
from src.data_loader.stream import BinanceStream
from src.processors.math_engine import run_math_engine

async def monitor_loop(blackboard: Blackboard):
    """
    The 'Cockpit View'. 
    This is just for YOU to see what is happening in the terminal.
    It does not affect the trading logic.
    """
    print("[SYSTEM] Monitor started. Waiting for data...")
    
    while True:
        # Cadence: Update screen every 0.5s
        await asyncio.sleep(0.5)
        
        # Get the latest "Truth"
        state = await blackboard.get_state()
        
        # Only print if we actually have data
        if state.price_a == 0:
            continue
            
        # Clear line (optional, makes it look like a dashboard)
        # print("\033[H\033[J", end="") 
        
        print(f"""
        --- SYSTEM STATUS ---
        [RAW]
        BTC: ${state.price_b:,.2f}
        ETH: ${state.price_a:,.2f}
        
        [FILTER]
        Beta (Ratio): {state.beta:.4f}
        Spread ($):   ${state.spread:.2f}
        
        [PHYSICS]
        Theta (Rev):  {state.theta:.5f}
        Vol (Sigma):  {state.volatility:.4f}
        
        [SIGNAL]
        Z-SCORE:      {state.z_score:.4f}
        ---------------------
        """)

async def main():
    # 1. Init Shared Resources
    bb = Blackboard()
    update_event = asyncio.Event() # The "Bell"
    
    # 2. Init Components
    # Note: We pass the event to BOTH so they can talk
    stream = BinanceStream(bb, update_event, "ethusdt", "btcusdt")
    
    # 3. Launch Tasks
    # Task A: Ingestion (Network Bound)
    task_stream = asyncio.create_task(stream.connect())
    
    # Task B: Math (CPU Bound - Event Driven)
    task_math = asyncio.create_task(run_math_engine(bb, update_event))
    
    # Task C: Monitor (Terminal Output)
    task_monitor = asyncio.create_task(monitor_loop(bb))
    
    # 4. Keep them running forever
    await asyncio.gather(task_stream, task_math, task_monitor)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down...")