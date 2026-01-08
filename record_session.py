import asyncio
import os
import sys

# Ensure python can find your src modules
sys.path.append(os.getcwd())

from src.shared.state import Blackboard
from src.data_loader.stream import BinanceStream
from src.processors.math_engine import run_math_engine
from src.data_loader.recorder import DataRecorder

async def monitor_loop(blackboard: Blackboard):
    """
    Optional: Keeps the visual dashboard running so you know it's alive.
    """
    while True:
        await asyncio.sleep(1)
        state = await blackboard.get_state()
        if state.price_a != 0:
            print(f"[SYSTEM] Z-Score: {state.z_score:.4f} | Recording to CSV...")

async def main():
    print("--- STARTING DATA RECORDING SESSION ---")
    
    # 1. Init Shared Memory
    bb = Blackboard()
    update_event = asyncio.Event() 
    
    # 2. Init Components
    # Stream: Connects to Binance (ETH/BTC)
    stream = BinanceStream(bb, update_event, "ethusdt", "btcusdt")
    
    # Recorder: Saves to 'data/raw/live_session.csv'
    recorder = DataRecorder(bb, filename="data/raw/live_session.csv")
    
    # 3. Create Tasks
    task_stream = asyncio.create_task(stream.connect())
    task_math = asyncio.create_task(run_math_engine(bb, update_event))
    task_recorder = asyncio.create_task(recorder.run())
    task_monitor = asyncio.create_task(monitor_loop(bb))
    
    # 4. Run Forever
    await asyncio.gather(task_stream, task_math, task_recorder, task_monitor)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SYSTEM] Recording Stopped. Check data/raw/live_session.csv")