import asyncio
from src.shared.state import Blackboard
from src.data_loader.stream import BinanceStream

async def main():
    # 1. Init Shared Memory
    bb = Blackboard()
    
    # 2. Init Sensor
    stream = BinanceStream(bb, "btcusdt", "ethusdt")
    
    # 3. Create Background Task for Stream
    task = asyncio.create_task(stream.connect())
    
    # 4. Create a "Watcher" loop (simulating the Math Engine)
    print("Watcher started...")
    while True:
        await asyncio.sleep(0.5) # Check every 1 second
        state = await bb.get_state()
        print(f"[BLACKBOARD] A: {state.price_a} | B: {state.price_b}")

if __name__ == "__main__":
    asyncio.run(main())