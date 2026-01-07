import asyncio
import json
import websockets
from datetime import datetime

class BinanceStream:
    def __init__(self):
        self.url = "wss://stream.binance.com:9443/ws"
        self.prices = {"BTCUSDT": None, "ETHUSDT": None}
        self.hedge_ratio = 14.5  # Hardcoded 'Vibe' Beta for tonight

    async def connect(self):
        print(f"[SYSTEM] Connecting to {self.url}...")
        async with websockets.connect(self.url) as websocket:
            # 1. Subscribe to streams
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": ["btcusdt@trade", "ethusdt@trade"],
                "id": 1
            }
            await websocket.send(json.dumps(subscribe_msg))
            print("[SYSTEM] Subscribed to BTC & ETH feeds.")

            # 2. Event Loop
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                # Check if it's a trade message (ignore heartbeats)
                if 'e' in data and data['e'] == 'trade':
                    symbol = data['s']
                    price = float(data['p'])
                    
                    # Update Cache
                    self.prices[symbol] = price
                    
                    # 3. The "Alignment" Check
                    if self.prices["BTCUSDT"] and self.prices["ETHUSDT"]:
                        self.process_tick()

    def process_tick(self):
        btc = self.prices["BTCUSDT"]
        eth = self.prices["ETHUSDT"]
        
        # Calculate the 'Live' Spread
        # Spread = BTC - (Beta * ETH)
        spread = btc - (self.hedge_ratio * eth)
        
        # Get timestamp
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Print the 'Heartbeat' of the market
        print(f"[{now}] BTC: {btc:.2f} | ETH: {eth:.2f} | SPREAD: {spread:.2f}")

# Boilerplate to run async code
if __name__ == "__main__":
    stream = BinanceStream()
    try:
        asyncio.run(stream.connect())
    except KeyboardInterrupt:
        print("\n[SYSTEM] Disconnected.")