import asyncio
import json
import websockets
# Import the type hint for the Blackboard (assuming it is in src.shared.state)
from src.shared.state import Blackboard

class BinanceStream:
    """
    Responsibility: Connect to Binance WebSocket, filter for relevant trades, 
    and write the 'Latest Truth' to the Blackboard.
    
    This is a 'Producer' in the Producer-Consumer pattern.
    """
    def __init__(self, blackboard: Blackboard, symbol_a: str, symbol_b: str):
        self.url = "wss://stream.binance.com:9443/ws"
        self.blackboard = blackboard
        
        # Store symbols in lower case as required by Binance API (e.g. 'btcusdt')
        self.symbol_a = symbol_a.lower() 
        self.symbol_b = symbol_b.lower()
        
        # Internal Cache (Buffer)
        # We need this because WebSocket messages arrive for ONE symbol at a time.
        # We only push to the blackboard when we have a valid price for BOTH.
        self.prices = {self.symbol_a: None, self.symbol_b: None}

    async def connect(self):
        print(f"[SENSOR] Connecting to Binance Stream for {self.symbol_a} & {self.symbol_b}...")
        
        async with websockets.connect(self.url) as websocket:
            # 1. Subscribe to the streams
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{self.symbol_a}@trade", f"{self.symbol_b}@trade"],
                "id": 1
            }
            await websocket.send(json.dumps(subscribe_msg))
            print("[SENSOR] Subscribed. Streaming data to Blackboard...")

            # 2. Event Loop (Infinite)
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Filter: specific 'trade' events only
                    if 'e' in data and data['e'] == 'trade':
                        self._process_message(data)
                        
                except Exception as e:
                    print(f"[SENSOR] Error in stream: {e}")
                    # In a real production system, you would add reconnection logic here.
                    await asyncio.sleep(5) 

    def _process_message(self, data: dict):
        """
        Parses the JSON and updates the shared state.
        """
        symbol = data['s'].lower()
        price = float(data['p'])
        
        # Binance timestamps are in milliseconds, convert to seconds
        event_time = data['E'] / 1000.0 
        
        # Update Internal Buffer
        self.prices[symbol] = price
        
        # CHECK: Do we have both prices?
        if self.prices[self.symbol_a] and self.prices[self.symbol_b]:
            
            # ASYNC WRITE: Update the shared memory
            # Note: We use asyncio.create_task to fire-and-forget this update 
            # so we don't block the websocket loop waiting for the lock.
            asyncio.create_task(
                self.blackboard.update_prices(
                    price_a=self.prices[self.symbol_a],
                    price_b=self.prices[self.symbol_b],
                    timestamp=event_time
                )
            )