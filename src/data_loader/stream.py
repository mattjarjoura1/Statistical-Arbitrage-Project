import asyncio
import json
import websockets
from src.shared.state import Blackboard

class BinanceStream:
    def __init__(self, blackboard: Blackboard, event: asyncio.Event, symbol_a: str, symbol_b: str):
        self.update_event = event  # The Bell
        self.url = "wss://stream.binance.com:9443/ws"
        self.blackboard = blackboard
        
        self.symbol_a = symbol_a.lower() 
        self.symbol_b = symbol_b.lower()
        
        self.prices = {self.symbol_a: None, self.symbol_b: None}

    async def connect(self):
        print(f"[SENSOR] Connecting to Binance Stream for {self.symbol_a} & {self.symbol_b}...")
        
        async with websockets.connect(self.url) as websocket:
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{self.symbol_a}@trade", f"{self.symbol_b}@trade"],
                "id": 1
            }
            await websocket.send(json.dumps(subscribe_msg))
            print("[SENSOR] Subscribed. Streaming data to Blackboard...")

            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if 'e' in data and data['e'] == 'trade':
                        self._process_message(data)
                        
                except Exception as e:
                    print(f"[SENSOR] Error in stream: {e}")
                    await asyncio.sleep(5) 

    def _process_message(self, data: dict):
        """
        Synchronous parser that fires an async task.
        """
        symbol = data['s'].lower()
        price = float(data['p'])
        event_time = data['E'] / 1000.0 
        
        self.prices[symbol] = price
        
        if self.prices[self.symbol_a] and self.prices[self.symbol_b]:
            
            # FIRE AND FORGET:
            # We launch the specific sequence: Update -> Then Signal
            asyncio.create_task(
                self._update_and_signal(
                    self.prices[self.symbol_a],
                    self.prices[self.symbol_b],
                    event_time
                )
            )

    async def _update_and_signal(self, price_a, price_b, timestamp):
        """
        Atomic sequence: Write to memory first, THEN wake up the consumer.
        """
        # 1. Wait for lock and write data
        await self.blackboard.update_prices(
            price_a=price_a,
            price_b=price_b,
            timestamp=timestamp
        )
        
        # 2. Ring the bell (Wake up Math Engine)
        # We do this AFTER the await ensures the write is complete.
        self.update_event.set()