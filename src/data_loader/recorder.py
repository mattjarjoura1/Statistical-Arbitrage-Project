import asyncio
import csv
import os
from src.shared.state import Blackboard

class DataRecorder:
    """
    Responsibility: Sample the Blackboard every 1 second and dump the 
    current 'Truth' to a CSV file.
    """
    def __init__(self, blackboard: Blackboard, filename="data/raw/live_session.csv"):
        self.blackboard = blackboard
        self.filename = filename
        
        # Ensure the folder exists (e.g., data/raw/)
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        # Define the columns we want to train on
        self.headers = [
            "timestamp", 
            "price_a",    # ETH (Dependent)
            "price_b",    # BTC (Independent)
            "beta",       # Hedge Ratio
            "theta",      # Mean Reversion Speed
            "volatility", # Spread Volatility
            "spread",     # The Raw Error ($)
            "z_score"     # The Signal
        ]
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            print(f"[RECORDER] Created new file: {self.filename}")

    async def run(self):
        print(f"[RECORDER] Started. Dumping state every 1.0s to {self.filename}...")
        
        while True:
            # 1. The Clock: Wait exactly 1 second
            await asyncio.sleep(1.0)
            
            # 2. The Read: Get atomic snapshot from Blackboard
            state = await self.blackboard.get_state()
            
            # 3. The Filter: Don't record empty zeros (waiting for first tick)
            if state.price_a == 0 or state.timestamp == 0:
                continue
                
            # 4. The Write: Append to CSV
            self._write_row(state)

    def _write_row(self, state):
        try:
            with open(self.filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    state.timestamp,
                    state.price_a,
                    state.price_b,
                    state.beta,
                    state.theta,
                    state.volatility,
                    state.spread,
                    state.z_score
                ])
        except Exception as e:
            print(f"[RECORDER] Error writing to CSV: {e}")