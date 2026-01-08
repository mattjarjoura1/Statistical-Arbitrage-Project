import asyncio
from dataclasses import dataclass
import copy

@dataclass
class MarketData:
    """
    Holds the atomic state of the market for a generic pair.
    """
    # Metadata
    timestamp: float = 0.0
    symbol_a: str = ""  # e.g. "BTCUSDT"
    symbol_b: str = ""  # e.g. "ETHUSDT"
    
    # Raw Inputs (The Sensors)
    price_a: float = 0.0
    price_b: float = 0.0
    
    # Derived State
    beta: float = 0.0
    theta: float = 0.0
    volatility: float = 0.0
    spread: float = 0.0
    z_score: float = 0.0

class Blackboard:
    """
    The Bridge. Thread-safe memory.
    """
    def __init__(self):
        self._market = MarketData()
        self._lock = asyncio.Lock()

    async def update_prices(self, price_a: float, price_b: float, timestamp: float):
        async with self._lock:
            self._market.price_a = price_a
            self._market.price_b = price_b
            self._market.timestamp = timestamp

    async def update_math(self, beta, theta, vol, spread, z_score):
        async with self._lock:
            self._market.beta = beta
            self._market.theta = theta
            self._market.volatility = vol
            self._market.spread = spread
            self._market.z_score = z_score

    async def get_state(self) -> MarketData:
        async with self._lock:
            return copy.deepcopy(self._market)