import asyncio
from src.shared.state import Blackboard
from src.math.kalman import KalmanFilter
from src.math.statistics import WindowStatistics

async def run_math_engine(blackboard: Blackboard, 
                          update_event: asyncio.Event):
    """
    The Brain Loop.
    Triggered ONLY when new market data arrives.
    """
    print("[SYSTEM] Math Engine Started.")
    
    # Initialize our Math Models
    # These persist across loops (Memory)
    kalman = KalmanFilter(delta=1e-4, R=1e-3)
    stats = WindowStatistics(window_size=300) # 5 min window
    
    while True:
        # 1. THE PAUSE
        # We wait here efficiently until the WebSocket tells us to wake up.
        await update_event.wait()
        update_event.clear() # Reset the flag immediately
        
        # 2. THE READ (Atomic Snapshot)
        # We need the generic price_a and price_b
        state = await blackboard.get_state()
        
        # Safety Check: Don't run math on empty data
        if state.price_a == 0 or state.price_b == 0:
            continue
            
        # 3. THE COMPUTE (Sequential Math Chain)
        
        # A. Update Kalman -> Get Beta
        beta, spread = kalman.update(state.price_a, state.price_b)
        
        # B. Calculate Raw Spread (The Error Signal)
        # spread = Price_A - (Beta * Price_B)
        # spread = state.price_a - (beta * state.price_b)
        
        # C. Update Statistics -> Get Physics
        theta, mu, sigma = stats.update(spread)
        
        # D. Calculate Z-Score (The Trading Signal)
        # Z = (Current_Value - Mean) / Volatility
        z_score = (spread - mu) / sigma
        
        # 4. THE WRITE (Update Shared State)
        await blackboard.update_math(
            beta=beta,
            theta=theta,
            vol=sigma,    # passing sigma as 'volatility'
            spread=spread,
            z_score=z_score
        )
        
        # (Optional) Logging to prove it's alive
        # print(f"[MATH] Z: {z_score:.2f} | Beta: {beta:.4f} | Theta: {theta:.4f}")