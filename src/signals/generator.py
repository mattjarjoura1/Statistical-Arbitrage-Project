import pandas as pd
import numpy as np

class SignalGenerator:
    """
    Responsibility: The Controller Logic.
    Input: Z-Score Series
    Output: A Series of Integers (-1, 0, 1) indicating target position.
    """
    
    def __init__(self, entry_threshold=2.0, exit_threshold=0.0):
        self.entry = entry_threshold
        self.exit = exit_threshold

    def generate_signals(self, z_scores: pd.Series):
        """
        Logic:
        - Short the Spread if Z > 2.0 (Expect reversion down)
        - Long the Spread if Z < -2.0 (Expect reversion up)
        - Exit if Z crosses 0.0
        """
        signals = pd.Series(index=z_scores.index, data=0)
        position = 0 # 0: Flat, 1: Long, -1: Short
        
        # Iterate through time (We do this loop to maintain 'State')
        # Note: Vectorizing this is harder because position depends on previous state
        for i in range(len(z_scores)):
            z = z_scores.iloc[i]
            
            # Check if Z is NaN (first 30 days)
            if np.isnan(z): 
                continue

            # State Machine Logic
            if position == 0:
                # ENTRY LOGIC
                if z > self.entry:
                    position = -1 # Sell Spread
                elif z < -self.entry:
                    position = 1  # Buy Spread
            
            elif position == 1:
                # EXIT LOGIC FOR LONG
                if z >= -self.exit:
                    position = 0
            
            elif position == -1:
                # EXIT LOGIC FOR SHORT
                if z <= self.exit:
                    position = 0
            
            signals.iloc[i] = position
            
        return signals
    