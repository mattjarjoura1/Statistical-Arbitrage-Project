import numpy as np
from collections import deque

class WindowStatistics:
    """
    Tracks rolling statistics (Mean, Std Dev) and OU Parameters.
    """
    def __init__(self, window_size=600):
        # 600 ticks approx 10 mins at 1 tick/sec
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, value: float):
        """
        Ingests a new spread value.
        Returns: (Theta, Mu, Sigma)
        """
        self.history.append(value)
        
        # Need minimum data to calculate stats
        if len(self.history) < 20:
            # Return 'Safe' defaults until we have data
            return 0.0, value, 1.0 

        # Convert to numpy for fast math
        series = np.array(self.history)
        
        # 1. Calculate Volatility (Sigma)
        # We use the standard deviation of the spread itself
        sigma = np.std(series)
        
        # 2. Calculate OU Parameters (Theta & Mu)
        # Fit: x_t = a + b * x_{t-1}
        x_t = series[1:]
        x_prev = series[:-1]
        
        # Fast Linear Regression
        # Polyfit returns [slope, intercept]
        b, a = np.polyfit(x_prev, x_t, 1)
        
        # Extract Physics
        if b >= 1.0:
            # Non-Stationary (Explosive or Random Walk)
            theta = 0.0
            mu = np.mean(series) # Fallback to simple mean
        else:
            # Mean Reverting
            # Theta = -ln(slope)
            theta = -np.log(b)
            # Mu = intercept / (1 - slope)
            mu = a / (1 - b)
            
        # Safety clamp for Sigma (prevent divide by zero in Z-Score)
        if sigma < 1e-6:
            sigma = 1.0
            
        return theta, mu, sigma