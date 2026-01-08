import numpy as np

class OUProcess:
    """
    Estimates the Ornstein-Uhlenbeck parameters of the spread
    using a rolling window of recent history.
    """
    def __init__(self, window_size=600):
        # Window size: How many data points to look back?
        # e.g., 600 points @ 1 tick/sec = 10 minutes of memory
        self.window_size = window_size
        self.history = []

    def update(self, spread_value: float):
        """
        Adds the new spread value and re-estimates Theta and Mu.
        Returns: (theta, mu, sigma)
        """
        self.history.append(spread_value)
        
        # Keep the history fixed to the window size
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        # We need enough data to run a regression
        if len(self.history) < 10:
            return 0.0, 0.0, 0.0
            
        # --- The Math (Linear Regression) ---
        # We are discretizing the OU process:
        # x(t) = a + b * x(t-1) + epsilon
        # Where theta = -ln(b) / dt
        # And mu = a / (1 - b)
        
        series = np.array(self.history)
        x_t = series[1:]      # Current values
        x_prev = series[:-1]  # Previous values
        
        # Linear Regression: Fit x_t against x_prev
        # This finds 'b' (slope) and 'a' (intercept)
        # We can use numpy's polyfit for speed (degree 1)
        b, a = np.polyfit(x_prev, x_t, 1)
        
        # Calculate OU Parameters
        # Note: We assume dt = 1 (one tick unit) for simplicity here.
        # If 'b' is >= 1, it's not mean reverting (it's trending or explosive).
        if b >= 1.0:
            theta = 0.0 # No mean reversion force
            mu = np.mean(series) # Fallback to simple mean
            sigma = np.std(series)
        else:
            theta = -np.log(b)
            mu = a / (1 - b)
            
            # Calculate Residuals (Sigma)
            residuals = x_t - (a + b * x_prev)
            sigma = np.std(residuals)
            
        return theta, mu, sigma