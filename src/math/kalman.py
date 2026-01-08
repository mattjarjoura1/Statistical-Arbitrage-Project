import numpy as np

class KalmanFilter:
    """
    An Online Kalman Filter estimates the hidden state (Slope and Intercept) 
    of a linear relationship between two assets.
    
    State State (x): [Beta, Alpha]^T
    Measurement (z): Price of Asset A (The 'dependent' variable)
    Input (H):       Price of Asset B (The 'independent' variable)
    """
    def __init__(self, delta=1e-4, R=1e-3):
        # 1. The State Vector [Beta, Alpha]
        # Initial guess: Beta=0, Alpha=0
        self.state = np.zeros(2) 
        
        # 2. The Covariance Matrix (P)
        # Represents our uncertainty. We start with high uncertainty (Identity matrix).
        self.P = np.eye(2)
        
        # 3. Process Noise Covariance (Q)
        # How much we expect the Beta/Alpha to drift over time.
        # Tunable: Higher 'delta' = Model adapts faster (less inertia).
        self.Q = np.eye(2) * delta
        
        # 4. Measurement Noise Covariance (R)
        # How much noise is in the raw price data.
        # Tunable: Higher 'R' = Model ignores spikes (more inertia).
        self.R = R

    def update(self, price_a: float, price_b: float):
        """
        Performs one Predict-Correct cycle.
        
        Args:
            price_a: The target asset (e.g., ETH) - y
            price_b: The reference asset (e.g., BTC) - x
        
        Returns:
            beta: The estimated hedge ratio (slope)
        """
        # --- STEP 1: PREDICT (Time Update) ---
        # "My best guess for now is the same as before, but I'm a little less certain."
        
        # State prediction: x(t|t-1) = x(t-1) (Random Walk assumption)
        # So self.state remains unchanged here.
        
        # Covariance prediction: P(t|t-1) = P(t-1) + Q
        self.P = self.P + self.Q

        # --- STEP 2: UPDATE (Measurement Update) ---
        # "Compare prediction to reality and correct the estimate."
        
        # Observation Matrix H = [price_b, 1]
        # We try to explain Price A using Price B.
        H = np.array([price_b, 1.0])
        
        # Calculate the Error (Innovation)
        # y = H * x
        expected_price_a = np.dot(H, self.state)
        error = price_a - expected_price_a
        
    
        # Calculate System Uncertainty (S)
        # S = H * P * H_transpose + R
        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        
        # Calculate Kalman Gain (K)
        # K = P * H_transpose * inv(S)
        # "How much should I trust this error?"
        K = np.dot(self.P, H.T) / S
        
        # Update State Estimate
        # x_new = x_old + K * error
        self.state = self.state + (K * error)
        
        # Update Uncertainty (P)
        # P_new = (I - K * H) * P_old
        # "I am now more certain because I have processed a new data point."
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        
        # Return just the Beta (the first element of the state)
        return self.state[0], error