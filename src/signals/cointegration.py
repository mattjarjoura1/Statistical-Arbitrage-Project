import statsmodels.api as sm
import pandas as pd

class CointegrationTests:
    """
    Responsibility: Research & Calibration.
    Finds the Hedge Ratio (Beta) and checks if the pair is actually mean-reverting.
    """
    
    def calculate_hedge_ratio(self, series_a: pd.Series, series_b: pd.Series):
        """
        Calculates how many units of B needed to hedge 1 unit of A.
        Uses Ordinary Least Squares (OLS).
        Formula: Price_A = Beta * Price_B + Epsilon
        """
        # Add a constant (intercept) to the independent variable (Asset B)
        X = sm.add_constant(series_b)
        model = sm.OLS(series_a, X).fit()
        
        # The slope coefficient is our hedge ratio
        beta = model.params.iloc[1] 
        return beta