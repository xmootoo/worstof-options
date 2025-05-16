import pandas as pd

def get_bs_prices(options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Black-Scholes-Merton pricing to an options DataFrame.
    
    Args:
        options_df: DataFrame with option contract information.
                   Must contain columns: option_type, spot_price, strike_price,
                   maturity, risk_free_rate, volatility, and dividend_yield.
    
    Returns:
        DataFrame with added bs_price column and optionally Greek columns.
    """
    import pandas as pd
    import numpy as np
    
    # Make a copy to avoid modifying the original DataFrame
    result_df = options_df.copy()
    
    # Initialize new columns
    result_df["bs_price"] = np.nan
    
    
    # Process each option
    for idx, option in result_df.iterrows():
        # Extract option parameters
        option_type = option["option_type"]
        S = option["spot_price"]
        K = option["strike_price"]
        T = option["maturity"]
        r = option["risk_free_rate"]
        sigma = option["volatility"]
        q = option["dividend_yield"]
        
        # Just get the price
        price = black_scholes_price(
            option_type=option_type,
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            q=q
        )
        
        # Store price
        result_df.at[idx, "bs_price"] = price

    return result_df

def black_scholes_price(
    option_type: str, 
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    q: float = 0
) -> float:
    """
    Calculate the Black-Scholes-Merton price for European options.
    
    Args:
        option_type: Type of option - 'call'/'c' or 'put'/'p'.
        S: Current stock/underlying price.
        K: Strike price of the option.
        T: Time to maturity in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying (annualized).
        q: Dividend yield of the underlying (annualized), defaults to 0.
        
    Returns:
        float: The option price according to Black-Scholes-Merton formula.
        
    Notes:
        This function implements the analytical solution to the Black-Scholes-Merton
        equation for European options without external libraries like QuantLib.
        
        The formula for a call option is:
        C = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
        
        The formula for a put option is:
        P = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)
        
        where:
        d1 = [ln(S/K) + (r - q + sigma²/2)*T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        N() is the cumulative distribution function of the standard normal distribution
    """
    import numpy as np
    from scipy.stats import norm
    
    # Handle edge cases
    if T <= 0:
        # For expired options
        if option_type.lower() in ["call", "c"]:
            return max(0, S - K)
        else:
            return max(0, K - S)
    
    # Validate option type
    if isinstance(option_type, str):
        option_type_str = option_type.lower()
    else:
        option_type_str = str(option_type).lower()
        
    if option_type_str in ["call", "c"]:
        option_type_str = "call"
    elif option_type_str in ["put", "p"]:
        option_type_str = "put"
    else:
        raise ValueError("option_type must be 'call'/'c' or 'put'/'p'")
    
    # Calculate d1 and d2
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price
    if option_type_str == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price