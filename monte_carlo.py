import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import time

def generate_paths(
    S: float,
    mu: float,
    dt: float,
    sigma: float,
    num_time_steps: int,
    num_samples: int,
    antithetic: bool = False,
    direct_terminal: bool = True,
    T: Optional[float] = None,  # Total time period (needed for direct_terminal)
) -> np.ndarray:
    """
    Generates simulated price paths for a geometric Brownian motion model.

    Args:
        S: Initial stock price.
        mu: Drift parameter (annualized return).
        dt: Time step size.
        sigma: Volatility parameter (annualized).
        num_time_steps: Number of time steps to simulate.
        num_samples: Number of price paths to generate.
        antithetic: Boolean flag for antithetic sampling.
        direct_terminal: Whether to use direct terminal price simulation.
        T: Total time period (required if direct_terminal=True).

    Returns:
        np.ndarray: Simulated price paths.
    """
    import numpy as np

    # Direct terminal price simulation for European options
    if direct_terminal:
        if T is None:
            raise ValueError("T must be provided when direct_terminal=True")

        # Generate random normal samples for terminal prices
        if antithetic:
            Z = np.random.normal(0, 1, size=(num_samples // 2, 1))
            Z = np.vstack([Z, -Z])  # Antithetic samples
        else:
            Z = np.random.normal(0, 1, size=(num_samples, 1))

        # Calculate terminal prices directly
        ST = S * np.exp((mu * T) + (sigma * np.sqrt(T) * Z[:, 0]))

        # Create "paths" array with just initial and terminal prices
        paths = np.zeros((Z.shape[0], 2))
        paths[:, 0] = S
        paths[:, 1] = ST

        return paths

    # Original path simulation code
    # Generate random normal samples
    if antithetic:
        Z = np.random.normal(0, 1, size=(num_samples // 2, num_time_steps))
        Z = np.vstack([Z, -Z])  # Antithetic samples
    else:
        Z = np.random.normal(0, 1, size=(num_samples, num_time_steps))

    # Initialize price paths array with starting price
    paths = np.zeros((Z.shape[0], num_time_steps + 1))
    paths[:, 0] = S

    # Generate price paths
    for t in range(1, num_time_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu * dt) + (sigma * np.sqrt(dt) * Z[:, t - 1])
        )

    return paths


def monte_carlo_price(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0,
    num_samples: int = 10000,
    random_seed: int = 42,
    antithetic: bool = False,
    control_variate: bool = False,
    num_time_steps: int = None,
    return_paths: bool = False,
    direct_terminal: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """Price a European option using Monte Carlo simulation.

    Args:
        option_type: Type of option: 'call'/'c' or 'put'/'p'.
        S: Current stock/underlying price.
        K: Strike price of the option.
        T: Time to maturity in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying (annualized).
        q: Dividend yield of the underlying (annualized).
        num_samples: Number of Monte Carlo paths to simulate.
        random_seed: Seed for random number generation.
        variance_reduction: Variance reduction technique.
        num_time_steps: Number of time steps in simulation path.
        return_paths: Whether to return the simulated price paths.
        control_variate_beta: Control variate coefficient.

    Returns:
        Dict with price and error estimates.
    """

    start_time = time.time()

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

    # Set number of time steps
    if num_time_steps is None:
        num_time_steps = max(1, int(T * 12))  # Monthly steps by default

    # Time steps size
    dt = T / num_time_steps

    # Risk-adjusted drift (Itô correction)
    mu = r - q - 0.5 * sigma**2

    # Initialize random number generator
    np.random.seed(random_seed)

    # Generate price paths
    paths = generate_paths(
            S=S,
            mu=mu,
            dt=dt,
            sigma=sigma,
            num_time_steps=num_time_steps,
            num_samples=num_samples,
            antithetic=antithetic,
            direct_terminal=direct_terminal,
            T=T if direct_terminal else None
        )

    # Get terminal stock prices
    ST = paths[:, -1]

    # Calculate option payoffs at maturity
    if option_type_str == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    # Apply control variate technique if selected
    if control_variate:
        # Calculate expected stock price at maturity
        expected_ST = S * np.exp((r - q) * T)
        
        # Calculate optimal beta (covariance between payoff and stock price divided by variance of stock price)
        cov_matrix = np.cov(payoffs, ST)
        cov_payoff_ST = cov_matrix[0, 1]
        var_ST = np.var(ST)
        optimal_beta = cov_payoff_ST / var_ST
        
        
        # Apply control variate adjustment
        # Important: both the ST and expected_ST need to be discounted
        discounted_ST = np.exp(-r * T) * ST
        discounted_expected_ST = np.exp(-r * T) * expected_ST
        
        # Apply the adjustment to already-discounted payoffs
        discounted_payoffs = np.exp(-r * T) * payoffs
        discounted_payoffs = discounted_payoffs - optimal_beta * (discounted_ST - discounted_expected_ST)
    else:
        # Discount to the present value
        discounted_payoffs = np.exp(-r * T) * payoffs

    # Calculate the option price as the average of discounted payoffs
    option_price = np.mean(discounted_payoffs)

    # Calculate standard error
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))

    # Prepare result
    result = {
        "price": option_price,
        "error": std_error,
        "computation_time": time.time() - start_time
    }

    if return_paths:
        result["paths"] = paths

    return result


def get_mc_prices(
    options_df: pd.DataFrame,
    num_samples: int =10000,
    seed: int=42,
    antithetic: bool = False,
    control_variate: bool = False,
    num_time_steps: int = None,
    return_paths: bool =False,
    direct_terminal: bool = True,  # Add this parameter
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[np.ndarray]]]:
    """
    Add Monte Carlo pricing to an options DataFrame.

    Args:
        options_df: DataFrame with option contract information.
        num_samples: Number of Monte Carlo samples.
        seed: Base random seed.
        variance_reduction: Variance reduction technique.
        num_time_steps: Number of time steps in simulation path.
        return_paths: Whether to return the simulated price paths.
        control_variate_beta: Control variate coefficient.
        direct_terminal: Whether to use direct terminal price simulation instead of path simulation.

    Returns:
        DataFrame with added mc_price and mc_error columns.
    """

    # Make a copy to avoid modifying the original DataFrame
    result_df = options_df.copy()

    # Initialize new columns
    result_df["mc_price"] = np.nan
    result_df["mc_se"] = np.nan
    result_df["mc_compute_time"] = np.nan  # Optionally track computation time

    # Store paths if requested
    if return_paths:
        paths_storage = []

    # Process each option
    for idx, option in result_df.iterrows():
        # Calculate Monte Carlo price
        idx_seed = seed + idx

        mc_result = monte_carlo_price(
            option_type=options_df["option_type"].iloc[idx],
            S=options_df["spot_price"].iloc[idx],
            K=options_df["strike_price"].iloc[idx],
            T=options_df["maturity"].iloc[idx],
            r=options_df["risk_free_rate"].iloc[idx],
            sigma=options_df["volatility"].iloc[idx],
            q=options_df["dividend_yield"].iloc[idx],
            num_samples=num_samples,
            random_seed=idx_seed,
            antithetic=antithetic,
            control_variate=control_variate,
            num_time_steps=num_time_steps,
            return_paths=return_paths,
            direct_terminal=direct_terminal  # Pass the parameter
        )

        # Update DataFrame with results
        result_df.at[idx, "mc_price"] = mc_result["price"]
        result_df.at[idx, "mc_se"] = mc_result["error"]

        # Store computation time if available
        if "computation_time" in mc_result:
            result_df.at[idx, "mc_compute_time"] = mc_result["computation_time"]

        # Store paths if requested
        if return_paths and "paths" in mc_result:
            paths_storage.append(mc_result["paths"])

    # Return both DataFrame and paths if requested
    if return_paths:
        return result_df, paths_storage
    else:
        return result_df
