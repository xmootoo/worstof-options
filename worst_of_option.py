import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union
import time

def generate_correlated_paths(
    S1: float,
    S2: float,
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    T: float,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates correlated terminal asset values using Cholesky decomposition.

    Args:
        S1: Initial price of asset 1.
        S2: Initial price of asset 2.
        mu1: Drift parameter for asset 1 (annualized return).
        mu2: Drift parameter for asset 2 (annualized return).
        sigma1: Volatility parameter for asset 1 (annualized).
        sigma2: Volatility parameter for asset 2 (annualized).
        rho: Correlation coefficient between the assets.
        T: Time to maturity in years.
        num_samples: Number of price paths to generate.

    Returns:
        Tuple of arrays containing the terminal values for both assets.
    """
    # Generate uncorrelated standard normal random variables
    X1 = np.random.normal(0, 1, size=num_samples)
    X2 = np.random.normal(0, 1, size=num_samples)

    # Generate correlated standard normal random variables using Cholesky decomposition
    Z1 = X1
    Z2 = rho * X1 + np.sqrt(1 - rho**2) * X2

    # Calculate terminal asset values using GBM solution
    S1_T = S1 * np.exp((mu1 - 0.5 * sigma1**2) * T + sigma1 * np.sqrt(T) * Z1)
    S2_T = S2 * np.exp((mu2 - 0.5 * sigma2**2) * T + sigma2 * np.sqrt(T) * Z2)

    return S1_T, S2_T

def monte_carlo_worstof_option(
    option_type: str,
    S1: float,
    S2: float,
    K: float,
    T: float,
    r: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    q1: float = 0,
    q2: float = 0,
    num_samples: int = 10000,
    random_seed: int = 42,
    control_variate: bool = False,
    dampening_factor: float = 0.01,
    clip_payoffs: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Price a European Worst-Off option on two assets using Monte Carlo simulation.

    Args:
        option_type: Type of option: 'call'/'c' or 'put'/'p'.
        S1: Current price of asset 1.
        S2: Current price of asset 2.
        K: Strike price of the option.
        T: Time to maturity in years.
        r: Risk-free interest rate (annualized).
        sigma1: Volatility of asset 1 (annualized).
        sigma2: Volatility of asset 2 (annualized).
        rho: Correlation coefficient between the assets.
        q1: Dividend yield of asset 1 (annualized).
        q2: Dividend yield of asset 2 (annualized).
        num_samples: Number of Monte Carlo paths to simulate.
        random_seed: Seed for random number generation.
        control_variate: Whether to use control variate technique.
        dampening_factor: Factor to dampen control variate adjustment.
        clip_payoffs: Whether to clip negative payoffs to zero.

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

    # Initialize random number generator
    np.random.seed(random_seed)

    # Risk-adjusted drifts (Itô correction)
    mu1 = r - q1 - 0.5 * sigma1**2
    mu2 = r - q2 - 0.5 * sigma2**2

    # Generate correlated terminal asset values
    S1_T, S2_T = generate_correlated_paths(
        S1=S1,
        S2=S2,
        mu1=mu1,
        mu2=mu2,
        sigma1=sigma1,
        sigma2=sigma2,
        rho=rho,
        T=T,
        num_samples=num_samples,
    )

    # Calculate normalized terminal prices (relative performance)
    S1_norm = S1_T / S1
    S2_norm = S2_T / S2

    # Worst performing asset mask
    worst_is_S1 = S1_norm <= S2_norm

    # Normalized maturity prices
    S1_norm = S1_T / S1
    S2_norm = S2_T / S2
    worst_ST = np.where(worst_is_S1, S1_T, S2_T) # Worst performing actual terminal value
    worst_norm = np.minimum(S1_norm, S2_norm) # Worst performing normalized value

    # Calculate option payoffs at maturity
    if option_type_str == "call":
        payoffs = np.maximum(worst_norm - K, 0) * worst_ST
    else:
        payoffs = np.maximum(K - worst_norm, 0) * worst_ST

    if control_variate:
        # Expected normalized terminal values
        expected_S1_norm = np.exp((r - q1 - 0.5 * sigma1**2) * T)
        expected_S2_norm = np.exp((r - q2 - 0.5 * sigma2**2) * T)

        # For paths where S1 is the worst performing
        S1_worst_indices = np.where(worst_is_S1)[0]
        if len(S1_worst_indices) > 0:
            cov_payoff_S1 = np.cov(payoffs[S1_worst_indices], S1_norm[S1_worst_indices])[0, 1]
            var_S1 = np.var(S1_norm[S1_worst_indices])
            beta1 = cov_payoff_S1 / var_S1 if var_S1 > 0 else 0

            # Add dampening factor to prevent overadjustment
            beta1 = dampening_factor * beta1
        else:
            beta1 = 0

        # For paths where S2 is the worst performing
        S2_worst_indices = np.where(~worst_is_S1)[0]
        if len(S2_worst_indices) > 0:
            cov_payoff_S2 = np.cov(payoffs[S2_worst_indices], S2_norm[S2_worst_indices])[0, 1]
            var_S2 = np.var(S2_norm[S2_worst_indices])
            beta2 = cov_payoff_S2 / var_S2 if var_S2 > 0 else 0

            # Add dampening factor to prevent overadjustment
            beta2 = dampening_factor * beta2
        else:
            beta2 = 0

        # Apply selective control variate adjustment
        adjusted_payoffs = np.copy(payoffs)

        # Adjust paths where S1 is the worst performing
        if len(S1_worst_indices) > 0:
            adjusted_payoffs[S1_worst_indices] = (
                payoffs[S1_worst_indices] -
                beta1 * (S1_norm[S1_worst_indices] - expected_S1_norm) * S1
            )

        # Adjust paths where S2 is the worst performing
        if len(S2_worst_indices) > 0:
            adjusted_payoffs[S2_worst_indices] = (
                payoffs[S2_worst_indices] -
                beta2 * (S2_norm[S2_worst_indices] - expected_S2_norm) * S2
            )

        if clip_payoffs:
            # Safety check to prevent negative option prices
            adjusted_payoffs = np.maximum(adjusted_payoffs, 0)

        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * adjusted_payoffs
    else:
        # Discount to present value without control variate
        discounted_payoffs = np.exp(-r * T) * payoffs

    # Calculate the option price as the average of discounted payoffs
    option_price = np.mean(discounted_payoffs)

    # Calculate standard error
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))

    # Prepare result
    result = {
        "price": option_price,
        "error": std_error,
        "computation_time": time.time() - start_time,
        "beta1": beta1 if control_variate else None,
        "beta2": beta2 if control_variate else None,
        "worst_is_S1_pct": np.mean(worst_is_S1) * 100  # Percentage of paths where S1 is worst
    }

    return result

def get_worstof_mc_prices(
    options_df: pd.DataFrame,
    num_samples: int = 10000,
    seed: int = 42,
    control_variate: bool = False,
    dampening_factor: float = 0.01,
    clip_payoffs: bool = True
) -> pd.DataFrame:
    """
    Add Monte Carlo pricing for worst-off options to a DataFrame.

    Args:
        options_df: DataFrame with option contract information.
                   Required columns: option_type, S1, S2, strike_price, maturity,
                   risk_free_rate, sigma1, sigma2, rho, q1, q2
        num_samples: Number of Monte Carlo samples.
        seed: Base random seed.
        control_variate: Whether to use control variate technique.
        dampening_factor: Factor to dampen control variate adjustment.
        clip_payoffs: Whether to clip negative payoffs to zero.

    Returns:
        DataFrame with added mc_price and mc_error columns.
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = options_df.copy()

    # Initialize new columns
    result_df["mc_price"] = np.nan
    result_df["mc_se"] = np.nan
    result_df["mc_compute_time"] = np.nan
    result_df["beta1"] = np.nan
    result_df["beta2"] = np.nan
    result_df["worst_is_S1_pct"] = np.nan

    # Process each option
    for idx in range(len(options_df)):
        # Calculate Monte Carlo price
        idx_seed = seed + idx

        mc_result = monte_carlo_worstof_option(
            option_type=options_df["option_type"].iloc[idx],
            S1=options_df["spot_price1"].iloc[idx],
            S2=options_df["spot_price2"].iloc[idx],
            K=options_df["strike_performance"].iloc[idx],
            T=options_df["maturity"].iloc[idx],
            r=options_df["risk_free_rate"].iloc[idx],
            sigma1=options_df["volatility1"].iloc[idx],
            sigma2=options_df["volatility2"].iloc[idx],
            rho=options_df["correlation"].iloc[idx],
            q1=options_df["dividend_yield1"].iloc[idx],
            q2=options_df["dividend_yield2"].iloc[idx],
            num_samples=num_samples,
            random_seed=idx_seed,
            control_variate=control_variate,
            dampening_factor=dampening_factor,
            clip_payoffs=clip_payoffs,
        )

        # Update DataFrame with results
        result_df.at[idx, "mc_price"] = mc_result["price"]
        result_df.at[idx, "mc_se"] = mc_result["error"]
        result_df.at[idx, "mc_compute_time"] = mc_result["computation_time"]
        result_df.at[idx, "beta1"] = mc_result["beta1"]
        result_df.at[idx, "beta2"] = mc_result["beta2"]
        result_df.at[idx, "worst_is_S1_pct"] = mc_result["worst_is_S1_pct"]

    return result_df

# Example usage
if __name__ == "__main__":

    from data import generate_option_dataset

    options_df = generate_option_dataset(worst_of=True, n_samples=1000)

    # Price without control variate
    results_no_cv = get_worstof_mc_prices(
        options_df,
        num_samples=10000,
        control_variate=False
    )
    print("Without Control Variate:")
    print(results_no_cv[["mc_price", "mc_se", "mc_compute_time", "worst_is_S1_pct"]])

    # Price with control variate
    results_with_cv = get_worstof_mc_prices(
        options_df,
        num_samples=10000,
        control_variate=True,
        dampening_factor=0.005,
        clip_payoffs=False
    )
    print("\nWith Control Variate:")
    print(results_with_cv[["mc_price", "mc_se", "mc_compute_time", "beta1", "beta2", "worst_is_S1_pct"]])

    # Compare standard errors
    if results_no_cv["mc_se"].iloc[0] > 0:
        variance_reduction = (
            (results_no_cv["mc_se"].iloc[0] - results_with_cv["mc_se"].iloc[0]) /
            results_no_cv["mc_se"].iloc[0] * 100
        )
        print(f"\nVariance reduction: {variance_reduction:.2f}%")


    # Print the number of entries in the control variate method which have negative mc_price
    negative_prices = results_with_cv[results_with_cv["mc_price"] < 0]
    print(f"\nNumber of negative prices with control variate: {len(negative_prices)}")
