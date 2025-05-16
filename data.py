import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple


def generate_option_dataset(
    n_samples: int = 100,
    seed: int = 1995,
    maturity_distribution: Dict[str, float] = {
        "weekly": 0.25,
        "bi-weekly": 0.2,
        "monthly": 0.45,
        "leaps": 0.10,
    },
    underlying_log_mean: float = np.log(100),
    underlying_log_std: float = 0.5,
    moneyness_mean: float = 1.0,
    moneyness_std: float = 0.15,
    moneyness_range: Tuple[float, float] = (0.7, 1.3),
    rate_mean: float = 0.035,
    rate_std: float = 0.005,
    rate_range: Tuple[float, float] = (0.005, 0.05),
    vol_log_mean: float = np.log(0.25),
    vol_log_std: float = 0.4,
    vol_range: Tuple[float, float] = (0.1, 0.6),
    zero_div_prob: float = 0.3,
    div_log_mean: float = np.log(0.02),
    div_log_std: float = 0.5,
    div_range: Tuple[float, float] = (0.0, 0.06),
    worst_of: bool = False,
) -> pd.DataFrame:
    """Generate a dataset of option prices using both Black-Scholes and Monte Carlo.

    Args:
        n_samples: Number of different option combinations to generate. Default is 100.
        seed: Random seed for reproducibility. Default is 1995.

        # Maturity distribution parameters
        maturity_distribution: Dictionary with probabilities for different option maturities. The keys are:
            - weekly: 1 week
            - bi-weekly: 2 weeks
            - monthly: 1-12 months
            - leaps: 1-2 years
        The values are the probabilities for each type of option. The sum of probabilities should equal 1.

        # Underlying price parameters
        underlying_log_mean: Mean of log-normal distribution for underlying prices in log space.
                            Default is 4.6 (~ln(100)), representing a stock price around $100.
        underlying_log_std: Standard deviation of log-normal distribution for underlying prices.
                        Default is 0.5, producing a realistic spread of stock prices mostly between $40-$250.

        # Strike price parameters
        moneyness_mean: Mean of moneyness ratio (K/S) distribution.
                        Default is 1.0 (at-the-money) where most liquid options are found.
        moneyness_std: Standard deviation of moneyness ratio distribution.
                        Default is 0.15 (15%), reflecting higher density of strikes near ATM
                        and fewer strikes deep ITM or OTM based on typical options chains.
        moneyness_range: Tuple with (min, max) allowed moneyness ratios.
                            Default is (0.7, 1.3) representing 70%-130% of the stock price,
                            which covers the most actively traded strikes in typical markets.

        # Risk-free rate parameters
        rate_mean: Mean of normal distribution for risk-free rates. Default is 0.035 (3.5%).
        rate_std: Standard deviation of normal distribution for risk-free rates. Default is 0.005 (0.5%),
                    producing realistic variability in rates.
        rate_range: Tuple with (min, max) allowed risk-free rates.
                    Default is (0.005, 0.05) or 0.5%-5%, reflecting the historical range of
                    short to 10-year treasury yields in recent decades.

        # Volatility parameters
        vol_log_mean: Mean of log-normal distribution for volatility (in log space).
                    Default is ln(0.25), producing a mean volatility around 25%,
                    which is typical for average stock volatility.
        vol_log_std: Standard deviation of log-normal distribution for volatility.
                    Default is 0.4, generating a right-skewed distribution where most stocks
                    have volatilities between 15%-40% with some higher outliers.
        vol_range: Tuple with (min, max) allowed volatility values.
                    Default is (0.1, 0.6) or 10%-60%, covering the range from low volatility
                    blue chips (10-15%) to higher volatility growth stocks (40-60%).

        # Dividend parameters
        zero_div_prob: Probability of a stock having zero dividend yield.
                        Default is 0.3 (30%), reflecting that roughly a third of stocks
                        (especially growth/tech) don't pay dividends.
        div_log_mean: Mean of log-normal distribution for dividend yields (in log space).
                    Default is ln(0.02), centering positive dividends around 2%,
                    which is close to the average S&P 500 dividend yield.
        div_log_std: Standard deviation of log-normal distribution for dividend yields.
                    Default is 0.5, creating a realistic spread where most dividend-paying
                    stocks have yields between 1%-4%.
        div_range: Tuple with (min, max) allowed dividend yields.
                    Default is (0.0, 0.06) or 0%-6%, where 6% represents high-yield
                    stocks/sectors (REITs, utilities, etc.).

        # Worst-Off Basket of Two Options
        worst_of: If True, generates a basket of two options for worst-off option pricing (bonus challenge).

    Returns:
        pd.DataFrame: DataFrame with columns for option parameters and prices:
            - option_type: "call" or "put"
            - spot_price: Current price of the underlying
            - strike_price: Strike price of the option
            - maturity: Time to expiration in years
            - risk_free_rate: Annual risk-free rate
            - volatility: Annual volatility of the underlying
            - dividend_yield: Annual dividend yield of the underlying
            - bs_price: Black-Scholes analytical price (if requested)
            - mc_price: Monte Carlo price estimate (if requested)
            - mc_error: Standard error of Monte Carlo estimate (if requested)
            - price_difference: Absolute difference between BS and MC prices
            - price_difference_percent: Percentage difference between BS and MC prices
    """
    np.random.seed(seed)
    reference_date = datetime.now()

    assert (
        sum(maturity_distribution.values()) == 1
    ), "Maturity distribution must sum to 1."
    assert 0 <= zero_div_prob <= 1.0, "Zero dividend probability must be between 0 and 1"

    # Underlying price checks
    assert np.exp(underlying_log_mean - 3*underlying_log_std) >= 1.0, "Minimum realistic stock price should be at least $1"
    assert np.exp(underlying_log_mean + 3*underlying_log_std) <= 10000.0, "Maximum realistic stock price should be below $10,000"

    # Moneyness ratio checks
    assert 0.1 <= moneyness_range[0] < moneyness_range[1] <= 3.0, "Moneyness range should be within [0.1, 3.0]"
    assert moneyness_range[0] <= moneyness_mean <= moneyness_range[1], "Moneyness mean must be within specified range"

    # Risk-free rate checks
    assert 0.0 <= rate_range[0] < rate_range[1] <= 0.05, "Risk-free rates should be within [0%, 15%]"
    assert rate_range[0] <= rate_mean <= rate_range[1], "Rate mean must be within specified range"

    # Volatility checks
    assert 0.05 <= vol_range[0] < vol_range[1] <= 2.0, "Volatility should be within [5%, 200%]"
    assert vol_range[0] <= np.exp(vol_log_mean) <= vol_range[1], "Volatility mean must be within specified range"

    # Dividend yield checks
    assert 0.0 <= div_range[0] < div_range[1] <= 0.15, "Dividend yields should be within [0%, 15%]"
    assert div_range[0] <= np.exp(div_log_mean) <= div_range[1], "Dividend mean must be within specified range"


    # Option types: call or put
    option_types = np.random.choice(["call", "put"], size=n_samples)

    # Underlying prices: sampled from a log-normal distribution
    spot_prices = np.random.lognormal(
        underlying_log_mean, underlying_log_std, n_samples
    )

    if worst_of:
        spot_prices2 = np.random.lognormal(
            underlying_log_mean, underlying_log_std, n_samples
        )

    # Strikes prices: are selected w.r.t. the underlying price using the normal distribution
    # Generate all moneyness values at once using normal distribution
    moneyness = np.random.normal(moneyness_mean, moneyness_std, size=n_samples)

    # Clip all values to the valid bounds in one operation (e.g., 0.7 to 1.3)
    moneyness = np.clip(moneyness, moneyness_range[0], moneyness_range[1])

    # Calculate all strike prices from spot prices and moneyness in one operation
    strikes = spot_prices * moneyness

    if worst_of:
        strike_performance = moneyness # Use the moneyness which represents the % value of the underlying

    # Maturities: standard option expiration cycles with concentrations at weekly, monthly, quarterly
    maturities = np.zeros(n_samples)
    for i in range(n_samples):
        # Calculate cumulative probabilities
        cumulative_prob = 0
        thresholds = {}
        for key in ["weekly", "bi-weekly", "monthly", "leaps"]:
            cumulative_prob += maturity_distribution[key]
            thresholds[key] = cumulative_prob

        # Generate a random number to determine the option maturity
        r = np.random.random()
        if r < thresholds["weekly"]:
            # Weekly options (1 week)
            weeks = 1
            expiry_date = reference_date + timedelta(weeks=weeks)
        elif r < thresholds["bi-weekly"]:
            # Bi-weekly options (2 weeks)
            weeks = 2
            expiry_date = reference_date + timedelta(weeks=weeks)
        elif r < thresholds["monthly"]:
            # Monthly options (1-12 months)
            months = int(np.random.choice([1, 2, 3, 6, 9, 12]))
            expiry_date = reference_date + timedelta(days=30 * months)
        else:
            # LEAPS (1-2 years)
            years = float(np.random.uniform(1, 2))
            expiry_date = reference_date + timedelta(days=int(365 * years))

        # Calculate time to maturity in years
        days_to_expiry = (expiry_date - reference_date).days
        maturities[i] = days_to_expiry / 365.0

    # Risk-free interest rates: sampled from a normal distribution
    rates = np.random.normal(rate_mean, rate_std, n_samples)
    rates = np.clip(
        rates, rate_range[0], rate_range[1]
    )  # Clip to valid range (e.g., 0.5%-5%)

    # Volatility: sampled from a log-normal distribution
    sigmas = np.random.lognormal(vol_log_mean, vol_log_std, n_samples)
    sigmas = np.clip(
        sigmas, vol_range[0], vol_range[1]
    )  # Clip to valid range (e.g., 10%-60%)

    if worst_of:
        sigmas2 = np.random.lognormal(vol_log_mean, vol_log_std, n_samples)
        sigmas2 = np.clip(
            sigmas2, vol_range[0], vol_range[1]
        )  # Clip to valid range (e.g., 10%-60%)

    # Dividend yields: mix of zero and log-normal
    dividends = np.zeros(n_samples)
    div_mask = (
        np.random.random(n_samples) >= zero_div_prob
    )  # Configurable percentage of stocks don't pay dividends
    dividends[div_mask] = np.random.lognormal(
        div_log_mean, div_log_std, sum(div_mask)
    )  # Use log-normal distribution
    dividends = np.clip(
        dividends, div_range[0], div_range[1]
    )  # Clip to valid range (e.g., 0%-6%)

    if worst_of:
        dividends2 = np.zeros(n_samples)
        div_mask2 = (
            np.random.random(n_samples) >= zero_div_prob
        )
        dividends2[div_mask2] = np.random.lognormal(
            div_log_mean, div_log_std, sum(div_mask2)
        )
        dividends2 = np.clip(
            dividends2, div_range[0], div_range[1]
        )


    # Create dict and return DataFrame
    if not worst_of:
        option_data = {
            "option_type": option_types,
            "spot_price": spot_prices,
            "strike_price": strikes,
            "maturity": maturities,
            "risk_free_rate": rates,
            "volatility": sigmas,
            "dividend_yield": dividends,
        }
    else:
        correlation = np.random.uniform(-1, 1, n_samples)
        option_data = {
            "option_type": option_types,
            "spot_price1": spot_prices,
            "spot_price2": spot_prices2,
            "strike_performance": strike_performance,
            "maturity": maturities,
            "risk_free_rate": rates,
            "volatility1": sigmas,
            "volatility2": sigmas2,
            "dividend_yield1": dividends,
            "dividend_yield2": dividends2,
            "correlation": correlation,
        }

    return pd.DataFrame(option_data)
