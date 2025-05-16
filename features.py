import numpy as np
import pandas as pd


def get_features(options_df: pd.DataFrame, worst_of: bool = False) -> pd.DataFrame:
    """Transforms option parameters into features suitable for machine learning models for option pricing.
    This function creates a focused set of non-redundant features that capture the most important
    statistical and financial relationships in options pricing without directly implementing the
    Black-Scholes formula.

    Args:
        options_df: DataFrame with basic option parameters, expected to contain:
            - option_type: "call" or "put"
            - spot_price: Current price of the underlying
            - strike_price: Strike price of the option
            - maturity: Time to expiration in years
            - risk_free_rate: Annual risk-free rate
            - volatility: Annual volatility of the underlying
            - dividend_yield: Annual dividend yield of the underlying

        worst_of: Whether to generate features for pricing worst-off options.

    Returns:
        pd.DataFrame: Original DataFrame with 12 additional key features, all normalized using StandardScaler:
            - is_call: Binary indicator for call options (1=call, 0=put)
            - moneyness: Ratio of spot price to strike price (S/K)
            - log_moneyness: Natural log of moneyness
            - time_sqrt: Square root of time to maturity
            - vol_time: Volatility × square root of time
            - carry_cost: Cost of carry (r-q)
            - intrinsic_value: Intrinsic value of the option
            - is_itm: Binary indicator if option is in-the-money (1=ITM, 0=OTM)
            - moneyness_vol: Interaction between moneyness and volatility
            - moneyness_time: Interaction between moneyness and time
            - vol_squared: Squared volatility term for non-linear effects
            - carry_time: Interaction between carry cost and time
    """
    # Create a copy to avoid modifying the original DataFrame
    df = options_df.copy()

    # Basic option properties
    df["is_call"] = (df["option_type"] == "call").astype(int)

    # Key price relationship
    if worst_of:
        df["moneyness"] =df["strike_performance"]
    else:
        df["moneyness"] = df["spot_price"] / df["strike_price"]
    df["log_moneyness"] = np.log(df["moneyness"])

    # Time transformation
    df["time_sqrt"] = np.sqrt(df["maturity"])

    # Statistical uncertainty measure
    if worst_of:
        df["vol_time1"] = df["volatility1"] * df["time_sqrt"]
        df["vol_time2"] = df["volatility2"] * df["time_sqrt"]
    else:
        df["vol_time"] = df["volatility"] * df["time_sqrt"]

    # Cost of carry (adjusted interest rate for dividends)
    if worst_of:
        df["carry_cost1"] = df["risk_free_rate"] - df["dividend_yield1"]
        df["carry_cost2"] = df["risk_free_rate"] - df["dividend_yield2"]
    else:
        df["carry_cost"] = df["risk_free_rate"] - df["dividend_yield"]

    # Intrinsic value
    if not worst_of:
        df["intrinsic_value"] = np.where(
            df["option_type"] == "call",
            np.maximum(0, df["spot_price"] - df["strike_price"]),
            np.maximum(0, df["strike_price"] - df["spot_price"])
        )

    # In-the-money indicator
    if not worst_of:
        df["is_itm"] = np.where(
            df["option_type"] == "call",
            df["spot_price"] > df["strike_price"],
            df["spot_price"] < df["strike_price"]
        ).astype(int)

    # Non-linear volatility effect
    if worst_of:
        df["vol_squared1"] = df["volatility1"] ** 2
        df["vol_squared2"] = df["volatility2"] ** 2
    else:
        df["vol_squared"] = df["volatility"] ** 2

    # Key interaction features
    if worst_of:
        df["moneyness_vol1"] = df["moneyness"] * df["volatility1"]
        df["moneyness_vol2"] = df["moneyness"] * df["volatility2"]
        df["moneyness_time"] = df["moneyness"] * df["maturity"]
        df["carry_time1"] = df["carry_cost1"] * df["maturity"]
        df["carry_time2"] = df["carry_cost2"] * df["maturity"]
    else:
        df["moneyness_vol"] = df["moneyness"] * df["volatility"]
        df["moneyness_time"] = df["moneyness"] * df["maturity"]
        df["carry_time"] = df["carry_cost"] * df["maturity"]

    return df
