# Pricing Worst-Of Options
The core analysis is contained in [`main.ipynb`](main.ipynb), with relevant functions and modules stored in the associated python scripts in this project for clarity (as referenced in [`main.ipynb`](main.ipynb)).

## Installation
To use [`main.ipynb`](main.ipynb), please create an environment with and install dependencies using:
```bash
pip install xgboost scikit-learn matplotlib pandas rich ipykernel
```
or run the first code cell in [`main.ipynb`](main.ipynb) while having an active Python environment with existing Jupyter support.

## 📘 What is a Worst-Of Option?
A **worst-of option** is an exotic derivative based on a basket of underlying assets. Its payoff depends on the **worst-performing asset** in the basket at maturity, thereby making it riskier and more complex than vanilla options.

### 🔍 Key Features

- **Basket Dependency**: Involves two or more underlying assets, typically equities or indices.
- **Payoff Trigger**: The option references the *minimum* normalized return across assets.
- **European Payoff Example**:
    $$\text{Payoff} = \max(\min(\bar{S}_1(T), \bar{S}_2(T)) - K, 0) \tag{Call}$$
    $$\text{Payoff} = \max(K - \min(\bar{S}_1(T), \bar{S}_2(T)), 0) \tag{Put}$$

Where $\bar{S}_i(T) = S_i(T) / S_i(0)$ is normalized return at maturity, and $K \in [0, 1]$ is the strike performance.

### 🧠 Pricing Considerations

Pricing these options is non-trivial due to:
- Dependency on **multiple correlated underlyings**
- Need to **model asset correlations**
- **Non-linear** payoff structure

The lack of an analytical solution makes this especially difficult. This repo implements a Monte Carlo simulation engine, with a novel **control variate** method to price European-style worst-of options and analyze their behavior under various market scenarios.

## 🛠️ Proposed Pricing Method
In this section, I detail how to generate correlated asset price paths to model a basket of 2 options and ultimately price the worst-of option derived from it. I also go into detail of how to adapt standard control variate techniques to pricing worst-of options, which reduce the variance of monte carlo paths.

### Monte Carlo Simulation for Correlated Assets
To simulate correlated asset terminal values, we can use a direct implementation of the GBM solution with Cholesky decomposition. Starting with uncorrelated standard normal variables $X^{(1)}, X^{(2)} \sim \mathcal{N}(0,1)$, we construct correlated random variables:

$$
\begin{align*}
   Z^{(1)} &= X^{(1)}, \\
   Z^{(2)} &= \rho X^{(1)} + \sqrt{1 - \rho^2}X^{(2)},
\end{align*}
$$

This transformation forces $\text{Corr}(Z^{(1)}, Z^{(2)}) = \rho$. Using these correlated variables, we directly simulate the terminal asset values at time $T$:

$$
\begin{align*}
   S^{(1)}_T &= S^{(1)}_{0} \exp\Big(\left(\mu^{(1)} - \frac{(\sigma^{(1)})^2}{2}\right)T + \sigma^{(1)} \sqrt{T} Z^{(1)}\Big), \\
   S^{(2)}_T &= S^{(2)}_{0} \exp\Big(\left(\mu^{(2)} - \frac{(\sigma^{(2)})^2}{2}\right)T + \sigma^{(2)} \sqrt{T} Z^{(2)}\Big).
\end{align*}
$$

This approach ensures that the log returns of both assets from initial to terminal time maintain correlation $\rho$:

$$\rho = \text{Corr}\left(\log\left( \frac{S^{(1)}_T}{S^{(1)}_{0}} \right), \log\left( \frac{S^{(2)}_T}{S^{(2)}_{0}} \right)\right) =  \text{Corr}\left(\log\left(\bar{S}_T^{(1)} \right), \log\left( \bar{S}_T^{(2)} \right)\right), $$

where:
$$
\begin{align}
   \bar{S}_T^{(1)} = \frac{S_T^{(1)}}{S_0^{(1)}}, \ \ \ \bar{S}_T^{(2)} = \frac{S_T^{(2)}}{S_0^{(2)}}.
\end{align}
$$
are the normalized maturity prices. By engineering correlation at the log-return level over the entire time horizon, we capture the fundamental relationship between asset movements in a single step, which is computationally more efficient than simulating the full path while maintaining consistency with financial theory.


#### Control Variates for Worst-Of Options
For basket options with worst-of payoffs, we can extend the control variate technique using an indicator-based approach that focuses on the asset determining the payoff. Since the option pays based on the worst-performing asset, we apply a selective control variate adjustment with:

$$\hat{f}(\mathbf{S}_T) = f(\mathbf{S}_T) - I(\mathbf{\bar{S}}_T) \cdot \beta_1({S_T^{(1)} - \mathbb{E}[S_T^{(1)}]}) - (1-I(\mathbf{\bar{S}}_T)) \cdot \beta_2(S_T^{(2)} - \mathbb{E}[S_T^{(2)}]),$$

where $\mathbf{S}_T = (S_T^{(1)}, S_T^{(2)})^T$ are the terminal values from Monte Carlo, and:
$$
\begin{align*}
   I(\mathbf{\bar{S}}_T) = \begin{cases} 
      1 &\text{if } \bar{S}_T^{(1)} \leq \bar{S}_T^{(2)} \\
      0 &\text{otherwise}
   \end{cases}
\end{align*}
$$
is an indicator function which is $1$ if asset 1 is the worst-performing asset, and 0 otherwise, with respect to the normalized maturity prices $\mathbf{\bar{S}}_T = (\bar{S}_T^{(1)}, \bar{S}_T^{(2)})^T$. The optimal coefficients are calculated conditionally:

$$\beta_1 = \frac{\text{Cov}(f(\mathbf{S}_T), S_T^{(1)} | S_T^{(1)} \leq S_T^{(2)})}{\text{Var}(S_T^{(1)} | S_T^{(1)} \leq S_T^{(2)})}, \quad \beta_2 = \frac{\text{Cov}(f(\mathbf{S}_T), S_T^{(2)} | S_T^{(2)} < S_T^{(1)})}{\text{Var}(S_T^{(2)} | S_T^{(2)} < S_T^{(1)})}$$

This approach targets the variance reduction specifically to the asset driving the payoff in each simulation path. The underlying asset correlation structure is implicitly captured through the Monte Carlo paths generated via Cholesky decomposition, without requiring explicit handling in the control variate implementation. This selective adjustment efficiently reduces variance in worst-of option pricing, particularly for options sensitive to the relative performance between the basket assets.

#### Implementation
The implementation is contained in [`worst_of_option.py`](worst_of_option.py), which follows a similar structure to [`monte_carlo.py`](monte_carlo.py). The key differences are:

1. The [`generate_correlated_paths`](worst_of_option.py#L6) function creates correlated asset paths for multi-asset options
2. The [`monte_carlo_worstof_option`](worst_of_option.py#L50)  function handles:
  - Calculating the nonlinear worst-of payoff
  - Implementing the custom control variate model
  - Computing the final option price

#### Limitations & Future Improvements

While this method effectively generalizes the approach and experiments confirm variance reduction, there are opportunities for enhancement. The control variate model faces challenges with discontinuities at boundaries where assets have similar performance, and the current formulation of $\hat{f}(\mathbf{S}_T)$ doesn't fully capture asset interaction effects.

These limitations occasionally affected the calculation of $\beta_1$ and $\beta_2$ coefficients, sometimes resulting in large values that could produce theoretically impossible negative payoffs, and negative option prices as a result. To address this, I implemented two stabilizing mechanisms in `monte_carlo_worstof_option`:

1. A `dampening_factor` parameter to scale the control variate correction (providing smaller corrections)
2. A `clip_payoffs` argument to ensure non-negative payoffs

These adjustments successfully balance variance reduction with pricing accuracy for most scenarios. Future enhancements could incorporate correlation coefficients $\rho$ and interaction terms directly into $\hat{f}(\mathbf{S}_T)$, potentially achieving even greater variance reduction while maintaining pricing integrity.


#### Dataset Generation
The `worst_of` flag in [`generate_option_dataset`](data.py) provides us a synthetic dataset of a basket of two option contracts, with all the relevant parameters we will need to model a worst-of option.
