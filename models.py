from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List[Any]],
    n_iter: int = 100,
    cv: int = 5,
    random_state: int = 1995,
    n_jobs: int = -1,
    verbose: int = 0
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """Tunes XGBoost model hyperparameters using RandomizedSearchCV.

    Args:
        X_train: Training feature data.
        y_train: Training target data.
        param_grid: Dictionary of parameter distributions for RandomizedSearchCV.
            If None, uses default parameter grid.
        n_iter: Number of parameter settings sampled in RandomizedSearchCV.
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs for training.
        verbose: Verbosity level.

    Returns:
        Tuple containing:
            - Best estimator found by RandomizedSearchCV
            - Dictionary of best parameters
    """

    # Create the base XGBoost model
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        tree_method="hist"
    )

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state
    )

    # Fit the model to training data
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_model, best_params


def train_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: Dict[str, Any],
    num_boost_round: Optional[int] = 1000,
    early_stopping_rounds: int = 50,
    verbose_eval: bool = False,
    random_state: int = 1995
) -> Tuple[xgb.Booster, Dict[str, Dict[str, List[float]]]]:
    """Train a final XGBoost model using the best parameters from tuning.

    Args:
        X_train: Training feature data.
        y_train: Training target data.
        X_val: Validation feature data.
        y_val: Validation target data.
        best_params: Best parameters found during tuning.
        num_boost_round: Maximum number of boosting rounds. If "n_estimators" is provided in best_params
                         this will be ignored.
        early_stopping_rounds: Number of rounds with no improvement to stop training.
        verbose_eval: Whether to print evaluation logs.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple containing:
            - Trained XGBoost model
            - Dictionary containing evaluation results
    """
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Prepare parameters
    params = best_params.copy()
    params["objective"] = "reg:squarederror"
    params["seed"] = random_state
    num_boost_round = params.pop("n_estimators", num_boost_round)

    # Monitor both training and validation metrics. Use validation for early stopping
    evals = [(dtrain, "train"), (dval, "validation")]

    # Create a dictionary to store evaluation results
    evals_result = {}

    # Train final model
    print("Training final model with best parameters...")
    final_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
        evals_result=evals_result
    )

    return final_model, evals_result


def evaluate(
    model: xgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    evals_result: Dict[str, Dict[str, List[float]]],
    plot_results: bool = True,
) -> pd.DataFrame:
    """Evaluates the trained XGBoost model on the test set.

    Args:
        model: Trained XGBoost Booster model.
        X_test: Test feature data.
        y_test: Test target data.
        evals_result: Dictionary of evaluation results from training.
        plot_results: Whether to plot feature importance and learning curves.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Convert test data to DMatrix
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Make predictions
    y_pred = model.predict(dtest)

    # Calculate metrics
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_wmape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100

    if plot_results:
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()

        # Plot learning curves
        if evals_result and "train" in evals_result and "rmse" in evals_result["train"]:
            epochs = len(evals_result["train"]["rmse"])
            x_axis = range(0, epochs)
            plt.figure(figsize=(10, 6))
            plt.plot(x_axis, evals_result["train"]["rmse"], label="Train")
            plt.plot(x_axis, evals_result["validation"]["rmse"], label="Validation")
            plt.legend()
            plt.xlabel("Boosting Round")
            plt.ylabel("RMSE")
            plt.title("XGBoost RMSE")
            plt.grid()
            plt.show()

    # Return metrics
    return pd.DataFrame({
        "mse": [test_mse],
        "rmse": [test_rmse],
        "mae": [test_mae],
        "r2": [test_r2],
        "wmape": [test_wmape]
    })
