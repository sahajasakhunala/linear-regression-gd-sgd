"""
Model Evaluation and Metrics Module
===================================
Course: Fundamentals of Artificial Intelligence
Project: Linear Regression Using Gradient Descent and Stochastic Gradient Descent

This module provides comprehensive evaluation metrics, side-by-side comparison tables,
test-set sample prediction tables, and convergence analysis for GD and SGD models.
"""

import numpy as np
import pandas as pd
from src.linear_regression import predict, calculate_mse, calculate_rmse, calculate_r2


def evaluate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w: float,
    b: float,
    model_name: str = "Model"
) -> dict:
    """
    Compute full evaluation metrics on both training and unseen test sets.

    Parameters:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_test (np.ndarray): Unseen test features.
        y_test (np.ndarray): Unseen test targets.
        w (float): Learned slope / weight.
        b (float): Learned intercept / bias.
        model_name (str): Label for the model.

    Returns:
        dict: Evaluated performance metrics.
    """
    train_preds = predict(X_train, w, b)
    test_preds = predict(X_test, w, b)

    train_mse = calculate_mse(y_train, train_preds)
    train_rmse = calculate_rmse(y_train, train_preds)
    train_r2 = calculate_r2(y_train, train_preds)

    test_mse = calculate_mse(y_test, test_preds)
    test_rmse = calculate_rmse(y_test, test_preds)
    test_r2 = calculate_r2(y_test, test_preds)

    return {
        "model_name": model_name,
        "weight": w,
        "bias": b,
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "train_preds": train_preds,
        "test_preds": test_preds
    }


def compute_ols_solution(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Compute closed-form Ordinary Least Squares (OLS) solution on training set:
        w = Cov(X, y) / Var(X) = sum((x - x_bar)(y - y_bar)) / sum((x - x_bar)^2)
        b = y_bar - w * x_bar

    Parameters:
        X_train (np.ndarray): Training feature array.
        y_train (np.ndarray): Training target array.

    Returns:
        tuple: (w_ols, b_ols)
    """
    x_mean = np.mean(X_train)
    y_mean = np.mean(y_train)

    numerator = np.sum((X_train - x_mean) * (y_train - y_mean))
    denominator = np.sum((X_train - x_mean) ** 2)

    w_ols = float(numerator / denominator)
    b_ols = float(y_mean - w_ols * x_mean)

    return w_ols, b_ols


def create_comparison_table(gd_eval: dict, sgd_eval: dict, ols_eval: dict = None) -> pd.DataFrame:
    """
    Build a side-by-side comparative metric DataFrame between GD, SGD, and reference OLS.

    Parameters:
        gd_eval (dict): Evaluation metrics from Batch GD.
        sgd_eval (dict): Evaluation metrics from SGD.
        ols_eval (dict, optional): Reference metrics from analytical OLS.

    Returns:
        pd.DataFrame: Formatted comparison table.
    """
    metrics = [
        "Training MSE",
        "Training RMSE",
        "Training R^2",
        "Test MSE",
        "Test RMSE",
        "Test R^2",
        "Final Weight (w)",
        "Final Bias (b)"
    ]

    gd_vals = [
        f"{gd_eval['train_mse']:.4f}",
        f"{gd_eval['train_rmse']:.4f}",
        f"{gd_eval['train_r2']:.4f}",
        f"{gd_eval['test_mse']:.4f}",
        f"{gd_eval['test_rmse']:.4f}",
        f"{gd_eval['test_r2']:.4f}",
        f"{gd_eval['weight']:.4f}",
        f"{gd_eval['bias']:.4f}"
    ]

    sgd_vals = [
        f"{sgd_eval['train_mse']:.4f}",
        f"{sgd_eval['train_rmse']:.4f}",
        f"{sgd_eval['train_r2']:.4f}",
        f"{sgd_eval['test_mse']:.4f}",
        f"{sgd_eval['test_rmse']:.4f}",
        f"{sgd_eval['test_r2']:.4f}",
        f"{sgd_eval['weight']:.4f}",
        f"{sgd_eval['bias']:.4f}"
    ]

    data = {
        "Metric": metrics,
        "Gradient Descent (GD)": gd_vals,
        "Stochastic Gradient Descent (SGD)": sgd_vals
    }

    if ols_eval is not None:
        ols_vals = [
            f"{ols_eval['train_mse']:.4f}",
            f"{ols_eval['train_rmse']:.4f}",
            f"{ols_eval['train_r2']:.4f}",
            f"{ols_eval['test_mse']:.4f}",
            f"{ols_eval['test_rmse']:.4f}",
            f"{ols_eval['test_r2']:.4f}",
            f"{ols_eval['weight']:.4f}",
            f"{ols_eval['bias']:.4f}"
        ]
        data["Analytical OLS (Reference)"] = ols_vals

    return pd.DataFrame(data)


def create_test_prediction_table(
    X_test: np.ndarray,
    y_test: np.ndarray,
    gd_preds: np.ndarray,
    sgd_preds: np.ndarray
) -> pd.DataFrame:
    """
    Build a table of unseen test observations comparing actual scores with predictions.

    Parameters:
        X_test (np.ndarray): Test feature values (Hours Studied).
        y_test (np.ndarray): Actual test target scores.
        gd_preds (np.ndarray): Predictions from Batch GD model.
        sgd_preds (np.ndarray): Predictions from SGD model.

    Returns:
        pd.DataFrame: Formatted test predictions table.
    """
    df = pd.DataFrame({
        "Hours Studied": X_test,
        "Actual Score": y_test,
        "GD Prediction": np.round(gd_preds, 2),
        "SGD Prediction": np.round(sgd_preds, 2),
        "GD Error (y - y_hat)": np.round(y_test - gd_preds, 2),
        "SGD Error (y - y_hat)": np.round(y_test - sgd_preds, 2)
    })
    return df
