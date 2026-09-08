"""
Linear Regression Core Mathematics Module
=========================================
Course: Fundamentals of Artificial Intelligence
Project: Linear Regression Using Gradient Descent and Stochastic Gradient Descent

This module defines the foundational linear regression hypothesis, loss calculation,
and performance metrics implemented from scratch.

Hypothesis:
    y_hat = w * x + b
    where:
        w : weight / slope (rate of score increase per hour of study)
        b : bias / intercept (baseline exam score with zero study hours)
        x : input feature (hours studied)
        y_hat : predicted target (predicted exam score)
"""

import numpy as np


def predict(X: np.ndarray, w: float, b: float) -> np.ndarray:
    """
    Compute linear regression predictions for given inputs.

    Equation:
        y_hat = w * x + b

    Parameters:
        X (np.ndarray): 1D array of input feature values (Hours Studied).
        w (float): Learned slope / weight parameter.
        b (float): Learned intercept / bias parameter.

    Returns:
        np.ndarray: Predicted exam scores (y_hat).
    """
    return w * X + b


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE) manually from scratch.

    Equation:
        MSE = (1 / n) * sum_{i=1}^{n} (y_i - y_hat_i)^2

    Why MSE is suitable for Linear Regression:
        1. Convexity: MSE is a strictly convex quadratic bowl in (w, b) space,
           guaranteeing a unique global minimum with no local minima trap.
        2. Differentiability: The squared error term is smooth and continuously
           differentiable everywhere, providing well-defined, closed-form gradients.
        3. Quadratic Penalty: Squaring errors penalizes larger deviations more severely
           than smaller ones, encouraging models to avoid catastrophic outliers.
        4. Statistical Rationale: Minimizing MSE is equivalent to Maximum Likelihood
           Estimation (MLE) under the assumption of normally distributed errors.

    Parameters:
        y_true (np.ndarray): Actual target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Mean squared error.
    """
    errors = y_true - y_pred
    return float(np.mean(errors ** 2))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) manually.

    Equation:
        RMSE = sqrt(MSE)

    RMSE translates the squared error penalty back into original target units
    (exam points), allowing direct, intuitive error interpretation.

    Parameters:
        y_true (np.ndarray): Actual target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Root mean squared error in original units.
    """
    return float(np.sqrt(calculate_mse(y_true, y_pred)))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Coefficient of Determination (R-squared) manually.

    Equation:
        R^2 = 1 - (SS_res / SS_tot)
        where:
            SS_res = sum_{i=1}^{n} (y_i - y_hat_i)^2   (Residual sum of squares)
            SS_tot = sum_{i=1}^{n} (y_i - y_mean)^2    (Total sum of squares)

    R^2 measures the proportion of variance in the dependent variable (Exam Score)
    explained by the linear model of the independent variable (Hours Studied).
    An R^2 of 1.0 indicates a perfect fit, while 0.0 indicates performance
    equivalent to predicting the mean of the training target.

    Parameters:
        y_true (np.ndarray): Actual target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: R-squared score (can be <= 1.0).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return float(1.0 - (ss_res / ss_tot))
