"""
Batch Gradient Descent (GD) Module
==================================
Course: Fundamentals of Artificial Intelligence
Project: Linear Regression Using Gradient Descent and Stochastic Gradient Descent

This module implements Batch Gradient Descent from scratch.
In Batch GD, parameter updates are computed using the average gradient over the
ENTIRE training dataset in every iteration.

Gradient Equations:
    MSE = (1 / n) * sum_{i=1}^{n} (y_i - (w * x_i + b))^2

    Partial derivative with respect to w (dw):
        dw = d(MSE)/dw = -(2 / n) * sum_{i=1}^{n} x_i * (y_i - y_hat_i)

    Partial derivative with respect to b (db):
        db = d(MSE)/db = -(2 / n) * sum_{i=1}^{n} (y_i - y_hat_i)

Parameter Update Rule:
    w = w - alpha * dw
    b = b - alpha * db
    where alpha is the learning rate.
"""

import numpy as np
from src.linear_regression import predict, calculate_mse


def train_gradient_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.01,
    iterations: int = 1000,
    initial_w: float = 0.0,
    initial_b: float = 0.0,
    tolerance: float = 1e-6
) -> tuple:
    """
    Train a Linear Regression model using Batch Gradient Descent from scratch.

    Parameters:
        X_train (np.ndarray): Training feature array (Hours Studied).
        y_train (np.ndarray): Training target array (Exam Scores).
        learning_rate (float): Step size alpha (default 0.01).
        iterations (int): Total number of iterations to perform (default 1000).
        initial_w (float): Initial slope value (default 0.0).
        initial_b (float): Initial intercept value (default 0.0).
        tolerance (float): Threshold to monitor stabilization (|L_t - L_{t-1}| < tolerance).

    Returns:
        tuple: (w, b, history)
            w (float): Final learned weight/slope.
            b (float): Final learned bias/intercept.
            history (dict): Dictionary tracking iterations, w, b, loss, and convergence info.
    """
    n = len(X_train)
    w = float(initial_w)
    b = float(initial_b)

    # Pre-allocate history structures
    iteration_history = []
    w_history = []
    b_history = []
    loss_history = []
    grad_w_history = []
    grad_b_history = []

    # Initial loss at iteration 0 before any updates
    initial_preds = predict(X_train, w, b)
    initial_loss = calculate_mse(y_train, initial_preds)

    iteration_history.append(0)
    w_history.append(w)
    b_history.append(b)
    loss_history.append(initial_loss)
    grad_w_history.append(0.0)
    grad_b_history.append(0.0)

    converged_iteration = None

    for it in range(1, iterations + 1):
        # 1. Forward pass: compute predictions for all training samples
        y_hat = predict(X_train, w, b)

        # 2. Compute prediction errors
        errors = y_train - y_hat

        # 3. Compute gradients of MSE with respect to w and b
        # dw = -(2/n) * sum(x_i * (y_i - y_hat_i))
        # db = -(2/n) * sum(y_i - y_hat_i)
        dw = - (2.0 / n) * np.sum(X_train * errors)
        db = - (2.0 / n) * np.sum(errors)

        # 4. Simultaneous parameter update
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 5. Compute updated training loss
        updated_preds = predict(X_train, w, b)
        current_loss = calculate_mse(y_train, updated_preds)

        # 6. Check convergence condition based on loss stability
        prev_loss = loss_history[-1]
        loss_diff = abs(prev_loss - current_loss)
        if converged_iteration is None and loss_diff < tolerance:
            converged_iteration = it

        # Record history
        iteration_history.append(it)
        w_history.append(w)
        b_history.append(b)
        loss_history.append(current_loss)
        grad_w_history.append(dw)
        grad_b_history.append(db)

    history = {
        "iterations": iteration_history,
        "w": w_history,
        "b": b_history,
        "loss": loss_history,
        "dw": grad_w_history,
        "db": grad_b_history,
        "converged_iteration": converged_iteration,
        "final_loss": loss_history[-1],
        "total_updates": iterations,
        "total_sample_evaluations": iterations * n
    }

    return w, b, history
