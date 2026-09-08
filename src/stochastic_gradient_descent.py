"""
Stochastic Gradient Descent (SGD) Module
========================================
Course: Fundamentals of Artificial Intelligence
Project: Linear Regression Using Gradient Descent and Stochastic Gradient Descent

This module implements Stochastic Gradient Descent from scratch.
Unlike Batch GD, SGD updates the parameters w and b IMMEDIATELY after evaluating
EACH individual training sample.

Single Sample Gradient Equations:
    Loss_i = (y_i - (w * x_i + b))^2

    Partial derivative with respect to w (dw_i):
        error_i = y_i - y_hat_i
        dw_i = d(Loss_i)/dw = -2 * x_i * error_i

    Partial derivative with respect to b (db_i):
        db_i = d(Loss_i)/db = -2 * error_i

Parameter Update Rule (After EVERY single sample):
    w = w - alpha * dw_i
    b = b - alpha * db_i

Why SGD Produces Fluctuations:
    Because individual training points have individual noise/deviations, the gradient
    of a single sample is a noisy, high-variance estimate of the true full-batch
    gradient. Therefore, the parameter path zig-zags and oscillates around the optimum.
"""

import numpy as np
from src.linear_regression import predict, calculate_mse


def train_stochastic_gradient_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 100,
    initial_w: float = 0.0,
    initial_b: float = 0.0,
    random_seed: int = 42,
    tolerance: float = 1e-6
) -> tuple:
    """
    Train a Linear Regression model using Stochastic Gradient Descent from scratch.

    Parameters:
        X_train (np.ndarray): Training feature array (Hours Studied).
        y_train (np.ndarray): Training target array (Exam Scores).
        learning_rate (float): Step size alpha (default 0.01).
        epochs (int): Number of complete passes through the training set (default 100).
        initial_w (float): Initial slope value (default 0.0).
        initial_b (float): Initial intercept value (default 0.0).
        random_seed (int): Seed for reproducible data shuffling (default 42).
        tolerance (float): Threshold to monitor stabilization (|L_e - L_{e-1}| < tolerance).

    Returns:
        tuple: (w, b, history)
            w (float): Final learned weight/slope.
            b (float): Final learned bias/intercept.
            history (dict): Dictionary tracking epochs, parameter paths, loss history,
                            and intra-step update tracking.
    """
    n = len(X_train)
    rng = np.random.default_rng(random_seed)

    w = float(initial_w)
    b = float(initial_b)

    epoch_history = []
    w_epoch_history = []
    b_epoch_history = []
    loss_epoch_history = []

    # Also store step-by-step updates across all samples
    step_loss_history = []
    step_w_history = []
    step_b_history = []

    # Initial state before training (Epoch 0)
    initial_preds = predict(X_train, w, b)
    initial_loss = calculate_mse(y_train, initial_preds)

    epoch_history.append(0)
    w_epoch_history.append(w)
    b_epoch_history.append(b)
    loss_epoch_history.append(initial_loss)

    converged_epoch = None
    total_steps = 0

    for epoch in range(1, epochs + 1):
        # 1. Shuffle training data indices at the start of each epoch
        shuffled_indices = rng.permutation(n)

        # 2. Iterate through training instances one by one
        for idx in shuffled_indices:
            xi = X_train[idx]
            yi = y_train[idx]

            # Compute prediction for single instance
            y_hat_i = w * xi + b

            # Compute error for single instance
            error_i = yi - y_hat_i

            # Compute gradients from this single observation
            # dw = -2 * x_i * (y_i - y_hat_i)
            # db = -2 * (y_i - y_hat_i)
            dw_i = -2.0 * xi * error_i
            db_i = -2.0 * error_i

            # Immediately update parameters
            w = w - learning_rate * dw_i
            b = b - learning_rate * db_i

            total_steps += 1
            step_w_history.append(w)
            step_b_history.append(b)
            # Instantaneous squared error on this sample
            step_loss_history.append(error_i ** 2)

        # 3. Compute full training set MSE at the end of the epoch
        epoch_preds = predict(X_train, w, b)
        epoch_loss = calculate_mse(y_train, epoch_preds)

        # 4. Check convergence condition
        prev_loss = loss_epoch_history[-1]
        loss_diff = abs(prev_loss - epoch_loss)
        if converged_epoch is None and loss_diff < tolerance:
            converged_epoch = epoch

        # Record epoch-level statistics
        epoch_history.append(epoch)
        w_epoch_history.append(w)
        b_epoch_history.append(b)
        loss_epoch_history.append(epoch_loss)

    history = {
        "epochs": epoch_history,
        "w": w_epoch_history,
        "b": b_epoch_history,
        "loss": loss_epoch_history,
        "step_loss": step_loss_history,
        "step_w": step_w_history,
        "step_b": step_b_history,
        "converged_epoch": converged_epoch,
        "final_loss": loss_epoch_history[-1],
        "total_updates": total_steps,
        "total_sample_evaluations": total_steps
    }

    return w, b, history
