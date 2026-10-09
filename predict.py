"""
Interactive Prediction Demonstration Script
===========================================
Course: Foundations of Artificial Intelligence
Project: Linear Regression Using Gradient Descent and Stochastic Gradient Descent

This script allows you to demonstrate model predictions on demand:
- Shows the learned linear equations: y_hat = w * x + b
- Predicts exam score for any input study hours using Batch GD and SGD
- Supports command-line arguments (e.g., python predict.py 5.5) or interactive input
"""

import sys
import numpy as np

# Learned parameters from training
# Batch Gradient Descent (1000 iterations, alpha = 0.01)
W_GD = 7.7822
B_GD = 28.2121

# Stochastic Gradient Descent (100 epochs, alpha = 0.01)
W_SGD = 7.7156
B_SGD = 28.5394

# Analytical OLS Reference
W_OLS = 7.6379
B_OLS = 28.9753


def predict_score(hours: float):
    """Calculate predicted score using GD and SGD equations."""
    gd_score = W_GD * hours + B_GD
    sgd_score = W_SGD * hours + B_SGD
    ols_score = W_OLS * hours + B_OLS
    return gd_score, sgd_score, ols_score


def display_prediction(hours: float):
    """Print clean step-by-step prediction for given study hours."""
    gd_pred, sgd_pred, ols_pred = predict_score(hours)
    
    print("\n" + "-" * 55)
    print(f" Input: Hours Studied = {hours:.2f} hrs")
    print("-" * 55)
    print(f" 1. Batch GD  : y_hat = ({W_GD:.4f} * {hours:.2f}) + {B_GD:.4f} = {gd_pred:.2f} marks")
    print(f" 2. SGD       : y_hat = ({W_SGD:.4f} * {hours:.2f}) + {B_SGD:.4f} = {sgd_pred:.2f} marks")
    print(f" 3. OLS (Ref) : y_hat = ({W_OLS:.4f} * {hours:.2f}) + {B_OLS:.4f} = {ols_pred:.2f} marks")
    print("-" * 55)


def main():
    print("=" * 55)
    print(" LINEAR REGRESSION MODEL PREDICTION DEMO")
    print(" Subject: Foundations of Artificial Intelligence")
    print("=" * 55)
    print("Learned Regression Equations:")
    print(f" - Batch GD : y_hat = {W_GD:.4f} * Hours + {B_GD:.4f}")
    print(f" - SGD      : y_hat = {W_SGD:.4f} * Hours + {B_SGD:.4f}")
    print("=" * 55)

    # If arguments are passed on command line: e.g. python predict.py 5 6.5
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            try:
                hours = float(arg)
                display_prediction(hours)
            except ValueError:
                print(f"Invalid input: '{arg}'. Please pass numeric values.")
        return

    # Default: Show predictions on test set samples
    print("\nPrecomputed Predictions on Test Set (5 unseen students):")
    test_cases = [
        ("Student 04", 1.0, 35.0),
        ("Student 07", 3.5, 55.0),
        ("Student 08", 5.5, 71.0),
        ("Student 13", 6.5, 78.0),
        ("Student 10", 7.5, 85.0),
    ]
    print(f"{'Student':<12} {'Hours':<8} {'Actual':<8} {'GD Pred':<10} {'SGD Pred':<10} {'Difference'}")
    print("-" * 60)
    for student, hrs, actual in test_cases:
        gd_p, sgd_p, _ = predict_score(hrs)
        print(f"{student:<12} {hrs:<8.1f} {actual:<8.1f} {gd_p:<10.2f} {sgd_p:<10.2f} {abs(gd_p - actual):.2f}")

    print("\n" + "=" * 55)
    print(" Interactive Prediction Mode")
    print(" Enter study hours to test any value (or 'q' to quit):")
    print("=" * 55)

    while True:
        try:
            val = input("\nEnter Study Hours: ").strip()
            if not val or val.lower() in ('q', 'quit', 'exit'):
                print("Exiting prediction demo.")
                break
            hours = float(val)
            display_prediction(hours)
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nExiting prediction demo.")
            break


if __name__ == "__main__":
    main()
