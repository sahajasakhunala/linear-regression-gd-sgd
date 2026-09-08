"""
Visualization Module
====================
Course: Fundamentals of Artificial Intelligence
Project: Linear Regression Using Gradient Descent and Stochastic Gradient Descent

This module generates all 6 required educational visualizations:
1. Dataset Scatter Plot (Train vs Test split)
2. Fitted Regression Lines (GD vs SGD lines against data points)
3. Batch GD Loss Curve (Loss vs Iterations)
4. SGD Loss Curve (Average Loss vs Epochs)
5. Comparative GD vs SGD Loss Curves
6. Actual vs Predicted Scores on Unseen Test Data
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reliable automated plotting
import matplotlib.pyplot as plt
import numpy as np


def setup_plot_style():
    """Apply clean, academic styling to all plots."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["grid.color"] = "#E0E0E0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def plot_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "results/figures/01_dataset_scatter.png"
):
    """
    Graph 1: Scatter plot of raw dataset clearly distinguishing Train vs Test samples.
    """
    setup_plot_style()
    plt.figure(figsize=(9, 6))

    plt.scatter(
        X_train, y_train,
        color="#1f77b4", s=90, edgecolor="black", linewidth=1.2,
        label=f"Training Data (n = {len(X_train)}, 70%)", zorder=3
    )
    plt.scatter(
        X_test, y_test,
        color="#ff7f0e", s=110, marker="s", edgecolor="black", linewidth=1.2,
        label=f"Testing Data (n = {len(X_test)}, 30%)", zorder=4
    )

    # Annotate test samples
    for x, y in zip(X_test, y_test):
        plt.annotate(
            f"({x}, {int(y)})",
            xy=(x, y), xytext=(0, 9),
            textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
            color="#d95f02",
            bbox=dict(boxstyle="round,pad=0.2", fc="#fff2e6", ec="#ff7f0e", alpha=0.8)
        )

    plt.title("Study Hours vs. Exam Scores (Train/Test Partition)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Hours Studied", fontsize=12)
    plt.ylabel("Exam Score", fontsize=12)
    plt.xlim(0.5, 8.5)
    plt.ylim(30, 95)
    plt.grid(True)
    plt.legend(loc="upper left", frameon=True, fancybox=True, shadow=True, fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_regression_lines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w_gd: float,
    b_gd: float,
    w_sgd: float,
    b_sgd: float,
    save_path: str = "results/figures/02_regression_lines.png"
):
    """
    Graph 2: Regression lines for both GD and SGD alongside train and test points.
    """
    setup_plot_style()
    plt.figure(figsize=(10, 6.5))

    # Data points
    plt.scatter(
        X_train, y_train,
        color="#1f77b4", s=80, alpha=0.9, edgecolor="black",
        label="Training Observations (10)", zorder=3
    )
    plt.scatter(
        X_test, y_test,
        color="#ff7f0e", s=110, marker="s", edgecolor="black",
        label="Test Observations (5)", zorder=4
    )

    # Regression lines
    x_line = np.linspace(0.8, 8.2, 200)
    y_line_gd = w_gd * x_line + b_gd
    y_line_sgd = w_sgd * x_line + b_sgd

    plt.plot(
        x_line, y_line_gd,
        color="#2ca02c", linestyle="-", linewidth=2.5,
        label=f"GD Line:  ŷ = {w_gd:.2f}x + {b_gd:.2f}", zorder=2
    )
    plt.plot(
        x_line, y_line_sgd,
        color="#d62728", linestyle="--", linewidth=2.2,
        label=f"SGD Line: ŷ = {w_sgd:.2f}x + {b_sgd:.2f}", zorder=2
    )

    plt.title("Fitted Linear Regression Models: Batch GD vs. SGD", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Hours Studied (x)", fontsize=12)
    plt.ylabel("Exam Score (y)", fontsize=12)
    plt.xlim(0.5, 8.5)
    plt.ylim(30, 95)
    plt.grid(True)
    plt.legend(loc="upper left", frameon=True, fancybox=True, shadow=True, fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_gd_loss(
    gd_history: dict,
    save_path: str = "results/figures/03_gd_loss_curve.png"
):
    """
    Graph 3: Batch Gradient Descent MSE loss curve over iterations.
    """
    setup_plot_style()
    plt.figure(figsize=(9, 5.5))

    iterations = gd_history["iterations"]
    losses = gd_history["loss"]

    plt.plot(
        iterations, losses,
        color="#2ca02c", linewidth=2.2,
        label="Batch GD Training Loss"
    )

    plt.title("Batch Gradient Descent: MSE Loss vs. Iterations", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.grid(True)

    # Inset / zoom annotation for early descent
    final_loss = losses[-1]
    plt.annotate(
        f"Final Loss: {final_loss:.4f}\nat iter {iterations[-1]}",
        xy=(iterations[-1], final_loss),
        xytext=(-120, 40),
        textcoords="offset points",
        arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#2ca02c")
    )

    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_sgd_loss(
    sgd_history: dict,
    save_path: str = "results/figures/04_sgd_loss_curve.png"
):
    """
    Graph 4: Stochastic Gradient Descent average MSE loss curve over epochs.
    """
    setup_plot_style()
    plt.figure(figsize=(9, 5.5))

    epochs = sgd_history["epochs"]
    losses = sgd_history["loss"]

    plt.plot(
        epochs, losses,
        color="#d62728", linewidth=1.8, marker="o", markersize=3,
        label="SGD Epoch Average Loss"
    )

    plt.title("Stochastic Gradient Descent: Average MSE Loss vs. Epochs", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.grid(True)

    final_loss = losses[-1]
    plt.annotate(
        f"Final Loss: {final_loss:.4f}\nat epoch {epochs[-1]}",
        xy=(epochs[-1], final_loss),
        xytext=(-120, 40),
        textcoords="offset points",
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#d62728")
    )

    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_comparative_loss(
    gd_history: dict,
    sgd_history: dict,
    save_path: str = "results/figures/05_gd_vs_sgd_loss.png"
):
    """
    Graph 5: Comparative plot illustrating the distinct convergence trajectories
    of Batch GD (smooth, deterministic) vs. SGD (stochastic, fluctuating).
    """
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Batch GD trajectory
    ax1.plot(gd_history["iterations"], gd_history["loss"], color="#2ca02c", linewidth=2.2)
    ax1.set_title("Batch GD (1,000 Iterations)\nFull-Batch Gradients: Smooth Monotonic Trajectory", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Iteration", fontsize=10)
    ax1.set_ylabel("Training MSE Loss", fontsize=10)
    ax1.grid(True)
    ax1.set_ylim(0, max(gd_history["loss"][:10]) * 1.05)

    # Subplot 2: SGD trajectory
    ax2.plot(sgd_history["epochs"], sgd_history["loss"], color="#d62728", linewidth=1.8, marker=".", markersize=4)
    ax2.set_title("Stochastic GD (100 Epochs)\nSingle-Sample Gradients: High-Variance Fluctuations", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Epoch Average MSE Loss", fontsize=10)
    ax2.grid(True)
    ax2.set_ylim(0, max(sgd_history["loss"][:10]) * 1.05)

    plt.suptitle("Optimization Trajectory Comparison: Batch GD vs. Stochastic GD", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(
    X_test: np.ndarray,
    y_test: np.ndarray,
    gd_preds: np.ndarray,
    sgd_preds: np.ndarray,
    save_path: str = "results/figures/06_actual_vs_predicted_test.png"
):
    """
    Graph 6: Actual vs Predicted performance on unseen test data.
    """
    setup_plot_style()
    plt.figure(figsize=(10, 6))

    indices = np.arange(len(X_test))
    bar_width = 0.25

    # Grouped bar comparison for the 5 unseen test samples
    labels = [f"{x} hrs" for x in X_test]

    plt.bar(indices - bar_width, y_test, width=bar_width, color="#1f77b4", edgecolor="black", label="Actual Score")
    plt.bar(indices, gd_preds, width=bar_width, color="#2ca02c", edgecolor="black", label="GD Prediction")
    plt.bar(indices + bar_width, sgd_preds, width=bar_width, color="#d62728", edgecolor="black", label="SGD Prediction")

    # Value labels on top of bars
    for i in indices:
        plt.text(i - bar_width, y_test[i] + 1.0, f"{int(y_test[i])}", ha="center", fontsize=8, fontweight="bold")
        plt.text(i, gd_preds[i] + 1.0, f"{gd_preds[i]:.1f}", ha="center", fontsize=8)
        plt.text(i + bar_width, sgd_preds[i] + 1.0, f"{sgd_preds[i]:.1f}", ha="center", fontsize=8)

    plt.title("Unseen Test Set Performance: Actual vs. Predicted Scores", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Student Study Hours (Test Sample)", fontsize=12)
    plt.ylabel("Exam Score", fontsize=12)
    plt.xticks(indices, labels, fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
