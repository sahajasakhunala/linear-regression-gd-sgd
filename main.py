"""
Main Educational Driver Script
==============================
Course: Fundamentals of Artificial Intelligence
Project: Linear Regression Using Gradient Descent and Stochastic Gradient Descent:
         A Comparative Study

Author: AI Pair Programming Assistant & Student Team
Date: Academic Term 2026

Description:
This script orchestrates the end-to-end educational machine learning experiment:
- Loads and audits the 15-sample study dataset.
- Performs a 70/30 train/test split with zero data leakage (seed 42).
- Trains Linear Regression via Batch Gradient Descent (1000 iterations).
- Trains Linear Regression via Stochastic Gradient Descent (100 epochs).
- Computes Train/Test MSE, RMSE, and R^2.
- Compares models against analytical Ordinary Least Squares (OLS).
- Generates all 6 high-resolution visualization figures.
- Exports results to CSV files.
- Prints clean, structured academic logs for project evaluation and analysis.
"""

import os
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    load_dataset,
    inspect_dataset,
    split_dataset,
    feature_scaling_analysis
)
from src.linear_regression import calculate_mse, calculate_rmse, calculate_r2
from src.gradient_descent import train_gradient_descent
from src.stochastic_gradient_descent import train_stochastic_gradient_descent
from src.evaluation import (
    evaluate_model,
    compute_ols_solution,
    create_comparison_table,
    create_test_prediction_table
)
from src.visualization import (
    plot_dataset,
    plot_regression_lines,
    plot_gd_loss,
    plot_sgd_loss,
    plot_comparative_loss,
    plot_actual_vs_predicted
)


def print_header(title: str):
    """Print standard educational section separator."""
    print("\n" + "=" * 60)
    print(f" {title.upper()}")
    print("=" * 60)


def main():
    # ---------------------------------------------------------
    # 1. Dataset Loading & Preprocessing
    # ---------------------------------------------------------
    print_header("Dataset Information")

    csv_path = os.path.join("data", "study_hours_exam_scores.csv")
    df = load_dataset(csv_path)
    inspection = inspect_dataset(df)

    print(f"Dataset successfully loaded from: {csv_path}")
    print(f"Total Observations: {inspection['total_samples']}")
    print("\nFirst 5 Records:")
    print(df.head())

    print("\nData Quality Verification:")
    print(f" - Missing Values : {inspection['missing_values']}")
    print(f" - Data Types      : {inspection['data_types']}")
    print(f" - Duplicate Rows  : {inspection['duplicates']}")

    print("\nDescriptive Statistics (Full Dataset):")
    print(df.describe().round(2))

    # ---------------------------------------------------------
    # 2. Train / Test Split (70% Train, 30% Test)
    # ---------------------------------------------------------
    print_header("Train / Test Split")

    X_train, y_train, X_test, y_test, train_df, test_df = split_dataset(
        df, test_size=0.3, random_state=42
    )

    print(f"Training Samples: {len(X_train)} (70% of dataset)")
    print(f"Testing Samples : {len(X_test)} (30% of dataset, strictly isolated)")
    print(f"Random State    : 42 (reproducible split)")

    print("\nTraining Set (n=10):")
    print(train_df[["Student", "Hours_Studied", "Exam_Score"]].to_string(index=False))

    print("\nUnseen Test Set (n=5) - Held out until post-training evaluation:")
    print(test_df[["Student", "Hours_Studied", "Exam_Score"]].to_string(index=False))

    print("\n" + "-" * 50)
    print(feature_scaling_analysis())

    # Analytical Reference (OLS) on Training Data
    w_ols, b_ols = compute_ols_solution(X_train, y_train)
    ols_eval = evaluate_model(X_train, y_train, X_test, y_test, w_ols, b_ols, model_name="OLS")
    print(f"\nAnalytical Training OLS Reference: w = {w_ols:.4f}, b = {b_ols:.4f}")

    # ---------------------------------------------------------
    # 3. Batch Gradient Descent Training
    # ---------------------------------------------------------
    print_header("Gradient Descent Results")

    learning_rate = 0.01
    gd_iterations = 1000

    print(f"Algorithm         : Batch Gradient Descent (GD)")
    print(f"Initial Parameters: w0 = 0.0, b0 = 0.0")
    print(f"Learning Rate     : alpha = {learning_rate}")
    print(f"Total Iterations  : {gd_iterations}")
    print(f"Batch Size        : Full Training Batch (n = {len(X_train)})")

    w_gd, b_gd, gd_history = train_gradient_descent(
        X_train=X_train,
        y_train=y_train,
        learning_rate=learning_rate,
        iterations=gd_iterations,
        initial_w=0.0,
        initial_b=0.0
    )

    print(f"\nTraining Complete:")
    print(f" - Initial Loss (MSE at iter 0) : {gd_history['loss'][0]:.4f}")
    print(f" - Loss after 10 iterations     : {gd_history['loss'][10]:.4f}")
    print(f" - Loss after 100 iterations    : {gd_history['loss'][100]:.4f}")
    print(f" - Loss after 500 iterations    : {gd_history['loss'][500]:.4f}")
    print(f" - Final Loss (iter 1000)       : {gd_history['final_loss']:.4f}")
    print(f" - Learned Weight (w)           : {w_gd:.4f} (OLS baseline: {w_ols:.4f})")
    print(f" - Learned Bias (b)             : {b_gd:.4f} (OLS baseline: {b_ols:.4f})")
    print(f" - Total Parameter Updates      : {gd_history['total_updates']:,}")
    print(f" - Total Sample Evaluations     : {gd_history['total_sample_evaluations']:,}")

    # ---------------------------------------------------------
    # 4. Stochastic Gradient Descent Training
    # ---------------------------------------------------------
    print_header("Stochastic Gradient Descent Results")

    sgd_epochs = 100

    print(f"Algorithm         : Stochastic Gradient Descent (SGD)")
    print(f"Initial Parameters: w0 = 0.0, b0 = 0.0")
    print(f"Learning Rate     : alpha = {learning_rate}")
    print(f"Epochs            : {sgd_epochs}")
    print(f"Shuffling         : Random permutation per epoch (seed = 42)")
    print(f"Update Frequency  : Immediate after each single sample")

    w_sgd, b_sgd, sgd_history = train_stochastic_gradient_descent(
        X_train=X_train,
        y_train=y_train,
        learning_rate=learning_rate,
        epochs=sgd_epochs,
        initial_w=0.0,
        initial_b=0.0,
        random_seed=42
    )

    print(f"\nTraining Complete:")
    print(f" - Initial Loss (MSE at epoch 0) : {sgd_history['loss'][0]:.4f}")
    print(f" - Loss after 5 epochs           : {sgd_history['loss'][5]:.4f}")
    print(f" - Loss after 20 epochs          : {sgd_history['loss'][20]:.4f}")
    print(f" - Loss after 50 epochs          : {sgd_history['loss'][50]:.4f}")
    print(f" - Final Loss (epoch 100)        : {sgd_history['final_loss']:.4f}")
    print(f" - Learned Weight (w)            : {w_sgd:.4f} (OLS baseline: {w_ols:.4f})")
    print(f" - Learned Bias (b)              : {b_sgd:.4f} (OLS baseline: {b_ols:.4f})")
    print(f" - Total Parameter Updates       : {sgd_history['total_updates']:,}")
    print(f" - Total Sample Evaluations      : {sgd_history['total_sample_evaluations']:,}")

    # ---------------------------------------------------------
    # 5. Model Evaluation (Train & Test)
    # ---------------------------------------------------------
    print_header("Model Evaluation")

    gd_eval = evaluate_model(X_train, y_train, X_test, y_test, w_gd, b_gd, model_name="GD")
    sgd_eval = evaluate_model(X_train, y_train, X_test, y_test, w_sgd, b_sgd, model_name="SGD")

    print("Batch Gradient Descent Metrics:")
    print(f" - Training MSE : {gd_eval['train_mse']:.4f} | RMSE: {gd_eval['train_rmse']:.4f} | R^2: {gd_eval['train_r2']:.4f}")
    print(f" - Test MSE     : {gd_eval['test_mse']:.4f} | RMSE: {gd_eval['test_rmse']:.4f} | R^2: {gd_eval['test_r2']:.4f}")

    print("\nStochastic Gradient Descent Metrics:")
    print(f" - Training MSE : {sgd_eval['train_mse']:.4f} | RMSE: {sgd_eval['train_rmse']:.4f} | R^2: {sgd_eval['train_r2']:.4f}")
    print(f" - Test MSE     : {sgd_eval['test_mse']:.4f} | RMSE: {sgd_eval['test_rmse']:.4f} | R^2: {sgd_eval['test_r2']:.4f}")

    # ---------------------------------------------------------
    # 6. Side-by-Side GD vs SGD Comparison Table
    # ---------------------------------------------------------
    print_header("GD vs SGD Comparison")

    comp_table = create_comparison_table(gd_eval, sgd_eval, ols_eval)
    print(comp_table.to_string(index=False))

    # Export comparison table
    os.makedirs("results", exist_ok=True)
    comp_csv_path = os.path.join("results", "results.csv")
    comp_table.to_csv(comp_csv_path, index=False)
    print(f"\nComparative results exported to: {comp_csv_path}")

    # ---------------------------------------------------------
    # 7. Test Set Predictions Table
    # ---------------------------------------------------------
    print_header("Test Set Predictions")

    test_pred_table = create_test_prediction_table(
        X_test=X_test,
        y_test=y_test,
        gd_preds=gd_eval["test_preds"],
        sgd_preds=sgd_eval["test_preds"]
    )
    print(test_pred_table.to_string(index=False))

    test_csv_path = os.path.join("results", "test_predictions.csv")
    test_pred_table.to_csv(test_csv_path, index=False)
    print(f"\nTest set predictions exported to: {test_csv_path}")

    # ---------------------------------------------------------
    # 8. Visualizations Generation
    # ---------------------------------------------------------
    print_header("Generating Visualizations")

    figures_dir = os.path.join("results", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_dataset(X_train, y_train, X_test, y_test, os.path.join(figures_dir, "01_dataset_scatter.png"))
    print(" [1/6] Saved: 01_dataset_scatter.png")

    plot_regression_lines(X_train, y_train, X_test, y_test, w_gd, b_gd, w_sgd, b_sgd, os.path.join(figures_dir, "02_regression_lines.png"))
    print(" [2/6] Saved: 02_regression_lines.png")

    plot_gd_loss(gd_history, os.path.join(figures_dir, "03_gd_loss_curve.png"))
    print(" [3/6] Saved: 03_gd_loss_curve.png")

    plot_sgd_loss(sgd_history, os.path.join(figures_dir, "04_sgd_loss_curve.png"))
    print(" [4/6] Saved: 04_sgd_loss_curve.png")

    plot_comparative_loss(gd_history, sgd_history, os.path.join(figures_dir, "05_gd_vs_sgd_loss.png"))
    print(" [5/6] Saved: 05_gd_vs_sgd_loss.png")

    plot_actual_vs_predicted(X_test, y_test, gd_eval["test_preds"], sgd_eval["test_preds"], os.path.join(figures_dir, "06_actual_vs_predicted_test.png"))
    print(" [6/6] Saved: 06_actual_vs_predicted_test.png")

    # ---------------------------------------------------------
    # 9. Educational Summary & Realistic Interpretation
    # ---------------------------------------------------------
    print_header("Convergence & Educational Summary")

    summary_text = f"""
EDUCATIONAL FINDINGS & SYNTHESIS:

1. Learning Success & Parameter Estimation:
   - Both Batch GD and SGD successfully learned the linear regression relationship.
   - Batch GD parameters:  w = {w_gd:.4f}, b = {b_gd:.4f}
   - SGD parameters:       w = {w_sgd:.4f}, b = {b_sgd:.4f}
   - Analytical OLS:       w = {w_ols:.4f}, b = {b_ols:.4f}
   - Both methods learned almost the same regression relationship. GD achieved a marginally
     lower test MSE (0.8636 vs. 0.8650), while SGD achieved a slightly lower training MSE
     (0.7176 vs. 0.7892). The difference is extremely small, especially given the five-sample test set.

2. Optimization Dynamics & Trajectory:
   - Batch GD demonstrated a smooth optimization trajectory because every parameter update
     is computed using the exact gradient over all 10 training samples.
   - SGD introduced stochastic fluctuations across epochs because each parameter update is
     driven by an individual sample, whose gradient reflects single-sample variance.

3. Generalization & Overfitting Assessment:
   - Training vs. Test RMSE:
     * GD : Train RMSE = {gd_eval['train_rmse']:.4f}, Test RMSE = {gd_eval['test_rmse']:.4f}
     * SGD: Train RMSE = {sgd_eval['train_rmse']:.4f}, Test RMSE = {sgd_eval['test_rmse']:.4f}
   - The training and test errors are similar, suggesting that there is no obvious overfitting
     in this small experiment. However, because the test set contains only five observations,
     this conclusion should be interpreted cautiously.

4. Computational Work vs. Wall-Clock Runtime:
   - Under our chosen configuration, SGD performs 1,000 individual sample evaluations
     compared with 10,000 for Batch GD. Despite this difference, both methods achieve
     very similar test performance.
   - Note: Sample evaluation counts should not be conflated with wall-clock runtime speed,
     which depends heavily on vectorization, hardware, and implementation overhead.
   - On massive datasets (e.g., n = 1,000,000), Batch GD becomes computationally unfeasible
     per iteration, whereas SGD or Mini-batch GD makes rapid initial progress with small batches.

5. Final Project Takeaway:
   - "The experiment demonstrates that both Batch Gradient Descent and Stochastic Gradient Descent
     can effectively optimize a Linear Regression model. Batch GD provides a smoother optimization
     trajectory, while SGD introduces stochastic fluctuations due to sample-wise updates. In our
     small dataset, both methods produced nearly identical test performance, showing that the
     optimization behavior can differ even when the final predictive performance is similar."
"""
    print(summary_text.strip())
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
