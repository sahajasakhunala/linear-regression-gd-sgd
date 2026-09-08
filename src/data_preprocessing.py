"""
Data Preprocessing Module
=========================
Course: Fundamentals of Artificial Intelligence
Project: Linear Regression Using Gradient Descent and Stochastic Gradient Descent

This module handles:
1. Loading the dataset from CSV using Pandas.
2. Dataset inspection (missing values, data types, duplicates, summary statistics).
3. Train/Test splitting (70% train = 10 samples, 30% test = 5 samples) with fixed seed.
4. Educational feature scaling analysis and zero-data-leakage standardization utilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(file_path: str = "data/study_hours_exam_scores.csv") -> pd.DataFrame:
    """
    Load the study hours vs exam scores dataset from CSV.

    Parameters:
        file_path (str): Path to CSV dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Perform thorough data quality inspection:
    - Missing values
    - Data types
    - Duplicate rows
    - Descriptive statistics

    Parameters:
        df (pd.DataFrame): Input dataframe.

    Returns:
        dict: Inspection summary dictionary.
    """
    missing_vals = df.isnull().sum().to_dict()
    data_types = df.dtypes.astype(str).to_dict()
    num_duplicates = int(df.duplicated().sum())
    summary_stats = df.describe()

    return {
        "missing_values": missing_vals,
        "data_types": data_types,
        "duplicates": num_duplicates,
        "summary_statistics": summary_stats,
        "total_samples": len(df),
    }


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42
):
    """
    Divide the dataset into training and testing sets.

    Parameters:
        df (pd.DataFrame): Complete dataset.
        test_size (float): Proportion of dataset for test set (default 0.3 = 30%).
        random_state (int): Seed for reproducible pseudorandom splitting.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, train_df, test_df)
               X and y are returned as float64 1D NumPy arrays.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    # Sort indices or reset index for clear presentation
    train_df = train_df.sort_values(by="Hours_Studied").reset_index(drop=True)
    test_df = test_df.sort_values(by="Hours_Studied").reset_index(drop=True)

    X_train = train_df["Hours_Studied"].to_numpy(dtype=np.float64)
    y_train = train_df["Exam_Score"].to_numpy(dtype=np.float64)

    X_test = test_df["Hours_Studied"].to_numpy(dtype=np.float64)
    y_test = test_df["Exam_Score"].to_numpy(dtype=np.float64)

    return X_train, y_train, X_test, y_test, train_df, test_df


class ZeroLeakageStandardScaler:
    """
    Educational Feature Standardizer (Z-score normalization).

    Calculates mean and standard deviation strictly from the training set:
        mu = mean(X_train)
        sigma = std(X_train)
        z = (X - mu) / sigma

    Crucially, test data is transformed using ONLY the training statistics to
    prevent data leakage.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X_train: np.ndarray):
        """Compute mean and std from training set only."""
        self.mean_ = np.mean(X_train)
        self.std_ = np.std(X_train, ddof=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned training scaling parameters to any partition."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler must be fitted before transforming.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit to training data and return scaled training data."""
        return self.fit(X_train).transform(X_train)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Revert scaled features back to original units."""
        return X_scaled * self.std_ + self.mean_


def feature_scaling_analysis() -> str:
    """
    Provides an educational explanation regarding feature scaling for this project.

    Returns:
        str: Detailed explanation text.
    """
    explanation = (
        "FEATURE SCALING ANALYSIS & ZERO DATA LEAKAGE:\n"
        "1. When is scaling required in Gradient Descent?\n"
        "   - When multiple features have vastly different numerical ranges (e.g.,\n"
        "     Income: $10,000 to $200,000 vs. Age: 18 to 70).\n"
        "   - Different scales distort the MSE loss contour into elongated ellipses,\n"
        "     causing gradients along the larger scale to dominate and oscillate,\n"
        "     often requiring minuscule learning rates.\n"
        "\n"
        "2. Is scaling strictly necessary for this dataset?\n"
        "   - No. Here we have a single feature: Hours Studied in [1.0, 8.0].\n"
        "   - The range is naturally compact, well-behaved, and already centered near unity.\n"
        "   - With a moderate learning rate (alpha = 0.01), both GD and SGD converge\n"
        "     reliably without numerical overflow or oscillation.\n"
        "   - Keeping the raw scale preserves the direct physical interpretability of\n"
        "     model parameters:\n"
        "       * w (slope): points gained per hour of study (~7.6-7.8 marks/hour)\n"
        "       * b (intercept): estimated baseline score with 0 study hours (~28 marks)\n"
        "\n"
        "3. Zero Data Leakage Rule:\n"
        "   - If standardization is applied, parameters (mean and std) MUST be computed\n"
        "     strictly on the training data (10 samples) and never on the test set (5 samples).\n"
        "   - Using test data to compute mean/std leaks future distribution information into\n"
        "     the model, violating the fundamental premise of statistical learning."
    )
    return explanation
