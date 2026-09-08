# Linear Regression Using Gradient Descent and Stochastic Gradient Descent: A Comparative Study

Course: Fundamentals of Artificial Intelligence  
Project Type: Educational Machine Learning Laboratory and Comparative Analysis  
Repository: linear-regression-gd-sgd  

---

## 1. Executive Summary and Problem Statement

This educational project provides a transparent, foundational examination of how Linear Regression learns optimal parameters from empirical observations. Optimization lies at the heart of artificial intelligence and machine learning. In this project, two foundational optimization paradigms are designed and implemented entirely from scratch using Python and NumPy:

1. Batch Gradient Descent (GD)
2. Stochastic Gradient Descent (SGD)

The study is conducted on a canonical 15-observation dataset tracking the relationship between Hours Studied (input feature X) and Exam Score (target variable y). The implementation strictly avoids high-level machine learning abstractions such as scikit-learn LinearRegression or SGDRegressor for model fitting. Every forward prediction, residual error, analytical gradient vector, and parameter update is calculated explicitly from first principles to provide maximum educational clarity and mathematical rigor.

---

## 2. Project Objectives

The core objectives of this study are:

1. To formulate the univariate linear regression hypothesis mathematically.
2. To derive the Mean Squared Error (MSE) objective function and compute its analytical partial derivatives with respect to the slope (weight) and intercept (bias).
3. To implement Batch Gradient Descent from scratch, updating parameters simultaneously across the complete training batch.
4. To implement Stochastic Gradient Descent from scratch, shuffling data each epoch and updating parameters iteratively per individual observation.
5. To enforce strict machine learning methodologies by partitioning the data into 70% training and 30% testing subsets with zero data leakage.
6. To evaluate both models on unseen test observations using standard statistical metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the Coefficient of Determination (R-squared).
7. To compare the learned parameters against the closed-form Ordinary Least Squares (OLS) analytical solution.
8. To generate high-resolution, publication-quality visualizations depicting the dataset partition, fitted regression lines, optimization loss curves, and actual versus predicted test performance.
9. To analyze convergence dynamics, computational work, and practical trade-offs without making exaggerated claims of algorithmic superiority.

---

## 3. Dataset Description

The dataset consists of 15 student records measuring the impact of study hours on final examination performance.

| Student | Hours Studied (X) | Exam Score (y) | Experimental Partition |
| :---: | :---: | :---: | :---: |
| 1 | 1.0 | 35 | Test Set (Unseen) |
| 2 | 1.5 | 39 | Training Set |
| 3 | 2.0 | 44 | Training Set |
| 4 | 2.5 | 48 | Training Set |
| 5 | 3.0 | 52 | Training Set |
| 6 | 3.5 | 55 | Test Set (Unseen) |
| 7 | 4.0 | 61 | Training Set |
| 8 | 4.5 | 64 | Training Set |
| 9 | 5.0 | 68 | Training Set |
| 10 | 5.5 | 71 | Test Set (Unseen) |
| 11 | 6.0 | 75 | Training Set |
| 12 | 6.5 | 78 | Test Set (Unseen) |
| 13 | 7.0 | 82 | Training Set |
| 14 | 7.5 | 85 | Test Set (Unseen) |
| 15 | 8.0 | 89 | Training Set |

### Summary Statistics (Full Dataset)
- Total observations: 15
- Input Feature (Hours Studied): Minimum = 1.0, Maximum = 8.0, Mean = 4.50, Standard Deviation = 2.24
- Target Variable (Exam Score): Minimum = 35.0, Maximum = 89.0, Mean = 63.07, Standard Deviation = 17.20
- Data Quality Audit: Missing values = 0, duplicate rows = 0, data types = float64 / int64.

---

## 4. Train / Test Partitioning and Zero Data Leakage

A common vulnerability in machine learning experiments is data leakage, which occurs when information from outside the training dataset is inadvertently utilized to fit the model or compute scaling parameters.

### Partitioning Scheme
The dataset is divided using an approximate 70% train / 30% test split via scikit-learn train_test_split with random_state=42:
- Training Set: 10 observations (70%)
- Testing Set: 5 observations (30%)

### Training Partition (n = 10)
- Hours Studied: [5.0, 2.0, 1.5, 8.0, 3.0, 4.5, 6.0, 7.0, 2.5, 4.0]
- Exam Scores:   [68.0, 44.0, 39.0, 89.0, 52.0, 64.0, 75.0, 82.0, 48.0, 61.0]

### Unseen Test Partition (n = 5)
The 5 held-out test students are:

| Student Hours (X_test) | Actual Exam Score (y_test) |
| :---: | :---: |
| 1.0 | 35.0 |
| 3.5 | 55.0 |
| 5.5 | 71.0 |
| 6.5 | 78.0 |
| 7.5 | 85.0 |

### Data Leakage Safeguards
1. Optimization Isolation: The test samples are completely quarantined during training. Neither Batch GD nor SGD computes gradients or residuals on these points while optimizing w and b.
2. Independent Evaluation: Post-training evaluation on the test partition represents the first and only time the learned hypothesis interacts with these 5 observations.

---

## 5. Mathematical Foundations

### 5.1 Linear Regression Model
The univariate linear model defines the mapping from input study hours x to predicted exam score y_hat:

y_hat_i = w * x_i + b

where:
- w is the slope / weight, representing the marginal gain in exam points per additional hour studied.
- b is the intercept / bias, representing the expected baseline score when zero hours are studied.
- x_i is the feature value for student i.
- y_hat_i is the model prediction for student i.

Initial parameter values: w_0 = 0.0, b_0 = 0.0.

### 5.2 Loss Function: Mean Squared Error (MSE)
The Mean Squared Error quantifies empirical risk by averaging the squared differences between observed values y_i and predictions y_hat_i over a sample size of n:

MSE = (1 / n) * sum_{i=1}^{n} (y_i - y_hat_i)^2 = (1 / n) * sum_{i=1}^{n} (y_i - (w * x_i + b))^2

Root Mean Squared Error (RMSE) expresses the average deviation in the original measurement units (exam points):

RMSE = sqrt(MSE)

Mathematical Justification for MSE in Linear Regression:
1. Strict Convexity: MSE is a quadratic function of parameters (w, b). Its Hessian matrix is positive semi-definite, guaranteeing an elliptical bowl with a unique global minimum and no local minima traps.
2. Smooth Differentiability: The squared error term is continuous and differentiable over all real numbers, yielding well-behaved analytical gradients.
3. Outlier Sensitivity: By squaring errors, larger discrepancies are penalized disproportionately compared to smaller residuals.
4. Statistical Alignment: Under the classical assumption of normally distributed additive noise, minimizing MSE is mathematically equivalent to Maximum Likelihood Estimation (MLE).

### 5.3 Coefficient of Determination (R-squared)
R-squared quantifies the proportion of variance in the dependent variable explained by the linear model:

R^2 = 1 - (SS_res / SS_tot)

where:
- SS_res = sum_{i=1}^{n} (y_i - y_hat_i)^2 (Residual Sum of Squares)
- SS_tot = sum_{i=1}^{n} (y_i - y_mean)^2 (Total Sum of Squares)

### 5.4 Analytical Gradient Derivation
Applying the calculus chain rule to the MSE loss function:

Let e_i = y_i - (w * x_i + b).
Then MSE = (1 / n) * sum_{i=1}^{n} (e_i)^2.

Differentiating with respect to weight w:
d(MSE)/dw = (1 / n) * sum_{i=1}^{n} 2 * e_i * (d(e_i)/dw)
Since d(e_i)/dw = -x_i:
dw = -(2 / n) * sum_{i=1}^{n} x_i * (y_i - y_hat_i)

Differentiating with respect to bias b:
d(MSE)/db = (1 / n) * sum_{i=1}^{n} 2 * e_i * (d(e_i)/db)
Since d(e_i)/db = -1:
db = -(2 / n) * sum_{i=1}^{n} (y_i - y_hat_i)

---

## 6. Feature Scaling Analysis

In gradient-based optimization involving multiple features, feature scaling (such as z-score standardization) is frequently vital.

### Theoretical Mechanics of Feature Scaling
When multiple input features span disparate numerical magnitudes (e.g., Annual Income in thousands vs. Age in tens), the contours of the MSE loss surface deform into highly elongated, eccentric ellipses. The gradient vector, which points perpendicular to contour lines, does not point directly toward the minimum. Consequently:
- Gradients along large-magnitude features oscillate violently.
- The learning rate must be severely restricted to avoid numerical divergence.
- Standardization transforms elliptical contours into spherical circles, allowing gradient steps to head directly toward the minimum.

### Application to This Dataset
For this educational experiment:
1. Dimensionality: Only one feature (Hours Studied) is present.
2. Compact Domain: The feature values lie in [1.0, 8.0], which is naturally well-conditioned.
3. Optimization Stability: With learning rate alpha = 0.01, both Batch GD and SGD converge reliably without numerical instability or oscillation.
4. Physical Interpretability: Keeping the raw scale allows w and b to maintain direct intuitive meaning:
   - w represents marks per study hour.
   - b represents baseline score with zero study hours.

### Zero-Leakage Standardization Implementation
The project includes an educational standardizer (ZeroLeakageStandardScaler) in src/data_preprocessing.py demonstrating proper protocol:
- Mean and standard deviation are calculated strictly on the 10 training samples:
  mu_train = mean(X_train), sigma_train = std(X_train)
- The test samples are transformed using only mu_train and sigma_train.
- Test statistics are never utilized during scaling.

---

## 7. Optimization Algorithms: GD vs. SGD

### 7.1 Batch Gradient Descent (GD)
Batch Gradient Descent computes the exact gradient of the loss surface over the entire training set (n = 10) before updating parameters:

Algorithm:
1. Initialize w = 0.0, b = 0.0.
2. For iteration = 1 to 1000:
   a. Compute predictions for all 10 training observations: y_hat = w * X_train + b.
   b. Compute error vector: e = y_train - y_hat.
   c. Compute batch gradients:
      dw = -(2 / n) * sum(X_train * e)
      db = -(2 / n) * sum(e)
   d. Update parameters simultaneously:
      w = w - alpha * dw
      b = b - alpha * db
   e. Record training MSE loss.

Characteristics:
- Optimization Path: Smooth, deterministic, monotonic trajectory toward the minimum.
- Computational Work: 1,000 iterations * 10 samples = 10,000 sample evaluations producing 1,000 parameter updates.

### 7.2 Stochastic Gradient Descent (SGD)
Stochastic Gradient Descent approximates the batch gradient using an individual observation (sample-wise gradient) and updates parameters immediately:

Algorithm:
1. Initialize w = 0.0, b = 0.0. Set random seed = 42.
2. For epoch = 1 to 100:
   a. Shuffle the training data indices using a pseudorandom permutation.
   b. For each sample (x_i, y_i) in shuffled training data:
      i.   Compute prediction: y_hat_i = w * x_i + b.
      ii.  Compute error: e_i = y_i - y_hat_i.
      iii. Compute instantaneous gradients:
           dw_i = -2 * x_i * e_i
           db_i = -2 * e_i
      iv.  Update parameters immediately:
           w = w - alpha * dw_i
           b = b - alpha * db_i
   c. Record epoch-average training MSE across all 10 samples.

Characteristics:
- Optimization Path: Stochastic, fluctuating path. Because individual observations contain specific deviations, single-sample gradients are noisy approximations of the full-batch gradient.
- Computational Work: 100 epochs * 10 samples = 1,000 sample evaluations producing 1,000 parameter updates.

---

## 8. Experimental Configuration & Hyperparameters

To ensure a fair, rigorous comparative study:
- Identical initialization: w_0 = 0.0, b_0 = 0.0.
- Identical learning rate: alpha = 0.01.
- Identical training partition (10 samples) and test partition (5 samples).
- Equal total parameter updates: 1,000 updates each.

| Parameter | Batch Gradient Descent (GD) | Stochastic Gradient Descent (SGD) |
| :--- | :---: | :---: |
| Initial Weight (w_0) | 0.0 | 0.0 |
| Initial Bias (b_0) | 0.0 | 0.0 |
| Learning Rate (alpha) | 0.01 | 0.01 |
| Training Samples (n) | 10 | 10 |
| Test Samples | 5 | 5 |
| Iterations / Epochs | 1,000 iterations | 100 epochs |
| Total Parameter Updates | 1,000 updates | 1,000 updates |
| Total Sample Evaluations | 10,000 evaluations | 1,000 evaluations |
| Random Seed | 42 | 42 |

---

## 9. Experimental Results and Numerical Verification

All metrics below are computed dynamically by main.py from the actual trained models:

### 9.1 Comparative Performance Metrics

| Metric | Batch Gradient Descent (GD) | Stochastic Gradient Descent (SGD) | Analytical OLS Baseline |
| :--- | :---: | :---: | :---: |
| Training MSE | 0.7892 | 0.7176 | 0.6823 |
| Training RMSE | 0.8884 | 0.8471 | 0.8260 |
| Training R^2 | 0.9968 | 0.9971 | 0.9973 |
| Test MSE | 0.8636 | 0.8650 | 1.0151 |
| Test RMSE | 0.9293 | 0.9300 | 1.0075 |
| Test R^2 | 0.9973 | 0.9973 | 0.9968 |
| Final Weight (w) | 7.7822 | 7.7156 | 7.6379 |
| Final Bias (b) | 28.2121 | 28.5394 | 28.9753 |

### 9.2 Closed-Form OLS Reference
On the exact 10 training samples isolated by random_state=42:
- Analytical slope: w_ols = 7.63786 (rounded to 7.6379)
- Analytical intercept: b_ols = 28.97531 (rounded to 28.9753)
- GD parameters after 1,000 iterations: w = 7.7822, b = 28.2121
- SGD parameters after 100 epochs: w = 7.7156, b = 28.5394
Both optimization methods reach within 0.15 of the analytical least-squares solution.

### 9.3 Unseen Test Set Predictions (5 Held-Out Samples)

| Hours Studied (X_test) | Actual Score (y_test) | GD Prediction | SGD Prediction | GD Residual (y - y_hat) | SGD Residual (y - y_hat) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1.0 | 35.0 | 35.99 | 36.25 | -0.99 | -1.25 |
| 3.5 | 55.0 | 55.45 | 55.54 | -0.45 | -0.54 |
| 5.5 | 71.0 | 71.01 | 70.97 | -0.01 | +0.03 |
| 6.5 | 78.0 | 78.80 | 78.69 | -0.80 | -0.69 |
| 7.5 | 85.0 | 86.58 | 86.41 | -1.58 | -1.41 |

Across all 5 unseen test samples, the maximum difference between GD and SGD predictions is 0.26 marks (at 1.0 hour).

---

## 10. Visualization Analysis

The project automatically produces 6 high-resolution figures stored in results/figures/:

1. Figure 01: Dataset Partition (01_dataset_scatter.png)
   - Depicts all 15 observations with Hours Studied on the horizontal axis and Exam Score on the vertical axis.
   - Blue circular markers represent the 10 training samples; orange square markers with coordinate callouts denote the 5 held-out test samples.
   - Demonstrates balanced coverage of the feature domain across both partitions.

2. Figure 02: Fitted Regression Lines (02_regression_lines.png)
   - Visualizes the final learned models: GD line (y_hat = 7.78x + 28.21) and SGD line (y_hat = 7.72x + 28.54).
   - Overlays both lines against the training and testing data points, showing that both models capture the underlying empirical trend.

3. Figure 03: Batch GD Loss Curve (03_gd_loss_curve.png)
   - Plots training MSE loss versus iteration index from 0 to 1,000.
   - Highlights an initial steep descent from 4,117.60 down to 121.47 within 10 iterations, followed by smooth asymptotic stabilization toward 0.7892.

4. Figure 04: SGD Loss Curve (04_sgd_loss_curve.png)
   - Plots epoch-average MSE loss versus epoch index from 0 to 100.
   - Displays observable fluctuations across early and middle epochs caused by sample-wise parameter updates before settling near 0.7176.

5. Figure 05: Comparative Optimization Trajectories (05_gd_vs_sgd_loss.png)
   - Side-by-side subplots directly contrasting the smooth, monotonic trajectory of Batch GD with the noisy, fluctuating trajectory of SGD.

6. Figure 06: Actual vs. Predicted Test Performance (06_actual_vs_predicted_test.png)
   - Grouped bar chart comparing Actual Scores, GD Predictions, and SGD Predictions for each of the 5 test students.
   - Shows close alignment across the range of study hours.

---

## 11. Realistic Limitations & Academic Scrutiny

In the spirit of scientific rigor, the following experimental constraints must be acknowledged:

1. Small Sample Size (N = 15): The dataset contains only 15 observations. Small datasets exhibit higher statistical variance.
2. Limited Test Partition (n = 5): A test set of 5 students means each observation accounts for 20% of the test evaluation. Minor variations on a single observation noticeably influence test MSE.
3. Partition Sensitivity: Using a different random seed for train_test_split alters which 10 points form the training set, resulting in slight changes to learned coefficients.
4. Single Feature Modeling: In educational settings, student exam scores depend on multifaceted factors beyond study time (e.g., prior knowledge, sleep quality, attendance).
5. Educational Scope: This experiment is intended to illustrate the mechanics of optimization algorithms and should not be used for real-world academic grading policy.

---

## 12. Final Conclusion & Presentation / Viva Defense Summary

Based strictly on the numerical evidence obtained from the implementation:

1. Did both GD and SGD successfully learn the regression relationship?
   Yes. Both algorithms converged from w_0 = 0.0, b_0 = 0.0 to w in [7.71, 7.78] and b in [28.21, 28.54], closely matching the analytical OLS baseline (w = 7.6379, b = 28.9753) within a margin of less than 0.15.

2. Which method converged more smoothly?
   Batch Gradient Descent. Batch GD exhibited a smooth, predictable loss trajectory because each parameter update averages the true gradient across all 10 training instances. SGD exhibited noticeable epoch-to-epoch fluctuations because each individual sample pulls the parameters in its own idiosyncratic direction.

3. How different were their final predictions?
   Virtually indistinguishable. Across the 5 unseen test samples, the maximum prediction difference between GD and SGD was only 0.26 marks.

4. How did they perform on unseen test data?
   Both methods learned almost the same regression relationship. GD achieved a marginally lower test MSE (0.8636 vs. 0.8650), while SGD achieved a slightly lower training MSE (0.7176 vs. 0.7892). The difference is extremely small, especially given the five-sample test set.
   The training and test errors are similar, suggesting that there is no obvious overfitting in this small experiment. However, because the test set contains only five observations, this conclusion should be interpreted cautiously.

5. What did the experiment demonstrate about GD vs. SGD?
   Under our chosen configuration, SGD performs 1,000 individual sample evaluations compared with 10,000 for Batch GD. Despite this difference, both methods achieve very similar test performance.
   (Note: Sample evaluation counts should not be conflated with wall-clock runtime speed, which depends heavily on vectorization, hardware, and implementation overhead).

6. Why might the results be different on a much larger dataset?
   On massive datasets containing millions of records, Batch GD becomes computationally unfeasible because computing a single parameter update requires an entire pass through memory and disk. Conversely, SGD (and Mini-batch GD) makes rapid early progress and can reach near-optimal parameters after examining only a fraction of the data. Furthermore, in non-convex loss landscapes (such as deep neural networks), the stochastic noise in SGD helps parameters escape shallow saddle points and sharp local minima.

### Recommended Defense Statement for Viva:
"The experiment demonstrates that both Batch Gradient Descent and Stochastic Gradient Descent can effectively optimize a Linear Regression model. Batch GD provides a smoother optimization trajectory, while SGD introduces stochastic fluctuations due to sample-wise updates. In our small dataset, both methods produced nearly identical test performance, showing that the optimization behavior can differ even when the final predictive performance is similar."

---

## 13. Project Directory Structure

```
linear-regression-gd-sgd/
|
|-- data/
|   `-- study_hours_exam_scores.csv       # 15-observation CSV dataset
|
|-- src/
|   |-- __init__.py                       # Package initializer
|   |-- data_preprocessing.py             # Inspection, 70/30 split, zero-leakage standardizer
|   |-- linear_regression.py              # Hypothesis y_hat = wx + b, MSE, RMSE, R^2
|   |-- gradient_descent.py               # From-scratch Batch GD implementation
|   |-- stochastic_gradient_descent.py    # From-scratch SGD implementation with shuffling
|   |-- evaluation.py                     # Evaluation tables, OLS solver, test table
|   `-- visualization.py                  # Matplotlib rendering for all 6 figures
|
|-- results/
|   |-- figures/                          # 6 high-resolution figures
|   |   |-- 01_dataset_scatter.png
|   |   |-- 02_regression_lines.png
|   |   |-- 03_gd_loss_curve.png
|   |   |-- 04_sgd_loss_curve.png
|   |   |-- 05_gd_vs_sgd_loss.png
|   |   `-- 06_actual_vs_predicted_test.png
|   |-- results.csv                       # Summary metrics table
|   `-- test_predictions.csv              # Test set predictions table
|
|-- main.py                               # Master educational driver script
|-- requirements.txt                      # Project dependencies
`-- README.md                             # Comprehensive project documentation
```

---

## 14. Installation and Execution Guide

### Prerequisites
- Python 3.8 or higher (tested on Python 3.10)
- Git command line tools

### Installation
Clone the repository and install required dependencies:

```bash
git clone https://github.com/sahajasakhunala/linear-regression-gd-sgd.git
cd linear-regression-gd-sgd
pip install -r requirements.txt
```

### Running the Educational Pipeline
Execute the master script from the project root:

```bash
python main.py
```

Execution outputs:
- Real-time logging of all 8 analytical sections directly in the console.
- Export of comparative metric tables to results/results.csv and results/test_predictions.csv.
- Rendering of all 6 publication figures in results/figures/.
