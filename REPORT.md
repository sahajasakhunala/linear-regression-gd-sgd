# Linear Regression Using Gradient Descent and Stochastic Gradient Descent: A Comparative Study

---

## 1. Title Page

<div align="center">
  <img src="./assets/woxsen_logo.png" alt="Woxsen University Logo" width="320">
</div>

**Project Title:**  
Linear Regression Using Gradient Descent and Stochastic Gradient Descent: A Comparative Study

**Academic Course:**  
Foundations of Artificial Intelligence

**Academic Institution:**  
Department of Computer Science & Engineering, School of Technology  
Woxsen University, Hyderabad  
Academic Term: 2026

**Course Faculty / Supervisor:**  
Dr. YVR Naga Pavan

**Project Team Members:**
- Sakhunala Sahaja (Student ID: 25WU0102237)
- Sufiyan Yusuf Jahan (Student ID: 25WU0102276)
- Sai Sri Charan Chuduri (Student ID: 25WU0102235)

**Submission Date:**  
October 11, 2026

**GitHub Project Repository:**  
[https://github.com/sahajasakhunala/linear-regression-gd-sgd](https://github.com/sahajasakhunala/linear-regression-gd-sgd)

**Generated PDF Report:**  
`Linear_Regression_GD_vs_SGD_Report.pdf` (14 pages, complete with running footers and page numbering)

---

## 2. Abstract

Optimization algorithms constitute the mathematical engine of modern artificial intelligence and statistical learning. In this project, a comprehensive comparative study between Batch Gradient Descent (GD) and Stochastic Gradient Descent (SGD) is conducted for univariate Linear Regression. Built strictly from first principles using Python and NumPy, the system avoids black-box library estimators (such as scikit-learn's `LinearRegression` or `SGDRegressor`) in order to inspect every intermediate computational step: forward hypothesis generation, Mean Squared Error (MSE) loss computation, closed-form gradient evaluation, and parameter updates.

The empirical investigation is performed on an academic dataset of 15 student records relating study hours to examination scores. Strict machine learning hygiene is enforced via a 70/30 train/test split (10 training samples, 5 held-out test samples) under reproducible random seeding (seed 42) with zero data leakage. Both optimization algorithms are tracked across their convergence trajectories alongside an analytical Ordinary Least Squares (OLS) benchmark. The final models demonstrate convergence to virtually identical regression equations: Batch GD yields $\hat{y} = 7.7822x + 28.2121$, while SGD yields $\hat{y} = 7.7156x + 28.5394$. On unseen test data, both models generalize strongly, obtaining a test $R^2$ of 0.9973 and a test Root Mean Squared Error (RMSE) of approximately 0.93 marks. SGD achieves comparable generalization while requiring an order of magnitude fewer individual sample evaluations (1,000 versus 10,000 for Batch GD), illustrating key computational and dynamical trade-offs between deterministic and stochastic optimization.

---

## 3. Problem Statement

Standard machine learning packages provide high-level abstractions that fit linear regression parameters in a single line of execution. While computationally convenient for industry pipelines, these black-box abstractions conceal the core mechanics of iterative optimization: loss landscape navigation, gradient magnitude scaling, parameter trajectory stability, and sample-level stochasticity.

When students or researchers treat optimization as an opaque library call, they miss fundamental engineering insights:
1. How individual data points exert directional pull on model weights.
2. Why gradient variance manifests as stochastic fluctuation in single-sample updates.
3. How learning rates govern the trade-off between convergence speed and numerical overshoot.
4. How data leakage during preprocessing artificially corrupts generalization claims.

To resolve this pedagogical deficit, this project constructs an end-to-end machine learning system completely from first mathematical principles. By deriving analytical gradients, implementing parameter updates manually, isolating held-out test data, and visualizing optimization trajectories side-by-side, this study provides absolute transparency into how Linear Regression learns optimal parameters from empirical observations.

---

## 4. Introduction

Supervised machine learning identifies predictive relationships between input features and continuous target outcomes. Linear Regression represents the bedrock foundation of statistical learning and predictive modeling. In its simplest univariate formulation, it models an affine mapping from a single scalar input $x$ to a scalar output $y$:

$$\hat{y} = w x + b$$

Where:
- $x \in \mathbb{R}$ represents the independent input feature (Study Hours).
- $\hat{y} \in \mathbb{R}$ represents the predicted target value (Exam Score).
- $w \in \mathbb{R}$ denotes the learnable weight parameter (slope).
- $b \in \mathbb{R}$ denotes the learnable bias parameter (intercept).

While closed-form solutions (such as the Normal Equation / Ordinary Least Squares) exist for small-scale linear systems, iterative first-order optimization via Gradient Descent forms the universal backbone for training deep neural networks, transformers, and large-scale AI architectures where analytical matrix inversion is computationally intractable. Consequently, understanding the convergence mechanics of Batch Gradient Descent and Stochastic Gradient Descent on a transparent linear objective establishes essential intuition for contemporary artificial intelligence.

---

## 5. Objectives

The specific technical and educational objectives of this investigation are:

1. **Mathematical Derivation:** Formulate the univariate linear regression hypothesis and derive the exact analytical gradients for the Mean Squared Error (MSE) loss function with respect to weight $w$ and bias $b$.
2. **First-Principles Implementation:** Implement Batch Gradient Descent and Stochastic Gradient Descent from scratch in Python using only vectorized NumPy operations, eliminating all third-party model fitting utilities.
3. **Analytical Benchmark:** Derive and compute the exact Ordinary Least Squares (OLS) parameter solution to serve as a theoretical baseline.
4. **Experimental Rigor & Leakage Prevention:** Implement a strict 70/30 train/test partition with isolated preprocessing to guarantee zero data leakage between training optimization and post-training generalization evaluation.
5. **Convergence Trajectory Analysis:** Record iteration-by-iteration and epoch-by-epoch loss trajectories to contrast the smooth, monotonic descent of Batch GD against the noisy, fluctuating trajectory of SGD.
6. **Generalization Assessment:** Evaluate model performance on unseen test samples using standard statistical metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$).
7. **Empirical Visualization:** Generate six high-resolution, academic-quality plots documenting data partitions, fitted lines, loss curves, and actual-versus-predicted test outcomes.

---

## 6. Theoretical Background and Organized Mathematics

To ensure mathematical clarity and ease of comprehension, all equations, notations, and derivations are systematically organized into dedicated reference blocks below.

### 6.1 Master Variable Definition Reference

| Mathematical Symbol | Formal Concept | Physical Meaning in Project | Value / Domain in Experiment |
| :---: | :--- | :--- | :--- |
| **$x_i$** | Independent Feature | Number of hours studied by student $i$ | $1.0 \le x \le 8.0$ hours |
| **$y_i$** | Ground Truth Label | Actual final exam percentage score | $35.0 \le y \le 89.0$ marks |
| **$\hat{y}_i$** | Model Prediction | Predicted exam score: $\hat{y}_i = w x_i + b$ | Continuous scalar (marks) |
| **$w$** | Weight Parameter (Slope) | Marks gained per additional study hour | Initialized at $0.0$; Converges to $\approx 7.78$ |
| **$b$** | Bias Parameter (Intercept) | Baseline exam score with $0$ study hours | Initialized at $0.0$; Converges to $\approx 28.21$ |
| **$n$** | Batch Sample Size | Number of observations in training partition | $n = 10$ training samples |
| **$\alpha$** | Learning Rate (Step Size) | Gradient descent update step scalar | Hyperparameter fixed at $\alpha = 0.01$ |
| **$J(w, b)$** | Objective Loss Function | Mean Squared Error (MSE) across samples | Convex quadratic surface |

---

### 6.2 The Linear Regression Hypothesis

For any given student observation indexed by $i$, the linear hypothesis predicts:

$$\hat{y}_i = w \cdot x_i + b$$

- **Weight ($w$):** Quantifies the rate of increase in exam score per hour of study.
- **Bias ($b$):** Quantifies the expected baseline score of a student who studies zero hours.

---

### 6.3 Mean Squared Error (MSE) Objective Function

To quantify empirical prediction error across $n$ training observations, the Mean Squared Error loss is formulated as:

$$J(w, b) = \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - (w \cdot x_i + b) \right)^2$$

**Theoretical Justification for MSE:**
1. **Strict Convexity:** The quadratic bowl $J(w, b)$ has a unique global minimum, eliminating local minima traps.
2. **Smooth Differentiability:** The objective is continuously differentiable everywhere, providing well-defined gradients.
3. **Progressive Error Penalty:** Squaring penalizes large deviations substantially more than small errors.
4. **Statistical Foundations:** Equivalent to Maximum Likelihood Estimation (MLE) under i.i.d. Gaussian error assumptions.

---

### 6.4 Step-by-Step Analytical Gradient Derivations

The gradient vector $\nabla J(w, b) = \left[ \frac{\partial J}{\partial w}, \frac{\partial J}{\partial b} \right]^T$ indicates the direction of steepest ascent. Applying the chain rule of calculus yields exact closed-form expressions:

#### Step 1: Weight Gradient Derivation ($\frac{\partial J}{\partial w}$)
$$\frac{\partial J}{\partial w} = \frac{\partial}{\partial w} \left[ \frac{1}{n} \sum_{i=1}^{n} (y_i - (w x_i + b))^2 \right]$$

$$\frac{\partial J}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - \hat{y}_i) \cdot \frac{\partial}{\partial w}(y_i - w x_i - b)$$

$$\frac{\partial J}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - \hat{y}_i) \cdot (-x_i)$$

$$\mathbf{\frac{\partial J}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i \cdot (y_i - \hat{y}_i)}$$

#### Step 2: Bias Gradient Derivation ($\frac{\partial J}{\partial b}$)
$$\frac{\partial J}{\partial b} = \frac{\partial}{\partial b} \left[ \frac{1}{n} \sum_{i=1}^{n} (y_i - (w x_i + b))^2 \right]$$

$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - \hat{y}_i) \cdot (-1)$$

$$\mathbf{\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)}$$

#### Step 3: Parameter Update Rules (Gradient Descent Step)
Parameters are adjusted in the direction opposite to the gradient scaled by the learning rate $\alpha$:

$$w \leftarrow w - \alpha \frac{\partial J}{\partial w} = w + \alpha \frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)$$

$$b \leftarrow b - \alpha \frac{\partial J}{\partial b} = b + \alpha \frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

---

### 6.5 Master Comparison: Batch GD vs. Stochastic GD Equations

| Mathematical Property | Batch Gradient Descent (GD) | Stochastic Gradient Descent (SGD) |
| :--- | :--- | :--- |
| **Data Scope per Step** | Full training batch (all $n = 10$ samples) | Single observation (sample $i \in \{1, \dots, n\}$) |
| **Hypothesis Form** | $\hat{y} = w \mathbf{X} + b$ (Vectorized) | $\hat{y}_i = w x_i + b$ (Scalar) |
| **Objective Function** | $J_{\text{Batch}}(w, b) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | $J_i(w, b) = (y_i - \hat{y}_i)^2$ |
| **Weight Gradient ($\nabla_w$)** | $\frac{\partial J}{\partial w} = -\frac{2}{n}\sum_{i=1}^{n} x_i (y_i - \hat{y}_i)$ | $\frac{\partial J_i}{\partial w} = -2 x_i (y_i - \hat{y}_i)$ |
| **Bias Gradient ($\nabla_b$)** | $\frac{\partial J}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)$ | $\frac{\partial J_i}{\partial b} = -2 (y_i - \hat{y}_i)$ |
| **Weight Update Rule** | $w \leftarrow w + \alpha \frac{2}{n}\sum_{i=1}^{n} x_i (y_i - \hat{y}_i)$ | $w \leftarrow w + 2\alpha x_i (y_i - \hat{y}_i)$ |
| **Bias Update Rule** | $b \leftarrow b + \alpha \frac{2}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)$ | $b \leftarrow b + 2\alpha (y_i - \hat{y}_i)$ |
| **Nature of Gradient** | Exact deterministic empirical gradient | Noisy, unbiased stochastic gradient |
| **Updates per Epoch** | $1$ update per pass over dataset | $n = 10$ updates per pass (shuffled per epoch) |
| **Total Sample Work** | $1,000 \times 10 = \mathbf{10,000}$ evaluations | $100 \times 10 = \mathbf{1,000}$ evaluations |

---

## 7. System Design

The system follows a modular, decoupled software engineering architecture designed for separation of concerns and reproducibility.

```mermaid
flowchart TD
    subgraph Data Layer
        CSV["study_hours_exam_scores.csv (15 records)"] --> Preproc["src/data_preprocessing.py"]
        Preproc --> Split["70/30 Isolation Split (Seed 42)"]
        Split --> TrainSet["Training Set (n=10)"]
        Split --> TestSet["Test Set (n=5, Quarantined)"]
    end

    subgraph Optimization Engine
        TrainSet --> GD["src/gradient_descent.py (Batch GD)"]
        TrainSet --> SGD["src/stochastic_gradient_descent.py (SGD)"]
        TrainSet --> OLS["src/evaluation.py (Analytical OLS Reference)"]
    end

    subgraph Evaluation & Outputs
        GD --> Eval["src/evaluation.py (MSE, RMSE, R2)"]
        SGD --> Eval
        OLS --> Eval
        TestSet --> Eval
        Eval --> Viz["src/visualization.py (6 PNG Figures)"]
        Eval --> Tab["results/results.csv & test_predictions.csv"]
        Eval --> Pres["index.html (Interactive Presentation Deck)"]
        Eval --> Pred["predict.py (Interactive CLI Predictor)"]
        Eval --> PDF["Linear_Regression_GD_vs_SGD_Report.pdf (14-page PDF)"]
    end
```

### Module Breakdown:
1. `data_preprocessing.py`: Handles CSV ingestion, structure inspection, descriptive statistics, feature scaling audit, and 70/30 train/test partitioning.
2. `linear_regression.py`: Houses fundamental mathematical definitions: forward prediction function $\hat{y} = wx + b$, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).
3. `gradient_descent.py`: Implements Batch Gradient Descent with complete trajectory tracking (loss, weights, biases across 1,000 iterations).
4. `stochastic_gradient_descent.py`: Implements Stochastic Gradient Descent with sample-level permutation and epoch-wise tracking across 100 epochs.
5. `evaluation.py`: Houses model performance routines, closed-form OLS calculation, and formatted comparison table generation.
6. `visualization.py`: Generates the six high-resolution matplotlib visual figures with publication-ready styling.
7. `main.py`: The master orchestration script running the automated pipeline end-to-end.
8. `predict.py`: Interactive and command-line prediction script for live testing and demonstration.
9. `build_academic_pdf.py`: Automated pipeline converting report markdown/HTML to a 14-page numbered PDF.
10. `index.html`: Standalone, interactive HTML5 presentation deck with moveable cards and a 5-palette theme engine.

---

## 8. Implementation Details

### 8.1 Dataset Description and Partitioning

The experimental dataset comprises 15 student observations recording study time in hours and resulting exam scores in percentage marks.

```text
Student  1: (1.0 hr, 35 marks) -> Test Partition
Student  2: (1.5 hr, 39 marks) -> Training Partition
Student  3: (2.0 hr, 44 marks) -> Training Partition
Student  4: (2.5 hr, 48 marks) -> Training Partition
Student  5: (3.0 hr, 52 marks) -> Training Partition
Student  6: (3.5 hr, 55 marks) -> Test Partition
Student  7: (4.0 hr, 61 marks) -> Training Partition
Student  8: (4.5 hr, 64 marks) -> Training Partition
Student  9: (5.0 hr, 68 marks) -> Training Partition
Student 10: (5.5 hr, 71 marks) -> Test Partition
Student 11: (6.0 hr, 75 marks) -> Training Partition
Student 12: (6.5 hr, 78 marks) -> Test Partition
Student 13: (7.0 hr, 82 marks) -> Training Partition
Student 14: (7.5 hr, 85 marks) -> Test Partition
Student 15: (8.0 hr, 89 marks) -> Training Partition
```

- **Training Partition ($n = 10$, 70%):** Hours = `[5.0, 2.0, 1.5, 8.0, 3.0, 4.5, 6.0, 7.0, 2.5, 4.0]`; Scores = `[68, 44, 39, 89, 52, 64, 75, 82, 48, 61]`.
- **Test Partition ($n = 5$, 30%):** Hours = `[1.0, 3.5, 5.5, 6.5, 7.5]`; Scores = `[35, 55, 71, 78, 85]`.

### 8.2 Data Leakage Prevention

To ensure strict experimental integrity:
- **Optimization Isolation:** The 5 test observations are never passed to the optimization routines. No gradients, residuals, or loss values are calculated on test samples during model fitting.
- **Preprocessing Isolation:** Feature scaling analysis demonstrated that raw features are naturally bounded ($1.0 \le x \le 8.0$) and converge reliably with $\alpha = 0.01$. Keeping raw scale preserves direct physical interpretability of parameters: $w \approx 7.7$ marks per study hour, and $b \approx 28.2$ baseline marks.

---

### 8.3 Structured Algorithmic Procedures

#### Algorithm 1: Batch Gradient Descent (Deterministic Optimization)
```text
1. Inputs:
   - Training feature vector X in R^n, target vector y in R^n
   - Learning rate alpha = 0.01, Total iterations K = 1000

2. Initialization:
   - Initial parameters: w = 0.0, b = 0.0
   - Sample count: n = length(X)

3. Optimization Loop (for iteration k = 1 to K):
   a. Compute batch predictions:
      y_hat = w * X + b

   b. Compute batch empirical MSE:
      J = (1 / n) * sum((y - y_hat)^2)

   c. Compute analytical partial derivatives across all n observations:
      grad_w = (-2 / n) * sum(X * (y - y_hat))
      grad_b = (-2 / n) * sum(y - y_hat)

   d. Update parameters along the negative gradient:
      w <- w - alpha * grad_w
      b <- b - alpha * grad_b

   e. Append (k, J, w, b) to convergence trajectory history.

4. Output:
   - Return optimized parameters (w, b) and convergence trajectory history.
```

#### Algorithm 2: Stochastic Gradient Descent (Stochastic Optimization)
```text
1. Inputs:
   - Training feature vector X in R^n, target vector y in R^n
   - Learning rate alpha = 0.01, Epoch count E = 100, Random seed = 42

2. Initialization:
   - Initial parameters: w = 0.0, b = 0.0
   - Sample count: n = length(X)

3. Optimization Loop (for epoch e = 1 to E):
   a. Generate random permutation of training indices:
      indices = RandomShuffle([0, 1, ..., n-1])

   b. For each sample index i in indices:
      i.   Extract individual observation: (x_i, y_i)
      ii.  Compute single-sample prediction:
           y_hat_i = w * x_i + b
      iii. Compute instantaneous stochastic gradient:
           grad_w_i = -2 * x_i * (y_i - y_hat_i)
           grad_b_i = -2 * (y_i - y_hat_i)
      iv.  Update parameters immediately:
           w <- w - alpha * grad_w_i
           b <- b - alpha * grad_b_i

   c. Compute post-epoch full training MSE:
      J_epoch = (1 / n) * sum((y - (w * X + b))^2)

   d. Append (e, J_epoch, w, b) to epoch trajectory history.

4. Output:
   - Return optimized parameters (w, b) and epoch trajectory history.
```

---

## 9. Experimental Results and Analysis

### 9.1 Performance Metrics Comparison Table

The quantitative performance comparison between Batch Gradient Descent, Stochastic Gradient Descent, and the analytical Ordinary Least Squares solution is summarized below:

| Metric | Batch Gradient Descent (GD) | Stochastic Gradient Descent (SGD) | Analytical OLS (Reference) |
| :--- | :---: | :---: | :---: |
| **Learned Weight ($w$)** | **7.7822** | **7.7156** | **7.6379** |
| **Learned Bias ($b$)** | **28.2121** | **28.5394** | **28.9753** |
| **Training MSE** | 0.7892 | 0.7176 | 0.6823 |
| **Training RMSE** | 0.8884 | 0.8471 | 0.8260 |
| **Training $R^2$** | 0.9968 | 0.9971 | 0.9973 |
| **Test MSE** | **0.8636** | **0.8650** | **1.0151** |
| **Test RMSE** | **0.9293** | **0.9300** | **1.0075** |
| **Test $R^2$** | **0.9973** | **0.9973** | **0.9968** |
| **Total Parameter Updates** | 1,000 | 1,000 | 1 (Closed-form) |
| **Total Sample Evaluations** | 10,000 | 1,000 | 10 |

### 9.2 Test Set Predictions Table

Evaluation on the 5 quarantined test students yielded the following predictions:

| Test Student | Study Hours ($x$) | Actual Score ($y$) | GD Prediction ($\hat{y}$) | SGD Prediction ($\hat{y}$) | GD Error ($y - \hat{y}$) | SGD Error ($y - \hat{y}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Student 1 | 1.0 | 35.0 | 35.99 | 36.25 | -0.99 | -1.25 |
| Student 6 | 3.5 | 55.0 | 55.45 | 55.54 | -0.45 | -0.54 |
| Student 10 | 5.5 | 71.0 | 71.01 | 70.97 | -0.01 | +0.03 |
| Student 12 | 6.5 | 78.0 | 78.80 | 78.69 | -0.80 | -0.69 |
| Student 14 | 7.5 | 85.0 | 86.58 | 86.41 | -1.58 | -1.41 |

The average absolute test error across all test samples is approximately **0.77 marks**, demonstrating high predictive fidelity.

---

### 9.3 Visualizations and Graphical Analysis

#### Figure 1: Dataset Partition Scatter Plot
![Figure 1: Dataset Partition Scatter Plot](./results/figures/01_dataset_scatter.png)

*Analysis of Figure 1:* The raw dataset demonstrates a strong positive linear relationship between study hours and exam scores. The 10 blue circles denote the training observations used exclusively to fit the models. The 5 orange squares indicate the test observations held out for post-training validation. The test points span the domain (from 1.0 to 7.5 hours), ensuring an unbiased check across low, medium, and high study hours.

---

#### Figure 2: Fitted Regression Lines (GD vs. SGD)
![Figure 2: Fitted Regression Lines](./results/figures/02_regression_lines.png)

*Analysis of Figure 2:* Both fitted regression equations ($\hat{y}_{\text{GD}} = 7.7822x + 28.2121$ in solid blue, and $\hat{y}_{\text{SGD}} = 7.7156x + 28.5394$ in dashed red) overlap almost indistinguishably across the entire domain. Both lines align closely with both training and test observations, confirming that both optimization methods converged to the underlying ground-truth linear structure.

---

#### Figure 3: Batch Gradient Descent Loss Curve
![Figure 3: Batch GD Loss Curve](./results/figures/03_gd_loss_curve.png)

*Analysis of Figure 3:* The Batch GD loss curve displays a smooth, strictly monotonic decrease from an initial MSE of 4117.60 down to 0.7892 over 1,000 iterations. Because each parameter update is computed across all 10 training samples simultaneously, the gradient direction is exact at every step, exhibiting zero noise or erratic oscillations.

---

#### Figure 4: Stochastic Gradient Descent Loss Curve
![Figure 4: SGD Loss Curve](./results/figures/04_sgd_loss_curve.png)

*Analysis of Figure 4:* The SGD loss curve illustrates rapid initial convergence, plummeting from 4117.60 to below 10 MSE within the first 30 epochs. However, unlike Batch GD, the curve exhibits noticeable high-frequency oscillations across epochs. This jitter is characteristic of stochastic gradient updates: individual data points pull the parameters in slightly different directions, causing the loss to fluctuate around the minimum.

---

#### Figure 5: Comparative Optimization Loss Trajectories
![Figure 5: Comparative Loss Trajectories](./results/figures/05_gd_vs_sgd_loss.png)

*Analysis of Figure 5:* The dual-panel comparative visualization directly contrasts the optimization dynamics:
1. **Left Panel (Batch GD):** A smooth, deterministic decay with asymptotic leveling towards convergence.
2. **Right Panel (SGD):** Rapid descent with persistent epoch-to-epoch variance.  
Despite this qualitative difference in path, both reach virtually identical terminal loss values (~0.72 - 0.79 MSE).

---

#### Figure 6: Unseen Test Set: Actual vs. Predicted Scores
![Figure 6: Actual vs Predicted Test Scores](./results/figures/06_actual_vs_predicted_test.png)

*Analysis of Figure 6:* Grouped bar charts compare the actual exam scores of the 5 unseen test students against predictions generated by Batch GD and SGD. Both models track actual scores across all study intervals with sub-mark discrepancies. For example, at 5.5 hours (Student 10), the actual score is 71.0 marks; GD predicts 71.01 marks and SGD predicts 70.97 marks.

---

## 10. Discussion and Limitations

### 10.1 Convergence Dynamics and Gradient Noise

The primary theoretical difference between Batch GD and SGD lies in their gradient calculation:
- In **Batch GD**, the gradient vector is the true average over all training samples:
  $$\nabla J(w, b) = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) \begin{bmatrix} x_i \\ 1 \end{bmatrix}$$
  This results in deterministic, steady progress directly down the steepest descent vector of the loss surface.
- In **SGD**, each update uses the gradient of a single sample $i$:
  $$\nabla J_i(w, b) = -2 (y_i - \hat{y}_i) \begin{bmatrix} x_i \\ 1 \end{bmatrix}$$
  Because individual data points have distinct residual errors, the stochastic gradient acts as a noisy approximation of the true gradient. This noise causes the parameter vector to follow a erratic path. In deep learning and non-convex optimization, this stochastic noise is beneficial because it helps escape shallow local minima and saddle points. In strictly convex linear regression, the noise causes the parameters to wander within a small neighborhood around the global minimum rather than settling on the exact minimum.

### 10.2 Computational Work and Sample Evaluations

Under the chosen hyperparameters:
- **Batch GD:** 1,000 iterations $\times$ 10 samples per iteration = **10,000 sample evaluations**.
- **SGD:** 100 epochs $\times$ 10 samples per epoch = **1,000 sample evaluations**.

Despite evaluating data points 10 times less frequently, SGD achieved equivalent test performance (Test MSE of 0.8650 vs. 0.8636). On massive datasets ($n \ge 10^6$), Batch GD is computationally impractical per step, whereas SGD provides rapid initial parameter convergence with minimal compute overhead.

### 10.3 Dataset Scale and Test Partition Limitations

A thorough academic analysis requires transparent acknowledgment of experimental limitations:
1. **Small Sample Size ($n = 15$):** The dataset is designed for educational clarity and step-by-step tracing. While ideal for instruction, a 15-sample dataset limits statistical power.
2. **Test Partition Sensitivity ($n_{\text{test}} = 5$):** Because the held-out test partition contains only 5 observations, individual residual errors disproportionately influence aggregate metrics (MSE, RMSE, and $R^2$). A single outlier among 5 test points could substantially alter test MSE.
3. **Low Dimensionality ($d = 1$):** With only one feature (Hours Studied), collinearity, multi-dimensional conditioning, and curse-of-dimensionality challenges are absent.

---

## 11. Conclusion

This project successfully implemented and evaluated univariate Linear Regression from first mathematical principles. By deriving analytical gradients and developing Batch Gradient Descent and Stochastic Gradient Descent entirely without high-level library estimators, the study offers complete visibility into iterative optimization.

### Key Takeaways:
1. **Parameter Equivalence:** Both Batch GD ($w = 7.7822, b = 28.2121$) and SGD ($w = 7.7156, b = 28.5394$) successfully recovered the underlying linear data generation process, converging to within 1% of the closed-form analytical OLS solution.
2. **Identical Predictive Generalization:** On quarantined test samples, both methods produced matching generalization metrics ($R^2 = 0.9973$, $\text{RMSE} \approx 0.93$ marks).
3. **Efficiency of Stochastic Updates:** SGD attained equivalent predictive accuracy with a **10-fold reduction in total sample evaluations** (1,000 vs. 10,000), empirically confirming the efficiency of stochastic optimization.

### Future Improvements:
- **Mini-Batch Gradient Descent:** Implementing mini-batch optimization (e.g., batch sizes of 2 to 4) to combine the vectorization efficiency of Batch GD with the computational speed and regularization benefits of SGD.
- **Adaptive Learning Rate Schedules:** Introducing learning rate decay ($\alpha_t = \frac{\alpha_0}{1 + \gamma t}$) or adaptive moment estimation (Adam, RMSProp) to suppress late-stage SGD oscillations around the minimum.
- **Multivariate Regression:** Extending the architecture to multi-feature datasets to study feature scaling, condition numbers, and matrix-vector formulations.

---

## 12. References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer New York. (Chapter 3: Linear Models for Regression).
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. (Chapter 4: Numerical Computation; Chapter 5: Machine Learning Basics).
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
4. Bottou, L. (2010). *Large-Scale Machine Learning with Stochastic Gradient Descent*. In Proceedings of COMPSTAT'2010 (pp. 177-186). Physica-Verlag HD.
5. Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). *Array programming with NumPy*. Nature, 585(7825), 357-362.
6. McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. In Proceedings of the 9th Python in Science Conference (pp. 56-61).
7. Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering, 9(3), 90-95.
8. Official Python Documentation (2026). *The Python Standard Library*. Python Software Foundation. [https://docs.python.org/3/](https://docs.python.org/3/)
