"""
Academic PDF Generation Pipeline
=================================
Course: Foundations of Artificial Intelligence
Institution: Woxsen University
Project: Linear Regression: Batch GD vs. Stochastic GD

This script compiles a comprehensive, publication-quality academic report:
1. Ingests all project results, tables, and 6 high-res figures.
2. Ingests the official Woxsen University logo.
3. Renders clean, organized mathematical equations and pseudocode algorithms.
4. Generates an A4 print-ready HTML document with base64 embedded assets.
5. Invokes Microsoft Edge in headless print-to-pdf mode.
6. Post-processes the PDF using PyMuPDF to stamp exact page numbers ('Page X of Y')
   and academic running footers on every page.
"""

import os
import sys
import base64
import subprocess
import pymupdf


def image_to_base64(filepath: str) -> str:
    """Convert an image file to a base64 data URI string."""
    if not os.path.exists(filepath):
        print(f"Warning: Image file not found: {filepath}")
        return ""
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(filepath)[1].lower().replace(".", "")
    mime = "image/png" if ext == "png" else f"image/{ext}"
    return f"data:{mime};base64,{encoded}"


def generate_report_html() -> str:
    # Asset paths
    logo_path = os.path.join("assets", "woxsen_logo.png")
    fig_dir = os.path.join("results", "figures")
    
    logo_b64 = image_to_base64(logo_path)
    fig1_b64 = image_to_base64(os.path.join(fig_dir, "01_dataset_scatter.png"))
    fig2_b64 = image_to_base64(os.path.join(fig_dir, "02_regression_lines.png"))
    fig3_b64 = image_to_base64(os.path.join(fig_dir, "03_gd_loss_curve.png"))
    fig4_b64 = image_to_base64(os.path.join(fig_dir, "04_sgd_loss_curve.png"))
    fig5_b64 = image_to_base64(os.path.join(fig_dir, "05_gd_vs_sgd_loss.png"))
    fig6_b64 = image_to_base64(os.path.join(fig_dir, "06_actual_vs_predicted_test.png"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Linear Regression: Batch GD vs. Stochastic GD - Academic Report</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 18mm 16mm 22mm 16mm;
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    line-height: 1.55;
    font-size: 10pt;
    background: #ffffff;
  }}

  .page-break {{
    page-break-after: always;
    break-after: page;
  }}

  /* Typography */
  h1, h2, h3, h4 {{
    color: #0f172a;
    font-weight: 700;
    page-break-after: avoid;
    break-after: avoid;
  }}

  h1 {{ font-size: 20pt; line-height: 1.25; margin-bottom: 8pt; }}
  h2 {{ font-size: 14pt; margin-top: 16pt; margin-bottom: 6pt; border-bottom: 1.5pt solid #e2e8f0; padding-bottom: 4pt; color: #1e3a8a; }}
  h3 {{ font-size: 11.5pt; margin-top: 12pt; margin-bottom: 5pt; color: #0369a1; }}
  h4 {{ font-size: 10pt; margin-top: 8pt; margin-bottom: 4pt; }}

  p {{
    margin-bottom: 8pt;
    text-align: justify;
  }}

  ul, ol {{
    margin-left: 18pt;
    margin-bottom: 8pt;
  }}

  li {{
    margin-bottom: 3pt;
  }}

  /* Title Page */
  .title-page {{
    min-height: 90vh;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    text-align: center;
    padding: 30pt 20pt 20pt 20pt;
  }}

  .title-logo-container {{
    margin-bottom: 25pt;
  }}

  .title-logo {{
    max-width: 260pt;
    height: auto;
  }}

  .title-academic-badge {{
    display: inline-block;
    background: #eff6ff;
    border: 1pt solid #bfdbfe;
    color: #1d4ed8;
    font-size: 9pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5pt;
    padding: 4pt 12pt;
    border-radius: 4pt;
    margin-bottom: 14pt;
  }}

  .report-main-title {{
    font-size: 23pt;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.2;
    margin-bottom: 10pt;
  }}

  .report-subtitle {{
    font-size: 12pt;
    color: #475569;
    margin-bottom: 30pt;
    font-weight: 500;
  }}

  .title-meta-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16pt;
    margin: 20pt 0;
    text-align: left;
  }}

  .meta-card {{
    background: #f8fafc;
    border: 1pt solid #e2e8f0;
    border-radius: 8pt;
    padding: 14pt;
  }}

  .meta-card-title {{
    font-size: 10pt;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 8pt;
    text-transform: uppercase;
    letter-spacing: 0.8pt;
    border-bottom: 1pt solid #cbd5e1;
    padding-bottom: 3pt;
  }}

  .team-member {{
    display: flex;
    justify-content: space-between;
    padding: 4pt 0;
    border-bottom: 0.5pt dashed #e2e8f0;
    font-size: 9.5pt;
  }}

  .team-member:last-child {{
    border-bottom: none;
  }}

  .member-name {{
    font-weight: 600;
    color: #1e293b;
  }}

  .member-id {{
    font-family: monospace;
    font-weight: 700;
    color: #2563eb;
  }}

  .meta-info-row {{
    margin-bottom: 6pt;
    font-size: 9.5pt;
  }}

  .meta-info-label {{
    color: #64748b;
    font-size: 8pt;
    text-transform: uppercase;
    font-weight: 600;
  }}

  .meta-info-val {{
    font-weight: 700;
    color: #0f172a;
  }}

  .title-footer {{
    margin-top: 25pt;
    border-top: 1pt solid #e2e8f0;
    padding-top: 12pt;
    font-size: 8.5pt;
    color: #64748b;
  }}

  /* Tables */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0;
    font-size: 9pt;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  th, td {{
    padding: 6pt 8pt;
    border: 1pt solid #cbd5e1;
    text-align: left;
  }}

  th {{
    background: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
  }}

  tr:nth-child(even) {{
    background: #f8fafc;
  }}

  /* Math Blocks & Equation Cards */
  .math-card {{
    background: #f8fafc;
    border: 1pt solid #cbd5e1;
    border-left: 3.5pt solid #2563eb;
    border-radius: 6pt;
    padding: 10pt 14pt;
    margin: 10pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .math-card-header {{
    font-size: 9.5pt;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 6pt;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .eq-tag {{
    font-size: 8pt;
    font-family: monospace;
    background: #e2e8f0;
    color: #334155;
    padding: 1pt 5pt;
    border-radius: 3pt;
  }}

  .math-formula {{
    font-family: "Cambria Math", "Latin Modern Math", "Times New Roman", serif;
    font-size: 12pt;
    text-align: center;
    padding: 6pt 0;
    color: #0f172a;
  }}

  .math-details {{
    margin-top: 6pt;
    font-size: 8.5pt;
    color: #475569;
    border-top: 0.5pt solid #e2e8f0;
    padding-top: 5pt;
  }}

  .var-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 4pt 12pt;
    font-size: 8.5pt;
    margin-top: 4pt;
  }}

  .var-item {{
    display: flex;
    gap: 6pt;
  }}

  .var-sym {{
    font-family: "Cambria Math", serif;
    font-weight: 700;
    color: #1d4ed8;
    min-width: 25pt;
  }}

  /* Algorithm Box */
  .algo-card {{
    background: #f8fafc;
    border: 1pt solid #cbd5e1;
    border-radius: 6pt;
    margin: 10pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .algo-header {{
    background: #0f172a;
    color: #ffffff;
    padding: 6pt 10pt;
    font-size: 9.5pt;
    font-weight: 700;
    border-top-left-radius: 5pt;
    border-top-right-radius: 5pt;
  }}

  .algo-body {{
    padding: 8pt 12pt;
    font-size: 8.5pt;
    line-height: 1.5;
  }}

  .algo-step {{
    display: flex;
    margin-bottom: 4pt;
  }}

  .algo-step-num {{
    width: 22pt;
    font-weight: 700;
    color: #2563eb;
    font-family: monospace;
    flex-shrink: 0;
  }}

  .algo-step-content {{
    flex: 1;
  }}

  /* Figure Styling */
  .figure-container {{
    margin: 12pt 0;
    text-align: center;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .figure-img {{
    width: 95%;
    max-height: 280pt;
    object-fit: contain;
    border: 1pt solid #cbd5e1;
    border-radius: 6pt;
    background: #ffffff;
    box-shadow: 0 1pt 3pt rgba(0,0,0,0.06);
  }}

  .figure-caption {{
    font-size: 8.5pt;
    color: #475569;
    margin-top: 5pt;
    font-style: italic;
  }}

  .figure-caption strong {{
    color: #0f172a;
    font-style: normal;
  }}

  /* Callout Boxes */
  .callout {{
    background: #eff6ff;
    border-left: 3pt solid #3b82f6;
    padding: 8pt 10pt;
    border-radius: 4pt;
    margin: 8pt 0;
    font-size: 9pt;
  }}

  .callout-success {{
    background: #ecfdf5;
    border-left-color: #10b981;
  }}

  .callout-warning {{
    background: #fffbeb;
    border-left-color: #f59e0b;
  }}
</style>
</head>
<body>

  <!-- ========================================================= -->
  <!-- 1. TITLE PAGE                                             -->
  <!-- ========================================================= -->
  <div class="title-page">
    <div class="title-logo-container">
      <img src="{logo_b64}" alt="Woxsen University Logo" class="title-logo">
    </div>

    <div>
      <h1 class="report-main-title">Linear Regression Using Gradient Descent & Stochastic Gradient Descent</h1>
      <div class="report-subtitle">A First-Principles Comparative Study on Optimization Dynamics & Generalization Parity</div>
    </div>

    <div class="title-meta-grid">
      <div class="meta-card">
        <div class="meta-card-title">Project Investigators</div>
        <div class="team-member">
          <span class="member-name">Sakhunala Sahaja</span>
          <span class="member-id">25WU0102237</span>
        </div>
        <div class="team-member">
          <span class="member-name">Sufiyan Yusuf Jahan</span>
          <span class="member-id">25WU0102276</span>
        </div>
        <div class="team-member">
          <span class="member-name">Sai Sri Charan Chuduri</span>
          <span class="member-id">25WU0102235</span>
        </div>
      </div>

      <div class="meta-card">
        <div class="meta-card-title">Academic Details</div>
        <div class="meta-info-row">
          <div class="meta-info-label">Course</div>
          <div class="meta-info-val">Foundations of Artificial Intelligence</div>
        </div>
        <div class="meta-info-row">
          <div class="meta-info-label">Faculty Supervisor</div>
          <div class="meta-info-val">Dr. YVR Naga Pavan</div>
        </div>
        <div class="meta-info-row">
          <div class="meta-info-label">Department & Institution</div>
          <div class="meta-info-val">Dept. of Computer Science & Engineering, Woxsen University</div>
        </div>
      </div>
    </div>

    <div class="title-footer">
      <div>Submission Date: October 2026 | Academic Term: 2026</div>
      <div>GitHub Repository: https://github.com/sahajasakhunala/linear-regression-gd-sgd</div>
    </div>
  </div>

  <div class="page-break"></div>

  <!-- ========================================================= -->
  <!-- 2. ABSTRACT                                               -->
  <!-- ========================================================= -->
  <h2>2. Abstract</h2>
  <p>
    Optimization algorithms constitute the mathematical engine of modern artificial intelligence and statistical learning. In this project, an end-to-end empirical and theoretical comparative study between Batch Gradient Descent (GD) and Stochastic Gradient Descent (SGD) is conducted for univariate Linear Regression. Built entirely from first mathematical principles using Python and NumPy, the system eliminates reliance on high-level library estimators (such as scikit-learn's <code>LinearRegression</code> or <code>SGDRegressor</code>) in order to make every intermediate computational step transparent: forward hypothesis generation, Mean Squared Error (MSE) loss computation, closed-form analytical gradient calculation, and iterative parameter updates.
  </p>
  <p>
    The experiment is conducted on a canonical 15-observation academic dataset tracking the relationship between study hours and exam scores. Methodological rigor is maintained via an isolated 70/30 train/test partition (10 training observations, 5 held-out test observations) under seed 42 with zero data leakage. Both iterative optimizers are evaluated against the closed-form Ordinary Least Squares (OLS) analytical reference. The experimental results demonstrate that both algorithms successfully recover the underlying data-generating relationship: Batch GD converges to <strong>ŷ = 7.7822x + 28.2121</strong>, while SGD converges to <strong>ŷ = 7.7156x + 28.5394</strong>. On unseen test data, both models achieve identical generalization fidelity (test R² = 0.9973, test RMSE ≈ 0.93 marks). Crucially, SGD achieves equivalent predictive accuracy while requiring an order of magnitude fewer individual sample evaluations (1,000 versus 10,000 for Batch GD), illustrating fundamental trade-offs between deterministic and stochastic optimization.
  </p>

  <!-- ========================================================= -->
  <!-- 3. PROBLEM STATEMENT                                      -->
  <!-- ========================================================= -->
  <h2>3. Problem Statement</h2>
  <p>
    Contemporary machine learning practice frequently treats model fitting as an opaque API invocation. While abstractions accelerate rapid prototyping, they obscure the underlying mechanics of optimization: loss landscape topology, learning rate sensitivity, gradient variance, and parameter trajectory stability. When optimization is treated as a black-box, practitioners miss foundational engineering insights into how individual data points guide parameter updates, why stochastic updates introduce variance, and how data leakage distorts empirical validation.
  </p>
  <p>
    To resolve this pedagogical challenge, this project develops a transparent, from-scratch implementation of Linear Regression. By deriving gradients analytically, implementing update rules manually, maintaining strict test-set isolation, and comparing Batch GD against SGD, this study provides complete clarity into how Linear Regression learns from data.
  </p>

  <!-- ========================================================= -->
  <!-- 4. INTRODUCTION & 5. OBJECTIVES                           -->
  <!-- ========================================================= -->
  <h2>4. Introduction</h2>
  <p>
    Supervised regression identifies continuous functional mappings between input features and target outcomes. In its univariate formulation, Linear Regression models an affine relationship:
  </p>
  <div class="math-card">
    <div class="math-card-header">
      <span>The Univariate Linear Hypothesis</span>
      <span class="eq-tag">Equation 1</span>
    </div>
    <div class="math-formula">
      ŷ<sub>i</sub> = w · x<sub>i</sub> + b
    </div>
    <div class="math-details">
      Where <strong>x<sub>i</sub></strong> denotes study hours for student i, <strong>ŷ<sub>i</sub></strong> denotes the predicted exam score, <strong>w</strong> denotes the slope/weight (rate of score gain per hour), and <strong>b</strong> denotes the baseline intercept score at zero study hours.
    </div>
  </div>
  <p>
    While small linear systems can be solved directly via the Normal Equation, iterative gradient-based optimization forms the foundation for modern machine learning, deep neural networks, and foundation models where analytical inversion is intractable.
  </p>

  <h2>5. Project Objectives</h2>
  <ul>
    <li><strong>Mathematical Formulation:</strong> Formulate the linear regression hypothesis and derive analytical partial derivatives of Mean Squared Error (MSE) with respect to weight (w) and bias (b).</li>
    <li><strong>First-Principles Implementation:</strong> Develop Batch GD and SGD completely from scratch in Python using vectorized NumPy routines, avoiding black-box library fits.</li>
    <li><strong>Zero Data Leakage Safeguards:</strong> Partition data into 70% training (n=10) and 30% testing (n=5) subsets with strict isolation during fitting.</li>
    <li><strong>Comparative Trajectory Tracking:</strong> Record iteration-by-iteration and epoch-by-epoch loss trajectories to contrast deterministic versus stochastic convergence.</li>
    <li><strong>Comprehensive Generalization Evaluation:</strong> Evaluate models on unseen test observations using MSE, RMSE, and R² alongside closed-form OLS solutions.</li>
    <li><strong>Publication-Quality Visualization:</strong> Generate six high-resolution visual figures illustrating dataset partition, fitted lines, loss curves, and actual versus predicted test scores.</li>
  </ul>

  <div class="page-break"></div>

  <!-- ========================================================= -->
  <!-- 6. THEORETICAL BACKGROUND (ORGANIZED MATHEMATICS)         -->
  <!-- ========================================================= -->
  <h2>6. Theoretical Background & Mathematical Derivations</h2>
  <p>
    To make the mathematics intuitive and structured, all equations, variables, and derivations are organized systematically below.
  </p>

  <h3>6.1 Master Variable Definition Reference</h3>
  <table>
    <thead>
      <tr>
        <th style="width: 15%;">Symbol</th>
        <th style="width: 25%;">Mathematical Meaning</th>
        <th style="width: 35%;">Project Physical Interpretation</th>
        <th style="width: 25%;">Experimental Value / Domain</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>x<sub>i</sub></strong></td>
        <td>Independent Feature</td>
        <td>Hours studied by student i</td>
        <td>1.0 ≤ x ≤ 8.0 hours</td>
      </tr>
      <tr>
        <td><strong>y<sub>i</sub></strong></td>
        <td>Ground Truth Target</td>
        <td>Actual exam percentage score</td>
        <td>35.0 ≤ y ≤ 89.0 marks</td>
      </tr>
      <tr>
        <td><strong>ŷ<sub>i</sub></strong></td>
        <td>Hypothesis Output</td>
        <td>Model predicted exam score: ŷ<sub>i</sub> = wx<sub>i</sub> + b</td>
        <td>Continuous scalar (marks)</td>
      </tr>
      <tr>
        <td><strong>w</strong></td>
        <td>Weight Parameter (Slope)</td>
        <td>Score gained per study hour logged</td>
        <td>Initialized at 0.0; Converges ~7.78</td>
      </tr>
      <tr>
        <td><strong>b</strong></td>
        <td>Bias Parameter (Intercept)</td>
        <td>Baseline exam score with zero study hours</td>
        <td>Initialized at 0.0; Converges ~28.21</td>
      </tr>
      <tr>
        <td><strong>n</strong></td>
        <td>Sample Size</td>
        <td>Number of observations in training batch</td>
        <td>n = 10 training observations</td>
      </tr>
      <tr>
        <td><strong>α</strong> (alpha)</td>
        <td>Learning Rate</td>
        <td>Optimization step size multiplier</td>
        <td>Fixed hyperparameter α = 0.01</td>
      </tr>
      <tr>
        <td><strong>J(w, b)</strong></td>
        <td>Objective Loss Function</td>
        <td>Mean Squared Error (MSE) across dataset</td>
        <td>Convex quadratic surface</td>
      </tr>
    </tbody>
  </table>

  <h3>6.2 Mean Squared Error (MSE) Objective Function</h3>
  <div class="math-card">
    <div class="math-card-header">
      <span>Mean Squared Error (MSE) Formula</span>
      <span class="eq-tag">Equation 2</span>
    </div>
    <div class="math-formula">
      J(w, b) = <span style="font-size: 11pt;">1</span>/<span style="font-size: 9pt;">n</span> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - ŷ<sub>i</sub>)<sup>2</sup> = <span style="font-size: 11pt;">1</span>/<span style="font-size: 9pt;">n</span> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - (w·x<sub>i</sub> + b))<sup>2</sup>
    </div>
    <div class="math-details">
      <strong>Why MSE is selected:</strong> (1) <em>Strict Convexity</em> guarantees a unique global minimum with zero local minima traps; (2) <em>Continuous Differentiability</em> produces smooth analytical gradients everywhere; (3) <em>Quadratic Penalty</em> strongly discourages large residual errors; (4) Minimizing MSE is mathematically equivalent to Maximum Likelihood Estimation (MLE) under Gaussian error assumptions.
    </div>
  </div>

  <h3>6.3 Step-by-Step Analytical Gradient Derivations</h3>
  <p>
    The gradient vector ∇J(w, b) is computed by taking partial derivatives of J(w, b) using the calculus chain rule:
  </p>

  <div class="math-card">
    <div class="math-card-header">
      <span>Step 1: Weight Gradient Derivation (∂J / ∂w)</span>
      <span class="eq-tag">Equation 3a</span>
    </div>
    <div style="font-family: 'Cambria Math', serif; font-size: 10.5pt; line-height: 1.8; padding: 4pt 0;">
      1. Apply chain rule: &nbsp;&nbsp; <sup>∂J</sup>/<sub>∂w</sub> = <span style="font-size: 10pt;">1</span>/<span style="font-size: 8pt;">n</span> ∑<sub>i=1</sub><sup>n</sup> 2(y<sub>i</sub> - ŷ<sub>i</sub>) · <sup>∂</sup>/<sub>∂w</sub>(y<sub>i</sub> - w·x<sub>i</sub> - b)<br>
      2. Substitute inner derivative: &nbsp;&nbsp; <sup>∂</sup>/<sub>∂w</sub>(y<sub>i</sub> - w·x<sub>i</sub> - b) = -x<sub>i</sub><br>
      3. Factor constants: &nbsp;&nbsp; <strong><sup>∂J</sup>/<sub>∂w</sub> = -<sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> x<sub>i</sub> · (y<sub>i</sub> - ŷ<sub>i</sub>)</strong>
    </div>
  </div>

  <div class="math-card">
    <div class="math-card-header">
      <span>Step 2: Bias Gradient Derivation (∂J / ∂b)</span>
      <span class="eq-tag">Equation 3b</span>
    </div>
    <div style="font-family: 'Cambria Math', serif; font-size: 10.5pt; line-height: 1.8; padding: 4pt 0;">
      1. Apply chain rule: &nbsp;&nbsp; <sup>∂J</sup>/<sub>∂b</sub> = <span style="font-size: 10pt;">1</span>/<span style="font-size: 8pt;">n</span> ∑<sub>i=1</sub><sup>n</sup> 2(y<sub>i</sub> - ŷ<sub>i</sub>) · <sup>∂</sup>/<sub>∂b</sub>(y<sub>i</sub> - w·x<sub>i</sub> - b)<br>
      2. Substitute inner derivative: &nbsp;&nbsp; <sup>∂</sup>/<sub>∂b</sub>(y<sub>i</sub> - w·x<sub>i</sub> - b) = -1<br>
      3. Factor constants: &nbsp;&nbsp; <strong><sup>∂J</sup>/<sub>∂b</sub> = -<sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - ŷ<sub>i</sub>)</strong>
    </div>
  </div>

  <div class="page-break"></div>

  <h3>6.4 Master Comparison: Batch GD vs. Stochastic GD Equations</h3>
  <p>
    The mathematical formulation of parameter updates differs fundamentally in whether gradients are computed over the full dataset or a single observation:
  </p>

  <table>
    <thead>
      <tr>
        <th style="width: 25%;">Mathematical Property</th>
        <th style="width: 37%;">Batch Gradient Descent (GD)</th>
        <th style="width: 38%;">Stochastic Gradient Descent (SGD)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Data Scope per Step</strong></td>
        <td>Entire Training Batch (all n = 10 samples)</td>
        <td>Single Observation (sample i ∈ {{1, ..., n}})</td>
      </tr>
      <tr>
        <td><strong>Objective Function</strong></td>
        <td>J(w, b) = <sup>1</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - ŷ<sub>i</sub>)<sup>2</sup></td>
        <td>J<sub>i</sub>(w, b) = (y<sub>i</sub> - ŷ<sub>i</sub>)<sup>2</sup></td>
      </tr>
      <tr>
        <td><strong>Weight Gradient (∇<sub>w</sub>)</strong></td>
        <td><sup>∂J</sup>/<sub>∂w</sub> = -<sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> x<sub>i</sub>(y<sub>i</sub> - ŷ<sub>i</sub>)</td>
        <td><sup>∂J<sub>i</sub></sup>/<sub>∂w</sub> = -2 x<sub>i</sub>(y<sub>i</sub> - ŷ<sub>i</sub>)</td>
      </tr>
      <tr>
        <td><strong>Bias Gradient (∇<sub>b</sub>)</strong></td>
        <td><sup>∂J</sup>/<sub>∂b</sub> = -<sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - ŷ<sub>i</sub>)</td>
        <td><sup>∂J<sub>i</sub></sup>/<sub>∂b</sub> = -2 (y<sub>i</sub> - ŷ<sub>i</sub>)</td>
      </tr>
      <tr>
        <td><strong>Weight Update Rule</strong></td>
        <td>w ← w + α · <sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> x<sub>i</sub>(y<sub>i</sub> - ŷ<sub>i</sub>)</td>
        <td>w ← w + 2α · x<sub>i</sub>(y<sub>i</sub> - ŷ<sub>i</sub>)</td>
      </tr>
      <tr>
        <td><strong>Bias Update Rule</strong></td>
        <td>b ← b + α · <sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - ŷ<sub>i</sub>)</td>
        <td>b ← b + 2α · (y<sub>i</sub> - ŷ<sub>i</sub>)</td>
      </tr>
      <tr>
        <td><strong>Gradient Estimation</strong></td>
        <td>Exact deterministic gradient vector</td>
        <td>Noisy, unbiased stochastic gradient estimate</td>
      </tr>
      <tr>
        <td><strong>Updates per Epoch</strong></td>
        <td>1 update per pass over dataset</td>
        <td>n = 10 updates per pass (shuffled per epoch)</td>
      </tr>
      <tr>
        <td><strong>Total Sample Work</strong></td>
        <td>1,000 steps × 10 samples = <strong>10,000 evaluations</strong></td>
        <td>100 epochs × 10 samples = <strong>1,000 evaluations</strong></td>
      </tr>
    </tbody>
  </table>

  <!-- ========================================================= -->
  <!-- 7. SYSTEM DESIGN & 8. IMPLEMENTATION                      -->
  <!-- ========================================================= -->
  <h2>7. System Architecture & Modularity</h2>
  <p>
    The codebase is organized into decoupled modules following software engineering best practices:
  </p>
  <ul>
    <li><code>src/data_preprocessing.py</code>: Data loading, descriptive audits, feature scaling analysis, and isolated 70/30 partitioning.</li>
    <li><code>src/linear_regression.py</code>: First-principles core mathematics: forward prediction ŷ = wx + b, MSE, RMSE, and R².</li>
    <li><code>src/gradient_descent.py</code>: Batch Gradient Descent implementation with complete parameter and loss trajectory tracking.</li>
    <li><code>src/stochastic_gradient_descent.py</code>: Stochastic Gradient Descent with random permutation and epoch-level loss logging.</li>
    <li><code>src/evaluation.py</code>: Model evaluation routines, closed-form OLS analytical baseline computation, and metric formatting.</li>
    <li><code>src/visualization.py</code>: Production of the six publication-quality matplotlib figures using a non-interactive Agg backend.</li>
    <li><code>main.py</code>: Master orchestration script executing training, evaluation, CSV exports, and visualization routines.</li>
    <li><code>predict.py</code>: Interactive command-line predictor allowing live demonstration for any user-provided study hours.</li>
  </ul>

  <h2>8. Implementation Details</h2>
  <h3>8.1 Dataset Partitioning & Zero Leakage Audit</h3>
  <p>
    The dataset contains 15 observations partitioned via scikit-learn's <code>train_test_split(test_size=0.3, random_state=42)</code>:
  </p>
  <ul>
    <li><strong>Training Partition (70%, n=10):</strong> Hours = [5.0, 2.0, 1.5, 8.0, 3.0, 4.5, 6.0, 7.0, 2.5, 4.0]; Scores = [68, 44, 39, 89, 52, 64, 75, 82, 48, 61].</li>
    <li><strong>Held-out Test Partition (30%, n=5):</strong> Hours = [1.0, 3.5, 5.5, 6.5, 7.5]; Scores = [35, 55, 71, 78, 85].</li>
  </ul>
  <div class="callout callout-success">
    <strong>Zero Data Leakage Safeguard:</strong> The 5 test observations are completely quarantined during optimization. No gradients, residuals, or loss updates are evaluated on these samples while fitting parameters. They are evaluated strictly post-training to measure generalization.
  </div>

  <div class="page-break"></div>

  <h3>8.2 Structured Algorithmic Pseudocode</h3>
  <p>
    The mathematical logic of both algorithms is formatted in structured, step-by-step procedures below:
  </p>

  <div class="algo-card">
    <div class="algo-header">Algorithm 1: Batch Gradient Descent (Deterministic Optimization)</div>
    <div class="algo-body">
      <div class="algo-step"><span class="algo-step-num">1.</span><span class="algo-step-content"><strong>Input:</strong> Training feature vector <strong>X</strong> ∈ ℝ<sup>n</sup>, ground truth vector <strong>y</strong> ∈ ℝ<sup>n</sup>, learning rate α = 0.01, iterations K = 1000.</span></div>
      <div class="algo-step"><span class="algo-step-num">2.</span><span class="algo-step-content"><strong>Initialization:</strong> Set initial parameters w = 0.0, b = 0.0, sample size n = length(X).</span></div>
      <div class="algo-step"><span class="algo-step-num">3.</span><span class="algo-step-content"><strong>Loop for iteration k = 1 to K:</strong></span></div>
      <div class="algo-step"><span class="algo-step-num"></span><span class="algo-step-content" style="margin-left: 15pt;">
        a. Compute batch predictions: &nbsp; <strong>ŷ = w · X + b</strong><br>
        b. Evaluate batch loss: &nbsp; <strong>J = (1 / n) ∑ (y - ŷ)²</strong><br>
        c. Compute exact analytical gradients across all n samples:<br>
        &nbsp;&nbsp;&nbsp;&nbsp; <strong>∇<sub>w</sub> = (-2 / n) ∑ X · (y - ŷ)</strong><br>
        &nbsp;&nbsp;&nbsp;&nbsp; <strong>∇<sub>b</sub> = (-2 / n) ∑ (y - ŷ)</strong><br>
        d. Update parameters along negative gradient:<br>
        &nbsp;&nbsp;&nbsp;&nbsp; <strong>w ← w - α · ∇<sub>w</sub></strong><br>
        &nbsp;&nbsp;&nbsp;&nbsp; <strong>b ← b - α · ∇<sub>b</sub></strong><br>
        e. Record J, w, b in convergence history.
      </span></div>
      <div class="algo-step"><span class="algo-step-num">4.</span><span class="algo-step-content"><strong>Output:</strong> Optimal parameters (w, b) and full convergence history.</span></div>
    </div>
  </div>

  <div class="algo-card">
    <div class="algo-header">Algorithm 2: Stochastic Gradient Descent (Stochastic Optimization)</div>
    <div class="algo-body">
      <div class="algo-step"><span class="algo-step-num">1.</span><span class="algo-step-content"><strong>Input:</strong> Training data (X, y), learning rate α = 0.01, epochs E = 100, random seed = 42.</span></div>
      <div class="algo-step"><span class="algo-step-num">2.</span><span class="algo-step-content"><strong>Initialization:</strong> Set w = 0.0, b = 0.0, n = length(X).</span></div>
      <div class="algo-step"><span class="algo-step-num">3.</span><span class="algo-step-content"><strong>Loop for epoch e = 1 to E:</strong></span></div>
      <div class="algo-step"><span class="algo-step-num"></span><span class="algo-step-content" style="margin-left: 15pt;">
        a. Generate random permutation of sample indices: &nbsp; <strong>π = Shuffle([0, 1, ..., n-1])</strong><br>
        b. <strong>For each index i in π:</strong><br>
        &nbsp;&nbsp;&nbsp;&nbsp; i. Extract individual sample: (x<sub>i</sub>, y<sub>i</sub>)<br>
        &nbsp;&nbsp;&nbsp;&nbsp; ii. Predict individual output: <strong>ŷ<sub>i</sub> = w · x<sub>i</sub> + b</strong><br>
        &nbsp;&nbsp;&nbsp;&nbsp; iii. Compute instantaneous sample gradient:<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>∇<sub>w, i</sub> = -2 · x<sub>i</sub> · (y<sub>i</sub> - ŷ<sub>i</sub>)</strong><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>∇<sub>b, i</sub> = -2 · (y<sub>i</sub> - ŷ<sub>i</sub>)</strong><br>
        &nbsp;&nbsp;&nbsp;&nbsp; iv. Update parameters immediately:<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>w ← w - α · ∇<sub>w, i</sub></strong><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>b ← b - α · ∇<sub>b, i</sub></strong><br>
        c. Compute epoch-level MSE over training set: &nbsp; <strong>J<sub>epoch</sub> = (1 / n) ∑ (y - (w·X + b))²</strong><br>
        d. Record J<sub>epoch</sub>, w, b in history.
      </span></div>
      <div class="algo-step"><span class="algo-step-num">4.</span><span class="algo-step-content"><strong>Output:</strong> Optimal parameters (w, b) and epoch convergence history.</span></div>
    </div>
  </div>

  <div class="page-break"></div>

  <!-- ========================================================= -->
  <!-- 9. EXPERIMENTAL RESULTS AND ANALYSIS                      -->
  <!-- ========================================================= -->
  <h2>9. Experimental Results & Comparative Analysis</h2>

  <h3>9.1 Quantitative Performance Metrics Table</h3>
  <p>
    The quantitative comparison between Batch Gradient Descent, Stochastic Gradient Descent, and the analytical Ordinary Least Squares solution is presented below:
  </p>
  <table>
    <thead>
      <tr>
        <th>Performance Metric</th>
        <th>Batch Gradient Descent (GD)</th>
        <th>Stochastic Gradient Descent (SGD)</th>
        <th>Analytical OLS (Reference)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Learned Weight (w)</strong></td>
        <td><strong>7.7822</strong></td>
        <td><strong>7.7156</strong></td>
        <td><strong>7.6379</strong></td>
      </tr>
      <tr>
        <td><strong>Learned Bias (b)</strong></td>
        <td><strong>28.2121</strong></td>
        <td><strong>28.5394</strong></td>
        <td><strong>28.9753</strong></td>
      </tr>
      <tr>
        <td><strong>Training MSE</strong></td>
        <td>0.7892</td>
        <td>0.7176</td>
        <td>0.6823</td>
      </tr>
      <tr>
        <td><strong>Training RMSE</strong></td>
        <td>0.8884</td>
        <td>0.8471</td>
        <td>0.8260</td>
      </tr>
      <tr>
        <td><strong>Training R²</strong></td>
        <td>0.9968</td>
        <td>0.9971</td>
        <td>0.9973</td>
      </tr>
      <tr>
        <td><strong>Test MSE</strong></td>
        <td><strong>0.8636</strong></td>
        <td><strong>0.8650</strong></td>
        <td><strong>1.0151</strong></td>
      </tr>
      <tr>
        <td><strong>Test RMSE</strong></td>
        <td><strong>0.9293</strong></td>
        <td><strong>0.9300</strong></td>
        <td><strong>1.0075</strong></td>
      </tr>
      <tr>
        <td><strong>Test R²</strong></td>
        <td><strong>0.9973</strong></td>
        <td><strong>0.9973</strong></td>
        <td><strong>0.9968</strong></td>
      </tr>
      <tr>
        <td><strong>Total Parameter Updates</strong></td>
        <td>1,000</td>
        <td>1,000</td>
        <td>1 (Closed-form)</td>
      </tr>
      <tr>
        <td><strong>Total Sample Evaluations</strong></td>
        <td>10,000</td>
        <td>1,000</td>
        <td>10</td>
      </tr>
    </tbody>
  </table>

  <h3>9.2 Held-Out Test Set Predictions Table</h3>
  <p>
    Evaluation on the 5 quarantined test students demonstrates high predictive fidelity across both optimizers:
  </p>
  <table>
    <thead>
      <tr>
        <th>Test Observation</th>
        <th>Hours Studied (x)</th>
        <th>Actual Score (y)</th>
        <th>GD Prediction (ŷ)</th>
        <th>SGD Prediction (ŷ)</th>
        <th>GD Error (y - ŷ)</th>
        <th>SGD Error (y - ŷ)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Student 1</td>
        <td>1.0</td>
        <td>35.0</td>
        <td>35.99</td>
        <td>36.25</td>
        <td>-0.99</td>
        <td>-1.25</td>
      </tr>
      <tr>
        <td>Student 6</td>
        <td>3.5</td>
        <td>55.0</td>
        <td>55.45</td>
        <td>55.54</td>
        <td>-0.45</td>
        <td>-0.54</td>
      </tr>
      <tr>
        <td>Student 10</td>
        <td>5.5</td>
        <td>71.0</td>
        <td>71.01</td>
        <td>70.97</td>
        <td>-0.01</td>
        <td>+0.03</td>
      </tr>
      <tr>
        <td>Student 12</td>
        <td>6.5</td>
        <td>78.0</td>
        <td>78.80</td>
        <td>78.69</td>
        <td>-0.80</td>
        <td>-0.69</td>
      </tr>
      <tr>
        <td>Student 14</td>
        <td>7.5</td>
        <td>85.0</td>
        <td>86.58</td>
        <td>86.41</td>
        <td>-1.58</td>
        <td>-1.41</td>
      </tr>
    </tbody>
  </table>

  <div class="page-break"></div>

  <h3>9.3 Graphical Analysis of the Six Visualizations</h3>

  <div class="figure-container">
    <img src="{fig1_b64}" alt="Figure 1: Dataset Partition Scatter Plot" class="figure-img">
    <div class="figure-caption"><strong>Figure 1: Dataset Partition Scatter Plot.</strong> 10 training samples (blue circles) and 5 held-out test samples (orange squares) distributed across 1.0 to 8.0 study hours without overlap.</div>
  </div>

  <div class="figure-container">
    <img src="{fig2_b64}" alt="Figure 2: Fitted Regression Lines" class="figure-img">
    <div class="figure-caption"><strong>Figure 2: Fitted Regression Lines (GD vs. SGD).</strong> Batch GD line (ŷ = 7.7822x + 28.2121) and SGD line (ŷ = 7.7156x + 28.5394) overlap almost indistinguishably, confirming identical convergence.</div>
  </div>

  <div class="page-break"></div>

  <div class="figure-container">
    <img src="{fig3_b64}" alt="Figure 3: Batch GD Loss Curve" class="figure-img">
    <div class="figure-caption"><strong>Figure 3: Batch Gradient Descent Loss Curve.</strong> Strictly monotonic, smooth descent from initial MSE of 4117.60 down to 0.7892 over 1,000 iterations.</div>
  </div>

  <div class="figure-container">
    <img src="{fig4_b64}" alt="Figure 4: SGD Loss Curve" class="figure-img">
    <div class="figure-caption"><strong>Figure 4: Stochastic Gradient Descent Loss Curve.</strong> Rapid initial convergence in the first 20 epochs followed by characteristic stochastic fluctuations around the minimum.</div>
  </div>

  <div class="page-break"></div>

  <div class="figure-container">
    <img src="{fig5_b64}" alt="Figure 5: Comparative Optimization Trajectories" class="figure-img">
    <div class="figure-caption"><strong>Figure 5: Optimization Trajectories Compared.</strong> Side-by-side contrast between deterministic smooth descent (left) and stochastic fluctuating descent (right).</div>
  </div>

  <div class="figure-container">
    <img src="{fig6_b64}" alt="Figure 6: Unseen Test Set Predictions" class="figure-img">
    <div class="figure-caption"><strong>Figure 6: Actual vs. Predicted Scores on Unseen Test Data.</strong> Grouped bar chart demonstrating tight alignment between actual exam scores and model predictions across all 5 test students.</div>
  </div>

  <div class="page-break"></div>

  <!-- ========================================================= -->
  <!-- 10. DISCUSSION AND LIMITATIONS                            -->
  <!-- ========================================================= -->
  <h2>10. Discussion & Limitations</h2>

  <h3>10.1 Convergence Dynamics and Gradient Variance</h3>
  <p>
    The experimental results illustrate the core mathematical contrast between full-batch and stochastic gradient estimation. In Batch GD, the gradient vector reflects the true empirical risk across all training observations:
  </p>
  <div class="math-card">
    <div class="math-formula">
      ∇J<sub>Batch</sub>(w, b) = -<sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - ŷ<sub>i</sub>) · [x<sub>i</sub>, 1]<sup>T</sup>
    </div>
  </div>
  <p>
    Because every step averages the full sample, the gradient direction is exact at every iteration. This guarantees smooth, monotonic descent. Conversely, in SGD, each update uses only a single sample gradient:
  </p>
  <div class="math-card">
    <div class="math-formula">
      ∇J<sub>i</sub>(w, b) = -2(y<sub>i</sub> - ŷ<sub>i</sub>) · [x<sub>i</sub>, 1]<sup>T</sup>
    </div>
  </div>
  <p>
    Because individual observations deviate from the sample mean, the stochastic gradient acts as a noisy estimate of the true gradient. While this introduces variance (visible as epoch-by-epoch fluctuations in Figure 4 and Figure 5), it prevents the optimizer from stagnating and drives rapid early convergence.
  </p>

  <h3>10.2 Computational Work & Scalability</h3>
  <p>
    A critical finding of this study is the sample evaluation efficiency of SGD. Batch GD required <strong>10,000 individual sample evaluations</strong> (1,000 iterations × 10 samples) to achieve test MSE = 0.8636. In contrast, SGD required only <strong>1,000 sample evaluations</strong> (100 epochs × 10 samples) to achieve test MSE = 0.8650. This represents a <strong>10× reduction in computational sample work</strong> for essentially identical predictive accuracy. On large-scale industry datasets (n ≥ 10⁶), Batch GD becomes computationally prohibitive per update, whereas SGD or Mini-batch GD makes rapid initial progress.
  </p>

  <h3>10.3 Methodological Limitations</h3>
  <ul>
    <li><strong>Dataset Scale (n=15):</strong> The 15-observation dataset provides transparency for step-by-step tracing, but lacks the scale necessary to test complex regularization or higher-order variance effects.</li>
    <li><strong>Test Set Sensitivity (n<sub>test</sub>=5):</strong> Because the held-out test partition comprises only 5 observations, summary metrics (MSE, RMSE, R²) are sensitive to individual residual deviations. While our average absolute error is low (~0.77 marks), larger test cohorts are recommended for statistical confidence intervals.</li>
    <li><strong>Univariate Linearity:</strong> With only one input feature (study hours), collinearity, matrix conditioning, and multi-dimensional saddle points are not encountered.</li>
  </ul>

  <!-- ========================================================= -->
  <!-- 11. CONCLUSION & 12. REFERENCES                           -->
  <!-- ========================================================= -->
  <h2>11. Conclusion & Future Work</h2>
  <p>
    This investigation successfully implemented, derived, and compared Batch Gradient Descent and Stochastic Gradient Descent from first principles for univariate Linear Regression.
  </p>
  <div class="callout callout-success">
    <strong>Key Conclusions:</strong>
    <ol>
      <li><strong>Convergence Parity:</strong> Both algorithms successfully converged to the underlying data trend, arriving at nearly identical parameters (Batch GD: w=7.7822, b=28.2121; SGD: w=7.7156, b=28.5394).</li>
      <li><strong>Generalization Parity:</strong> Both models generalized with high fidelity on unseen test observations, achieving matching test R² = 0.9973 and RMSE ≈ 0.93 marks.</li>
      <li><strong>Sample Efficiency:</strong> SGD achieved equivalent predictive performance with 10× fewer sample evaluations, empirically validating the efficiency of stochastic optimization.</li>
    </ol>
  </div>
  <p>
    <strong>Future Work:</strong> Future extensions include: (1) Implementing <em>Mini-Batch Gradient Descent</em> to balance vectorization speed and stochastic regularization; (2) Adding <em>adaptive learning rate schedules</em> (e.g., learning rate decay or Adam) to damp late-stage SGD oscillations; and (3) Scaling to multivariate datasets with feature normalization.
  </p>

  <h2>12. References</h2>
  <ol>
    <li>Bishop, C. M. (2006). <em>Pattern Recognition and Machine Learning</em>. Springer New York. (Chapter 3: Linear Models for Regression).</li>
    <li>Goodfellow, I., Bengio, Y., & Courville, A. (2016). <em>Deep Learning</em>. MIT Press. (Chapter 4: Numerical Computation; Chapter 5: Machine Learning Basics).</li>
    <li>Hastie, T., Tibshirani, R., & Friedman, J. (2009). <em>The Elements of Statistical Learning: Data Mining, Inference, and Prediction</em> (2nd ed.). Springer.</li>
    <li>Bottou, L. (2010). <em>Large-Scale Machine Learning with Stochastic Gradient Descent</em>. In Proceedings of COMPSTAT'2010 (pp. 177-186). Physica-Verlag HD.</li>
    <li>Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). <em>Array programming with NumPy</em>. Nature, 585(7825), 357-362.</li>
    <li>McKinney, W. (2010). <em>Data Structures for Statistical Computing in Python</em>. In Proceedings of the 9th Python in Science Conference (pp. 56-61).</li>
    <li>Hunter, J. D. (2007). <em>Matplotlib: A 2D Graphics Environment</em>. Computing in Science & Engineering, 9(3), 90-95.</li>
    <li>Official Python Documentation (2026). <em>The Python Standard Library</em>. Python Software Foundation. https://docs.python.org/3/</li>
  </ol>

</body>
</html>
"""
    return html


def convert_html_to_pdf_with_edge(html_path: str, raw_pdf_path: str):
    """Invoke Microsoft Edge headless print-to-pdf."""
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge_bin = None
    for p in edge_paths:
        if os.path.exists(p):
            edge_bin = p
            break
            
    if not edge_bin:
        raise RuntimeError("Microsoft Edge executable not found for PDF generation.")

    abs_html = os.path.abspath(html_path)
    abs_pdf = os.path.abspath(raw_pdf_path)
    file_url = f"file:///{abs_html.replace(os.sep, '/')}"

    cmd = [
        edge_bin,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={abs_pdf}",
        file_url
    ]
    print("Executing Edge headless print-to-pdf...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(abs_pdf):
        raise RuntimeError(f"Edge PDF conversion failed. Stderr: {res.stderr}")
    print(f"Raw PDF successfully generated ({os.path.getsize(abs_pdf):,} bytes).")


def stamp_page_numbers_and_footers(raw_pdf_path: str, final_pdf_path: str):
    """Add professional running footer and page numbers to each page."""
    doc = pymupdf.open(raw_pdf_path)
    total_pages = len(doc)
    print(f"Post-processing {total_pages} pages for numbering...")

    for page_idx, page in enumerate(doc):
        rect = page.rect
        width = rect.width
        height = rect.height
        
        # Omit footer from title page (page 0)
        if page_idx == 0:
            continue

        page_num_str = f"Page {page_idx + 1} of {total_pages}"
        left_footer_str = "Linear Regression: Batch GD vs. SGD | Foundations of AI (Woxsen University)"

        # Draw running footer line
        line_y = height - 36
        page.draw_line(
            pymupdf.Point(40, line_y),
            pymupdf.Point(width - 40, line_y),
            color=(0.8, 0.85, 0.9),
            width=0.75
        )

        # Draw left academic footer text
        page.insert_text(
            pymupdf.Point(40, line_y + 16),
            left_footer_str,
            fontsize=8,
            fontname="helv",
            color=(0.4, 0.45, 0.5)
        )

        # Draw right page number
        # Approximate width calculation for right alignment
        num_width = len(page_num_str) * 5.2
        page.insert_text(
            pymupdf.Point(width - 40 - num_width, line_y + 16),
            page_num_str,
            fontsize=8.5,
            fontname="helv",
            color=(0.15, 0.25, 0.4)
        )

    doc.save(final_pdf_path)
    doc.close()
    print(f"Final numbered PDF successfully saved to: {final_pdf_path}")


def main():
    print("=" * 60)
    print(" ACADEMIC REPORT PDF GENERATOR")
    print(" Subject: Foundations of Artificial Intelligence")
    print(" Institution: Woxsen University")
    print("=" * 60)

    html_file = "academic_report_render.html"
    raw_pdf_file = "raw_report_temp.pdf"
    final_pdf_file = "Linear_Regression_GD_vs_SGD_Report.pdf"

    print("1. Generating formatted academic HTML report...")
    html_content = generate_report_html()
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"   HTML written ({len(html_content):,} chars).")

    print("\n2. Converting HTML to PDF via headless Microsoft Edge...")
    convert_html_to_pdf_with_edge(html_file, raw_pdf_file)

    print("\n3. Stamping exact page numbers ('Page X of Y') and footers...")
    stamp_page_numbers_and_footers(raw_pdf_file, final_pdf_file)

    # Copy to brain artifact directory as well
    artifact_dir = r"C:\Users\LENOVO\.gemini\antigravity\brain\6b7064cf-0482-4f8d-a00e-74e0e4e2fb63"
    if os.path.exists(artifact_dir):
        artifact_pdf_path = os.path.join(artifact_dir, "Linear_Regression_GD_vs_SGD_Report.pdf")
        import shutil
        shutil.copy2(final_pdf_file, artifact_pdf_path)
        print(f"   Copied to artifact directory: {artifact_pdf_path}")

    # Clean up temporary raw PDF
    if os.path.exists(raw_pdf_file):
        os.remove(raw_pdf_file)
    print("\nAcademic PDF generation completed successfully!")
    print(f"PDF Location: {os.path.abspath(final_pdf_file)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
