"""
Academic PDF Generation Pipeline
=================================
Course: Foundations of Artificial Intelligence
Institution: School of Technology, Woxsen University
Project: Linear Regression: Batch GD vs. Stochastic GD

This script compiles a comprehensive, publication-quality academic report:
1. Ingests all project results, tables, and 6 high-res figures.
2. Ingests the official Woxsen University logo.
3. Renders clean, organized mathematical equations and pseudocode algorithms.
4. Serializes sections starting from 1. Abstract through 11. References.
5. Employs strict IEEE citation style in-text and in the reference list.
6. Generates an A4 print-ready HTML document with base64 embedded assets.
7. Invokes Microsoft Edge in headless print-to-pdf mode.
8. Post-processes the PDF using PyMuPDF to stamp exact page numbers ('Page X of Y')
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
<title>Linear Regression Using Batch Gradient Descent & Stochastic Gradient Descent - Academic Report</title>
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
    color: #000000;
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
    color: #000000;
    font-weight: 700;
    page-break-after: avoid;
    break-after: avoid;
  }}

  h1 {{ font-size: 20pt; line-height: 1.25; margin-bottom: 8pt; color: #000000; }}
  h2 {{ font-size: 13.5pt; margin-top: 14pt; margin-bottom: 6pt; border-bottom: 1.5pt solid #000000; padding-bottom: 3pt; color: #000000; }}
  h3 {{ font-size: 11pt; margin-top: 10pt; margin-bottom: 4pt; color: #000000; }}
  h4 {{ font-size: 9.5pt; margin-top: 8pt; margin-bottom: 4pt; color: #000000; }}

  p {{
    margin-bottom: 7pt;
    text-align: justify;
    color: #000000;
  }}

  ul, ol {{
    margin-left: 18pt;
    margin-bottom: 7pt;
    color: #000000;
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

  .report-main-title {{
    font-size: 23pt;
    font-weight: 800;
    color: #000000;
    line-height: 1.2;
    margin-bottom: 10pt;
  }}

  .report-subtitle {{
    font-size: 12pt;
    color: #333333;
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
    background: #ffffff;
    border: 1pt solid #000000;
    border-radius: 6pt;
    padding: 14pt;
  }}

  .meta-card-title {{
    font-size: 10pt;
    font-weight: 700;
    color: #000000;
    margin-bottom: 8pt;
    text-transform: uppercase;
    letter-spacing: 0.8pt;
    border-bottom: 1pt solid #000000;
    padding-bottom: 3pt;
  }}

  .team-member {{
    display: flex;
    justify-content: space-between;
    padding: 4pt 0;
    border-bottom: 0.5pt dashed #d1d5db;
    font-size: 9.5pt;
  }}

  .team-member:last-child {{
    border-bottom: none;
  }}

  .member-name {{
    font-weight: 600;
    color: #000000;
  }}

  .member-id {{
    font-family: monospace;
    font-weight: 700;
    color: #000000;
  }}

  .meta-info-row {{
    margin-bottom: 6pt;
    font-size: 9.5pt;
  }}

  .meta-info-label {{
    color: #555555;
    font-size: 8pt;
    text-transform: uppercase;
    font-weight: 600;
  }}

  .meta-info-val {{
    font-weight: 700;
    color: #000000;
  }}

  .title-footer {{
    margin-top: 25pt;
    border-top: 1pt solid #000000;
    padding-top: 12pt;
    font-size: 8.5pt;
    color: #333333;
  }}

  /* Tables */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8pt 0;
    font-size: 8.8pt;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  th, td {{
    padding: 5pt 7pt;
    border: 1pt solid #000000;
    text-align: left;
    color: #000000;
  }}

  th {{
    background: #000000;
    color: #ffffff;
    font-weight: 700;
  }}

  tr:nth-child(even) {{
    background: #f9fafb;
  }}

  /* Math Blocks & Equation Cards */
  .math-card {{
    background: #ffffff;
    border: 1pt solid #000000;
    border-left: 4pt solid #000000;
    border-radius: 4pt;
    padding: 8pt 12pt;
    margin: 8pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .math-card-header {{
    font-size: 9pt;
    font-weight: 700;
    color: #000000;
    margin-bottom: 4pt;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .eq-tag {{
    font-size: 7.5pt;
    font-family: monospace;
    background: #f3f4f6;
    color: #000000;
    padding: 1pt 5pt;
    border-radius: 3pt;
    border: 0.5pt solid #000000;
  }}

  .math-formula {{
    font-family: "Cambria Math", "Latin Modern Math", "Times New Roman", serif;
    font-size: 11.5pt;
    text-align: center;
    padding: 4pt 0;
    color: #000000;
  }}

  .math-details {{
    margin-top: 4pt;
    font-size: 8.2pt;
    color: #1f2937;
    border-top: 0.5pt solid #d1d5db;
    padding-top: 4pt;
  }}

  /* Algorithm Box */
  .algo-card {{
    background: #ffffff;
    border: 1pt solid #000000;
    border-radius: 4pt;
    margin: 8pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .algo-header {{
    background: #000000;
    color: #ffffff;
    padding: 5pt 9pt;
    font-size: 9pt;
    font-weight: 700;
    border-top-left-radius: 3pt;
    border-top-right-radius: 3pt;
  }}

  .algo-body {{
    padding: 7pt 10pt;
    font-size: 8.2pt;
    line-height: 1.45;
    color: #000000;
  }}

  .algo-step {{
    display: flex;
    margin-bottom: 3pt;
  }}

  .algo-step-num {{
    width: 20pt;
    font-weight: 700;
    color: #000000;
    font-family: monospace;
    flex-shrink: 0;
  }}

  .algo-step-content {{
    flex: 1;
    color: #000000;
  }}

  /* Figure Styling */
  .figure-container {{
    margin: 10pt 0;
    text-align: center;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .figure-img {{
    width: 95%;
    max-height: 270pt;
    object-fit: contain;
    border: 1pt solid #000000;
    border-radius: 4pt;
    background: #ffffff;
    box-shadow: 0 1pt 3pt rgba(0,0,0,0.06);
  }}

  .figure-caption {{
    font-size: 8.2pt;
    color: #333333;
    margin-top: 4pt;
    font-style: italic;
  }}

  .figure-caption strong {{
    color: #000000;
    font-style: normal;
  }}

  /* Callout Boxes */
  .callout {{
    background: #f9fafb;
    border: 1pt solid #000000;
    border-left: 3.5pt solid #000000;
    padding: 7pt 9pt;
    border-radius: 4pt;
    margin: 7pt 0;
    font-size: 8.8pt;
    color: #000000;
  }}

  .callout-success {{
    background: #f9fafb;
    border: 1pt solid #000000;
    border-left: 3.5pt solid #000000;
    color: #000000;
  }}

  .cite {{
    font-weight: 700;
    color: #000000;
  }}

  code {{
    font-family: Consolas, monospace;
    background: #f3f4f6;
    padding: 1pt 4pt;
    border-radius: 3pt;
    font-size: 8.5pt;
    color: #000000;
    border: 0.5pt solid #9ca3af;
  }}
</style>
</head>
<body>

  <!-- ========================================================= -->
  <!-- TITLE PAGE                                                -->
  <!-- ========================================================= -->
  <div class="title-page">
    <div class="title-logo-container">
      <img src="{logo_b64}" alt="Woxsen University Logo" class="title-logo">
    </div>

    <div>
      <h1 class="report-main-title">Linear Regression Using Batch Gradient Descent & Stochastic Gradient Descent</h1>
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
          <div class="meta-info-label">Institution</div>
          <div class="meta-info-val">School of Technology, Woxsen University</div>
        </div>
      </div>
    </div>

    <div class="title-footer">
      <div>Submission Date: October 11, 2026 | Academic Term: 2026</div>
      <div>GitHub Repository: https://github.com/sahajasakhunala/linear-regression-gd-sgd</div>
    </div>
  </div>

  <div class="page-break"></div>

  <!-- ========================================================= -->
  <!-- 1. ABSTRACT                                               -->
  <!-- ========================================================= -->
  <h2>1. Abstract</h2>
  <p>
    Optimization algorithms constitute the mathematical engine of modern artificial intelligence and statistical learning [1], [2]. In this project, an end-to-end empirical and theoretical comparative study between Batch Gradient Descent (GD) and Stochastic Gradient Descent (SGD) is conducted for univariate Linear Regression. Built entirely from first mathematical principles using Python and NumPy [5], the system eliminates reliance on high-level library estimators (such as scikit-learn's <code>LinearRegression</code> or <code>SGDRegressor</code>) in order to make every intermediate computational step transparent: forward hypothesis generation, Mean Squared Error (MSE) loss computation, closed-form analytical gradient calculation, and iterative parameter updates.
  </p>
  <p>
    The experiment is conducted on a canonical 15-observation academic dataset tracking the relationship between study hours and exam scores. Methodological rigor is maintained via an isolated 70/30 train/test partition (10 training observations, 5 held-out test observations) under seed 42 with zero data leakage. Both iterative optimizers are evaluated against the closed-form Ordinary Least Squares (OLS) analytical reference [3]. The experimental results demonstrate that both algorithms successfully recover the underlying data-generating relationship: Batch GD converges to <strong>ŷ = 7.7822x + 28.2121</strong>, while SGD converges to <strong>ŷ = 7.7156x + 28.5394</strong>. On unseen test data, both models achieve identical generalization fidelity (test R² = 0.9973, test RMSE ≈ 0.93 marks). Crucially, SGD achieves equivalent predictive accuracy while requiring an order of magnitude fewer individual sample evaluations (1,000 versus 10,000 for Batch GD), illustrating fundamental trade-offs between deterministic and stochastic optimization [2], [4].
  </p>

  <!-- ========================================================= -->
  <!-- 2. PROBLEM STATEMENT                                      -->
  <!-- ========================================================= -->
  <h2>2. Problem Statement</h2>
  <p>
    Contemporary machine learning practice frequently treats model fitting as an opaque API invocation. While abstractions accelerate rapid prototyping, they obscure the underlying mechanics of optimization: loss landscape topology, learning rate sensitivity, gradient variance, and parameter trajectory stability [1], [4]. When optimization is treated as a black-box, practitioners miss foundational engineering insights into how individual data points guide parameter updates, why stochastic updates introduce variance, and how data leakage distorts empirical validation [2].
  </p>
  <p>
    To resolve this pedagogical challenge, this project develops a transparent, from-scratch implementation of Linear Regression. By deriving gradients analytically, implementing update rules manually, maintaining strict test-set isolation, and comparing Batch GD against SGD, this study provides complete clarity into how Linear Regression learns from data.
  </p>

  <!-- ========================================================= -->
  <!-- 3. INTRODUCTION                                           -->
  <!-- ========================================================= -->
  <h2>3. Introduction</h2>
  <p>
    Supervised regression identifies continuous functional mappings between input features and target outcomes. In its univariate formulation, Linear Regression represents the bedrock foundation of statistical learning and predictive modeling [1], [3]:
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
    While small linear systems can be solved directly via the Normal Equation / Ordinary Least Squares closed-form solution [3], iterative first-order optimization via Gradient Descent forms the universal backbone for training deep neural networks, transformers, and large-scale artificial intelligence architectures where analytical matrix inversion is computationally intractable [2].
  </p>

  <!-- ========================================================= -->
  <!-- 4. PROJECT OBJECTIVES                                     -->
  <!-- ========================================================= -->
  <h2>4. Project Objectives</h2>
  <ul>
    <li><strong>Mathematical Formulation:</strong> Formulate the linear regression hypothesis and derive analytical partial derivatives of Mean Squared Error (MSE) with respect to weight (w) and bias (b) [1].</li>
    <li><strong>First-Principles Implementation:</strong> Develop Batch GD and SGD completely from scratch in Python [8] using vectorized NumPy routines [5], avoiding black-box library fits.</li>
    <li><strong>Zero Data Leakage Safeguards:</strong> Partition data into 70% training (n=10) and 30% testing (n=5) subsets with strict isolation during fitting [3].</li>
    <li><strong>Comparative Trajectory Tracking:</strong> Record iteration-by-iteration and epoch-by-epoch loss trajectories to contrast deterministic versus stochastic convergence [4].</li>
    <li><strong>Comprehensive Generalization Evaluation:</strong> Evaluate models on unseen test observations using MSE, RMSE, and R² alongside closed-form OLS solutions.</li>
    <li><strong>Publication-Quality Visualization:</strong> Generate six high-resolution visual figures illustrating dataset partition, fitted lines, loss curves, and actual versus predicted test scores using Matplotlib [7].</li>
  </ul>

  <!-- ========================================================= -->
  <!-- 5. THEORETICAL BACKGROUND (ORGANIZED MATHEMATICS)         -->
  <!-- ========================================================= -->
  <h2>5. Theoretical Background & Mathematical Derivations</h2>
  <p>
    To make the mathematics intuitive and structured, all equations, variables, and derivations are organized systematically below following standard statistical learning principles [1], [3].
  </p>

  <h3>5.1 Master Variable Definition Reference</h3>
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

  <h3>5.2 Mean Squared Error (MSE) Objective Function</h3>
  <div class="math-card">
    <div class="math-card-header">
      <span>Mean Squared Error (MSE) Formula</span>
      <span class="eq-tag">Equation 2</span>
    </div>
    <div class="math-formula">
      J(w, b) = <span style="font-size: 11pt;">1</span>/<span style="font-size: 9pt;">n</span> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - ŷ<sub>i</sub>)<sup>2</sup> = <span style="font-size: 11pt;">1</span>/<span style="font-size: 9pt;">n</span> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - (w·x<sub>i</sub> + b))<sup>2</sup>
    </div>
    <div class="math-details">
      <strong>Why MSE is selected [1], [3]:</strong> (1) <em>Strict Convexity</em> guarantees a unique global minimum with zero local minima traps; (2) <em>Continuous Differentiability</em> produces smooth analytical gradients everywhere; (3) <em>Quadratic Penalty</em> strongly discourages large residual errors; (4) Minimizing MSE is mathematically equivalent to Maximum Likelihood Estimation (MLE) under Gaussian error assumptions.
    </div>
  </div>

  <h3>5.3 Step-by-Step Analytical Gradient Derivations</h3>
  <p>
    The gradient vector ∇J(w, b) is computed by taking partial derivatives of J(w, b) using the calculus chain rule [2]:
  </p>

  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8pt; margin: 5pt 0;">
    <div class="math-card" style="margin: 0; padding: 6pt 10pt;">
      <div class="math-card-header">
        <span>Weight Gradient (∂J / ∂w)</span>
        <span class="eq-tag">Eq. 3a</span>
      </div>
      <div style="font-family: 'Cambria Math', serif; font-size: 8.8pt; line-height: 1.5;">
        1. Chain rule: <sup>∂J</sup>/<sub>∂w</sub> = <sup>1</sup>/<sub>n</sub> ∑ 2(y<sub>i</sub> - ŷ<sub>i</sub>) · (-x<sub>i</sub>)<br>
        2. Closed form:<br>
        <strong><sup>∂J</sup>/<sub>∂w</sub> = -<sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> x<sub>i</sub> · (y<sub>i</sub> - ŷ<sub>i</sub>)</strong>
      </div>
    </div>

    <div class="math-card" style="margin: 0; padding: 6pt 10pt;">
      <div class="math-card-header">
        <span>Bias Gradient (∂J / ∂b)</span>
        <span class="eq-tag">Eq. 3b</span>
      </div>
      <div style="font-family: 'Cambria Math', serif; font-size: 8.8pt; line-height: 1.5;">
        1. Chain rule: <sup>∂J</sup>/<sub>∂b</sub> = <sup>1</sup>/<sub>n</sub> ∑ 2(y<sub>i</sub> - ŷ<sub>i</sub>) · (-1)<br>
        2. Closed form:<br>
        <strong><sup>∂J</sup>/<sub>∂b</sub> = -<sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - ŷ<sub>i</sub>)</strong>
      </div>
    </div>
  </div>

  <div style="page-break-inside: avoid; break-inside: avoid; margin-top: 6pt;">
    <h3 style="margin-top: 6pt; margin-bottom: 3pt;">5.4 Master Comparison: Batch GD vs. Stochastic GD Equations</h3>
    <p style="margin-bottom: 4pt; font-size: 9.5pt;">
      The mathematical formulation of parameter updates differs fundamentally in whether gradients are computed over the full dataset or a single observation [2], [4]:
    </p>

    <table style="margin: 4pt 0; font-size: 8pt;">
      <thead>
        <tr>
          <th style="width: 25%; padding: 3pt 5pt;">Mathematical Property</th>
          <th style="width: 37%; padding: 3pt 5pt;">Batch Gradient Descent (GD)</th>
          <th style="width: 38%; padding: 3pt 5pt;">Stochastic Gradient Descent (SGD) [4]</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding: 2.5pt 5pt;"><strong>Data Scope per Step</strong></td>
          <td style="padding: 2.5pt 5pt;">Entire Training Batch (all n = 10 samples)</td>
          <td style="padding: 2.5pt 5pt;">Single Observation (sample i ∈ {{1, ..., n}})</td>
        </tr>
        <tr>
          <td style="padding: 2.5pt 5pt;"><strong>Objective Function</strong></td>
          <td style="padding: 2.5pt 5pt;">J(w, b) = <sup>1</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - ŷ<sub>i</sub>)<sup>2</sup></td>
          <td style="padding: 2.5pt 5pt;">J<sub>i</sub>(w, b) = (y<sub>i</sub> - ŷ<sub>i</sub>)<sup>2</sup></td>
        </tr>
        <tr>
          <td style="padding: 2.5pt 5pt;"><strong>Weight Gradient (∇<sub>w</sub>)</strong></td>
          <td style="padding: 2.5pt 5pt;"><sup>∂J</sup>/<sub>∂w</sub> = -<sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> x<sub>i</sub>(y<sub>i</sub> - ŷ<sub>i</sub>)</td>
          <td style="padding: 2.5pt 5pt;"><sup>∂J<sub>i</sub></sup>/<sub>∂w</sub> = -2 x<sub>i</sub>(y<sub>i</sub> - ŷ<sub>i</sub>)</td>
        </tr>
        <tr>
          <td style="padding: 2.5pt 5pt;"><strong>Bias Gradient (∇<sub>b</sub>)</strong></td>
          <td style="padding: 2.5pt 5pt;"><sup>∂J</sup>/<sub>∂b</sub> = -<sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - ŷ<sub>i</sub>)</td>
          <td style="padding: 2.5pt 5pt;"><sup>∂J<sub>i</sub></sup>/<sub>∂b</sub> = -2 (y<sub>i</sub> - ŷ<sub>i</sub>)</td>
        </tr>
        <tr>
          <td style="padding: 2.5pt 5pt;"><strong>Weight Update Rule</strong></td>
          <td style="padding: 2.5pt 5pt;">w ← w + α · <sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> x<sub>i</sub>(y<sub>i</sub> - ŷ<sub>i</sub>)</td>
          <td style="padding: 2.5pt 5pt;">w ← w + 2α · x<sub>i</sub>(y<sub>i</sub> - ŷ<sub>i</sub>)</td>
        </tr>
        <tr>
          <td style="padding: 2.5pt 5pt;"><strong>Bias Update Rule</strong></td>
          <td style="padding: 2.5pt 5pt;">b ← b + α · <sup>2</sup>/<sub>n</sub> ∑<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - ŷ<sub>i</sub>)</td>
          <td style="padding: 2.5pt 5pt;">b ← b + 2α · (y<sub>i</sub> - ŷ<sub>i</sub>)</td>
        </tr>
        <tr>
          <td style="padding: 2.5pt 5pt;"><strong>Gradient Estimation</strong></td>
          <td style="padding: 2.5pt 5pt;">Exact deterministic gradient vector</td>
          <td style="padding: 2.5pt 5pt;">Noisy, unbiased stochastic gradient estimate</td>
        </tr>
        <tr>
          <td style="padding: 2.5pt 5pt;"><strong>Updates per Epoch</strong></td>
          <td style="padding: 2.5pt 5pt;">1 update per pass over dataset</td>
          <td style="padding: 2.5pt 5pt;">n = 10 updates per pass (shuffled per epoch)</td>
        </tr>
        <tr>
          <td style="padding: 2.5pt 5pt;"><strong>Total Sample Work</strong></td>
          <td style="padding: 2.5pt 5pt;">1,000 steps × 10 samples = <strong>10,000 evaluations</strong></td>
          <td style="padding: 2.5pt 5pt;">100 epochs × 10 samples = <strong>1,000 evaluations</strong></td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- ========================================================= -->
  <!-- 6. SYSTEM DESIGN & ARCHITECTURE                           -->
  <!-- ========================================================= -->
  <div style="page-break-inside: avoid; break-inside: avoid;">
    <h2 style="margin-top: 8pt; margin-bottom: 4pt;">6. System Architecture & Modularity</h2>
    <p style="margin-bottom: 5pt; font-size: 8.8pt;">
      The codebase is organized into decoupled modules following software engineering best practices:
    </p>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4pt 8pt; font-size: 7.8pt; line-height: 1.35;">
      <div style="background: #ffffff; border: 1pt solid #000000; border-radius: 4pt; padding: 3pt 6pt; color: #000000;">
        <strong><code>src/data_preprocessing.py</code></strong>: Data loading, audits, and isolated 70/30 partitioning.
      </div>
      <div style="background: #ffffff; border: 1pt solid #000000; border-radius: 4pt; padding: 3pt 6pt; color: #000000;">
        <strong><code>src/linear_regression.py</code></strong>: First-principles prediction ŷ = wx + b, MSE, and R².
      </div>
      <div style="background: #ffffff; border: 1pt solid #000000; border-radius: 4pt; padding: 3pt 6pt; color: #000000;">
        <strong><code>src/gradient_descent.py</code></strong>: Batch GD with parameter and loss trajectory tracking.
      </div>
      <div style="background: #ffffff; border: 1pt solid #000000; border-radius: 4pt; padding: 3pt 6pt; color: #000000;">
        <strong><code>src/stochastic_gradient_descent.py</code></strong>: SGD with random index shuffling and epoch logging.
      </div>
      <div style="background: #ffffff; border: 1pt solid #000000; border-radius: 4pt; padding: 3pt 6pt; color: #000000;">
        <strong><code>src/evaluation.py</code></strong>: Evaluation routines and closed-form OLS analytical baseline [3].
      </div>
      <div style="background: #ffffff; border: 1pt solid #000000; border-radius: 4pt; padding: 3pt 6pt; color: #000000;">
        <strong><code>src/visualization.py</code></strong>: Six publication-quality figures via matplotlib Agg [7].
      </div>
      <div style="background: #ffffff; border: 1pt solid #000000; border-radius: 4pt; padding: 3pt 6pt; color: #000000;">
        <strong><code>main.py</code></strong>: Master orchestration script executing training, evaluation, and exports.
      </div>
      <div style="background: #ffffff; border: 1pt solid #000000; border-radius: 4pt; padding: 3pt 6pt; color: #000000;">
        <strong><code>predict.py</code></strong>: Interactive CLI tool for live score prediction on arbitrary study hours.
      </div>
    </div>
  </div>

  <!-- ========================================================= -->
  <!-- 7. IMPLEMENTATION DETAILS                                 -->
  <!-- ========================================================= -->
  <h2>7. Implementation Details</h2>
  <h3>7.1 Dataset Partitioning & Zero Leakage Audit</h3>
  <p>
    The dataset contains 15 observations partitioned via scikit-learn's <code>train_test_split(test_size=0.3, random_state=42)</code>:
  </p>
  <ul>
    <li><strong>Training Partition (70%, n=10):</strong> Hours = [5.0, 2.0, 1.5, 8.0, 3.0, 4.5, 6.0, 7.0, 2.5, 4.0]; Scores = [68, 44, 39, 89, 52, 64, 75, 82, 48, 61].</li>
    <li><strong>Held-out Test Partition (30%, n=5):</strong> Hours = [1.0, 3.5, 5.5, 6.5, 7.5]; Scores = [35, 55, 71, 78, 85].</li>
  </ul>
  <div class="callout callout-success">
    <strong>Zero Data Leakage Safeguard:</strong> The 5 test observations are completely quarantined during optimization. No gradients, residuals, or loss updates are evaluated on these samples while fitting parameters. They are evaluated strictly post-training to measure generalization [3].
  </div>

  <h3>7.2 Structured Algorithmic Pseudocode</h3>
  <p>
    The mathematical logic of both algorithms is formatted in structured, step-by-step procedures below:
  </p>

  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8pt; margin-top: 6pt;">
    <div class="algo-card" style="margin: 0;">
      <div class="algo-header">Algorithm 1: Batch Gradient Descent</div>
      <div class="algo-body" style="padding: 5pt 7pt; font-size: 7.6pt; line-height: 1.35;">
        <div class="algo-step"><span class="algo-step-num" style="width: 14pt;">1.</span><span class="algo-step-content"><strong>Input:</strong> X ∈ ℝ<sup>n</sup>, y ∈ ℝ<sup>n</sup>, α = 0.01, K = 1000.</span></div>
        <div class="algo-step"><span class="algo-step-num" style="width: 14pt;">2.</span><span class="algo-step-content"><strong>Init:</strong> w = 0.0, b = 0.0, n = length(X).</span></div>
        <div class="algo-step"><span class="algo-step-num" style="width: 14pt;">3.</span><span class="algo-step-content"><strong>Loop k = 1 to K:</strong><br>
          a. Predict: ŷ = w · X + b<br>
          b. Loss: J = (1/n) ∑ (y - ŷ)²<br>
          c. Full-batch gradients:<br>
          &nbsp;&nbsp;&nbsp;&nbsp;<strong>∇<sub>w</sub> = (-2/n) ∑ X · (y - ŷ)</strong><br>
          &nbsp;&nbsp;&nbsp;&nbsp;<strong>∇<sub>b</sub> = (-2/n) ∑ (y - ŷ)</strong><br>
          d. Update parameters:<br>
          &nbsp;&nbsp;&nbsp;&nbsp;<strong>w ← w - α · ∇<sub>w</sub></strong><br>
          &nbsp;&nbsp;&nbsp;&nbsp;<strong>b ← b - α · ∇<sub>b</sub></strong><br>
          e. Record J, w, b in history.</span></div>
        <div class="algo-step"><span class="algo-step-num" style="width: 14pt;">4.</span><span class="algo-step-content"><strong>Output:</strong> Converged (w, b), history.</span></div>
      </div>
    </div>

    <div class="algo-card" style="margin: 0;">
      <div class="algo-header">Algorithm 2: Stochastic Gradient Descent</div>
      <div class="algo-body" style="padding: 5pt 7pt; font-size: 7.6pt; line-height: 1.35;">
        <div class="algo-step"><span class="algo-step-num" style="width: 14pt;">1.</span><span class="algo-step-content"><strong>Input:</strong> (X, y), α = 0.01, E = 100, seed = 42.</span></div>
        <div class="algo-step"><span class="algo-step-num" style="width: 14pt;">2.</span><span class="algo-step-content"><strong>Init:</strong> w = 0.0, b = 0.0, n = length(X).</span></div>
        <div class="algo-step"><span class="algo-step-num" style="width: 14pt;">3.</span><span class="algo-step-content"><strong>Loop epoch e = 1 to E:</strong><br>
          a. Shuffle indices: π = Shuffle([0..n-1])<br>
          b. <strong>For each index i in π:</strong><br>
          &nbsp;&nbsp;&nbsp;&nbsp;i. Sample: (x<sub>i</sub>, y<sub>i</sub>)<br>
          &nbsp;&nbsp;&nbsp;&nbsp;ii. Predict: ŷ<sub>i</sub> = w · x<sub>i</sub> + b<br>
          &nbsp;&nbsp;&nbsp;&nbsp;iii. Sample gradient:<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>∇<sub>w,i</sub> = -2 x<sub>i</sub> (y<sub>i</sub> - ŷ<sub>i</sub>)</strong><br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>∇<sub>b,i</sub> = -2 (y<sub>i</sub> - ŷ<sub>i</sub>)</strong><br>
          &nbsp;&nbsp;&nbsp;&nbsp;iv. Instant update:<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>w ← w - α ∇<sub>w,i</sub></strong>, <strong>b ← b - α ∇<sub>b,i</sub></strong><br>
          c. Epoch loss: J<sub>epoch</sub> = (1/n) ∑ (y - ŷ)²<br>
          d. Record J<sub>epoch</sub>, w, b in history.</span></div>
        <div class="algo-step"><span class="algo-step-num" style="width: 14pt;">4.</span><span class="algo-step-content"><strong>Output:</strong> Converged (w, b), history.</span></div>
      </div>
    </div>
  </div>

  <div class="page-break"></div>

  <!-- ========================================================= -->
  <!-- 8. EXPERIMENTAL RESULTS AND ANALYSIS                      -->
  <!-- ========================================================= -->
  <h2>8. Experimental Results & Comparative Analysis</h2>

  <h3>8.1 Quantitative Performance Metrics Table</h3>
  <p>
    The quantitative comparison between Batch Gradient Descent, Stochastic Gradient Descent, and the analytical Ordinary Least Squares solution [3] is presented below:
  </p>
  <table>
    <thead>
      <tr>
        <th>Performance Metric</th>
        <th>Batch Gradient Descent (GD)</th>
        <th>Stochastic Gradient Descent (SGD) [4]</th>
        <th>Analytical OLS (Reference) [3]</th>
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

  <h3>8.2 Held-Out Test Set Predictions Table</h3>
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

  <h3>8.3 Graphical Analysis of the Six Visualizations</h3>

  <div class="figure-container">
    <img src="{fig1_b64}" alt="Figure 1: Dataset Partition Scatter Plot" class="figure-img">
    <div class="figure-caption"><strong>Figure 1: Dataset Partition Scatter Plot.</strong> 10 training samples (filled circles) and 5 held-out test samples (open squares) distributed across 1.0 to 8.0 study hours without overlap.</div>
  </div>

  <div class="figure-container">
    <img src="{fig2_b64}" alt="Figure 2: Fitted Regression Lines" class="figure-img">
    <div class="figure-caption"><strong>Figure 2: Fitted Regression Lines (Batch GD vs. SGD).</strong> Batch GD line (solid black) and SGD line (dashed line) overlap almost indistinguishably, confirming identical convergence.</div>
  </div>

  <div class="page-break"></div>

  <div class="figure-container">
    <img src="{fig3_b64}" alt="Figure 3: Batch GD Loss Curve" class="figure-img">
    <div class="figure-caption"><strong>Figure 3: Batch Gradient Descent Loss Curve.</strong> Strictly monotonic, smooth descent from initial MSE of 4117.60 down to 0.7892 over 1,000 iterations.</div>
  </div>

  <div class="figure-container">
    <img src="{fig4_b64}" alt="Figure 4: SGD Loss Curve" class="figure-img">
    <div class="figure-caption"><strong>Figure 4: Stochastic Gradient Descent Loss Curve.</strong> Rapid initial convergence in the first 20 epochs followed by characteristic stochastic fluctuations around the minimum [4].</div>
  </div>

  <div class="page-break"></div>

  <div class="figure-container">
    <img src="{fig5_b64}" alt="Figure 5: Comparative Optimization Trajectories" class="figure-img">
    <div class="figure-caption"><strong>Figure 5: Optimization Trajectories Compared.</strong> Side-by-side contrast between deterministic smooth descent (left) and stochastic fluctuating descent (right).</div>
  </div>

  <div class="figure-container">
    <img src="{fig6_b64}" alt="Figure 6: Unseen Test Set Predictions" class="figure-img">
    <div class="figure-caption"><strong>Figure 6: Actual vs. Predicted Scores on Unseen Test Data.</strong> Grouped bar chart in grayscale demonstrating tight alignment between actual exam scores and model predictions across all 5 test students.</div>
  </div>

  <div class="page-break"></div>

  <!-- ========================================================= -->
  <!-- 9. DISCUSSION AND LIMITATIONS                             -->
  <!-- ========================================================= -->
  <h2>9. Discussion & Limitations</h2>

  <h3>9.1 Convergence Dynamics and Gradient Variance</h3>
  <p>
    The experimental results illustrate the core mathematical contrast between full-batch and stochastic gradient estimation [2], [4]. In Batch GD, the gradient vector reflects the true empirical risk across all training observations:
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
    Because individual observations deviate from the sample mean, the stochastic gradient acts as a noisy estimate of the true gradient. While this introduces variance (visible as epoch-by-epoch fluctuations in Figure 4 and Figure 5), it prevents the optimizer from stagnating and drives rapid early convergence [2].
  </p>

  <h3>9.2 Computational Work & Scalability</h3>
  <p>
    A critical finding of this study is the sample evaluation efficiency of SGD. Batch GD required <strong>10,000 individual sample evaluations</strong> (1,000 iterations × 10 samples) to achieve test MSE = 0.8636. In contrast, SGD required only <strong>1,000 sample evaluations</strong> (100 epochs × 10 samples) to achieve test MSE = 0.8650. This represents a <strong>10× reduction in computational sample work</strong> for essentially identical predictive accuracy. On large-scale industry datasets (n ≥ 10⁶), Batch GD becomes computationally prohibitive per update, whereas SGD or Mini-batch GD makes rapid initial progress [4].
  </p>

  <h3>9.3 Methodological Limitations</h3>
  <ul>
    <li><strong>Dataset Scale (n=15):</strong> The 15-observation dataset provides transparency for step-by-step tracing, but lacks the scale necessary to test complex regularization or higher-order variance effects.</li>
    <li><strong>Test Set Sensitivity (n<sub>test</sub>=5):</strong> Because the held-out test partition comprises only 5 observations, summary metrics (MSE, RMSE, R²) are sensitive to individual residual deviations. While our average absolute error is low (~0.77 marks), larger test cohorts are recommended for statistical confidence intervals.</li>
    <li><strong>Univariate Linearity:</strong> With only one input feature (study hours), collinearity, matrix conditioning, and multi-dimensional saddle points are not encountered.</li>
  </ul>

  <!-- ========================================================= -->
  <!-- 10. CONCLUSION & FUTURE WORK                              -->
  <!-- ========================================================= -->
  <h2>10. Conclusion & Future Work</h2>
  <p>
    This investigation successfully implemented, derived, and compared Batch Gradient Descent and Stochastic Gradient Descent from first principles for univariate Linear Regression.
  </p>
  <div class="callout callout-success">
    <strong>Key Conclusions:</strong>
    <ol>
      <li><strong>Convergence Parity:</strong> Both algorithms successfully converged to the underlying data trend, arriving at nearly identical parameters (Batch GD: w=7.7822, b=28.2121; SGD: w=7.7156, b=28.5394).</li>
      <li><strong>Generalization Parity:</strong> Both models generalized with high fidelity on unseen test observations, achieving matching test R² = 0.9973 and RMSE ≈ 0.93 marks.</li>
      <li><strong>Sample Efficiency:</strong> SGD achieved equivalent predictive performance with 10× fewer sample evaluations, empirically validating the efficiency of stochastic optimization [4].</li>
    </ol>
  </div>
  <p>
    <strong>Future Work:</strong> Future extensions include: (1) Implementing <em>Mini-Batch Gradient Descent</em> to balance vectorization speed and stochastic regularization; (2) Adding <em>adaptive learning rate schedules</em> (e.g., learning rate decay or Adam [2]) to damp late-stage SGD oscillations; and (3) Scaling to multivariate datasets with feature normalization.
  </p>

  <!-- ========================================================= -->
  <!-- 11. REFERENCES (IEEE FORMAT)                              -->
  <!-- ========================================================= -->
  <h2>11. References</h2>
  <div style="font-size: 8.5pt; line-height: 1.6; margin-top: 6pt;">
    <p style="margin-bottom: 5pt; text-align: left;">[1] C. M. Bishop, <em>Pattern Recognition and Machine Learning</em>. New York, NY, USA: Springer, 2006.</p>
    <p style="margin-bottom: 5pt; text-align: left;">[2] I. Goodfellow, Y. Bengio, and A. Courville, <em>Deep Learning</em>. Cambridge, MA, USA: MIT Press, 2016.</p>
    <p style="margin-bottom: 5pt; text-align: left;">[3] T. Hastie, R. Tibshirani, and J. Friedman, <em>The Elements of Statistical Learning: Data Mining, Inference, and Prediction</em>, 2nd ed. New York, NY, USA: Springer, 2009.</p>
    <p style="margin-bottom: 5pt; text-align: left;">[4] L. Bottou, "Large-scale machine learning with stochastic gradient descent," in <em>Proc. COMPSTAT'2010</em>, Paris, France, 2010, pp. 177–186.</p>
    <p style="margin-bottom: 5pt; text-align: left;">[5] C. R. Harris <em>et al.</em>, "Array programming with NumPy," <em>Nature</em>, vol. 585, no. 7825, pp. 357–362, 2020.</p>
    <p style="margin-bottom: 5pt; text-align: left;">[6] W. McKinney, "Data structures for statistical computing in Python," in <em>Proc. 9th Python Sci. Conf.</em>, Austin, TX, USA, 2010, pp. 56–61.</p>
    <p style="margin-bottom: 5pt; text-align: left;">[7] J. D. Hunter, "Matplotlib: A 2D graphics environment," <em>Comput. Sci. Eng.</em>, vol. 9, no. 3, pp. 90–95, 2007.</p>
    <p style="margin-bottom: 5pt; text-align: left;">[8] Python Software Foundation, <em>The Python Standard Library Documentation</em>, Python 3.10, 2026. [Online]. Available: https://docs.python.org/3/</p>
  </div>

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
        left_footer_str = "Linear Regression: Batch GD vs. SGD | Foundations of AI"

        # Draw running footer line
        line_y = height - 36
        page.draw_line(
            pymupdf.Point(40, line_y),
            pymupdf.Point(width - 40, line_y),
            color=(0.7, 0.7, 0.7),
            width=0.75
        )

        # Draw left academic footer text
        page.insert_text(
            pymupdf.Point(40, line_y + 16),
            left_footer_str,
            fontsize=8,
            fontname="helv",
            color=(0.2, 0.2, 0.2)
        )

        # Draw right page number
        num_width = len(page_num_str) * 5.2
        page.insert_text(
            pymupdf.Point(width - 40 - num_width, line_y + 16),
            page_num_str,
            fontsize=8.5,
            fontname="helv",
            color=(0.0, 0.0, 0.0)
        )

    doc.save(final_pdf_path)
    doc.close()
    print(f"Final numbered PDF successfully saved to: {final_pdf_path}")


def main():
    print("=" * 60)
    print(" ACADEMIC REPORT PDF GENERATOR")
    print(" Subject: Foundations of Artificial Intelligence")
    print(" Institution: School of Technology, Woxsen University")
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
