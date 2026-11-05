"""
Generate stochastic instances from selected deterministic CFLP files (CV-driven).

This version parameterizes scenario generation by the **target Coefficient of
Variation (CV = sigma / mean)** and a chosen distribution. It also records the
**empirical CV** achieved after generation (which can differ slightly due to
finite sampling and any zero-clipping of negatives) and writes it to the output
file. Additionally, it computes a single vector-level dispersion metric
**R_overall = avg_i (max_s d_{i,s} - min_s d_{i,s}) / mean_s d_{i,s}**.

Key notes:
- Normal & Uniform may produce negatives for large CV or small means; we
  clamp at zero. Zero-clipping changes the empirical CV and R_overall.
- Capacity scaling (to ensure sum of demands <= total capacity) does NOT
  change CV or R_overall, because it scales both numerator and denominator.
- If you want guaranteed non-negativity without distortion, prefer Gamma or
  Lognormal parameterizations that match (mean, CV).

Author: Your Name
"""

import os
import numpy as np
import pickle
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from scipy.linalg import eigvals
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd
from datetime import datetime

# Module-level parameter for MERVE_WITH_COVARIATES method
_num_covariates_merve = 3

@dataclass
class Data:
    I: np.ndarray
    J: np.ndarray
    S: np.ndarray
    demands: np.ndarray
    scenarios: np.ndarray
    capacities: np.ndarray
    fixed_costs: np.ndarray
    shipment_costs: np.ndarray
    max_demand_sum_over_scenario: float
    # Metadata
    distribution: str
    target_cv: float
    empirical_cv_mean: float
    empirical_cv_std: float
    empirical_cv_min: float
    empirical_cv_max: float
    num_negatives_clipped: int
    total_draws: int
    scaling_factor_applied: float
    # Additional overall dispersion metric
    r_overall: float
    # Merve_Normal specific parameters
    sigma_lower: float = 0.0
    sigma_upper: float = 0.0
    # ARMA covariate generation specific parameters
    covariate_vectors: Optional[np.ndarray] = None
    num_covariates: int = 0
    arma_r2_scores: Optional[np.ndarray] = None
    arma_validation_r2: Optional[np.ndarray] = None
    arma_validation_mse: Optional[np.ndarray] = None
    arma_validation_mape: Optional[np.ndarray] = None


def word_reader(file_path):
    with open(file_path, "r") as file:
        for line in file:
            for word in line.split():
                yield word


def _draw_with_target_cv(
    means: np.ndarray,
    target_cv: float,
    method: str,
    rng: np.random.Generator,
    sigma_lower: float = 0.1,
    sigma_upper: float = 0.3,
) -> Tuple[np.ndarray, int]:
    """Draw one scenario vector with per-customer mean and target CV.

    Returns the scenario and the count of negative raw draws (before clipping).
    For Merve_Normal method, sigma_lower and sigma_upper parameters control 
    the per-customer standard deviation range.
    """
    method = method.lower()

    if method == "normal":
        # X_i ~ N(mu_i, (target_cv * mu_i)^2)
        sigma = target_cv * means
        draws = rng.normal(loc=means, scale=sigma)
    elif method == "merve_normal":
        # Merve_Normal method: per-customer sigma_i ~ Uniform(sigma_lower * d_i, sigma_upper * d_i)
        # Then d_{i,m} ~ Normal(mean = d_i, std = sigma_i)
        sigma_i = rng.uniform(
            low=sigma_lower * means, 
            high=sigma_upper * means
        )
        draws = rng.normal(loc=means, scale=sigma_i)
    elif method == "uniform":
        # X_i ~ U[mu_i - a_i, mu_i + a_i], a_i = target_cv * mu_i * sqrt(3)
        a = target_cv * means * np.sqrt(3.0)
        low = means - a
        high = means + a
        draws = rng.uniform(low=low, high=high)
    elif method == "gamma":
        # mean = k*theta = mu, CV = 1/sqrt(k) = target_cv -> k = 1/target_cv^2, theta = mu / k
        k = np.where(target_cv > 0, 1.0 / (target_cv ** 2), np.inf)
        theta = np.divide(means, k, out=np.zeros_like(means), where=np.isfinite(k))
        draws = rng.gamma(shape=k, scale=theta)
    elif method == "lognormal":
        # For lognormal with parameters (m, s^2): E[X] = exp(m + s^2/2), CV^2 = exp(s^2)-1
        # => s^2 = ln(1+CV^2), m = ln(mu) - s^2/2
        s2 = np.log1p(target_cv ** 2)
        s = np.sqrt(s2)
        m = np.log(means) - 0.5 * s2
        draws = rng.lognormal(mean=m, sigma=s)
    else:
        raise ValueError("Unsupported method. Use 'normal', 'uniform', 'gamma', 'lognormal', 'merve_normal', 'arma(2,2)_with_covariate_generation', or 'merve_with_covariates'.")

    neg_count = int(np.sum(draws < 0))
    draws = np.maximum(draws, 0.0)  # clip negatives (normal/uniform)
    return draws, neg_count


def generate_scenarios_cv(
    demands: np.ndarray,
    capacities: np.ndarray,
    num_scenarios: int = 10,
    target_cv: float = 0.2,
    method: str = "normal",
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Generate demand scenarios parameterized by target CV and distribution.

    Returns:
        scenarios: (S, |I|) array
        info: dict with diagnostics (negatives clipped, scaling factor, etc.)
    """
    rng = np.random.default_rng(seed)
    I = len(demands)

    scenarios = np.zeros((num_scenarios, I), dtype=float)
    total_negatives = 0

    for s in range(num_scenarios):
        scenario, neg = _draw_with_target_cv(demands, target_cv, method, rng)
        scenarios[s, :] = scenario
        total_negatives += neg

    total_capacity = float(np.sum(capacities))
    sums = scenarios.sum(axis=1)
    max_scenario_demand = float(np.max(sums))

    scaling_factor = 1.0
    if max_scenario_demand > total_capacity:
        # Scale all scenarios down with a 5% safety factor; CV and R_overall are invariant to positive scaling
        safety = 0.95
        scaling_factor = (total_capacity / max_scenario_demand) * safety
        scenarios *= scaling_factor

    info = {
        "total_negatives": total_negatives,
        "total_draws": int(num_scenarios * I),
        "max_scenario_demand": max_scenario_demand,
        "total_capacity": total_capacity,
        "scaling_factor": scaling_factor,
    }
    return scenarios, info


def generate_scenarios_merve_normal(
    demands: np.ndarray,
    capacities: np.ndarray,
    num_scenarios: int = 10,
    sigma_lower: float = 0.1,
    sigma_upper: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Generate demand scenarios using Merve_Normal method.
    
    This method:
    1. Generates per-customer standard deviations sigma_i ~ Uniform(sigma_lower * d_i, sigma_upper * d_i)
    2. For each scenario, generates d_{i,m} ~ Normal(mean=d_i, std=sigma_i)
    3. Clips negative demands to 0
    4. Enforces capacity constraint by iterative scaling (0.9 factor) when total demand > total capacity
    
    Returns:
        scenarios: (S, |I|) array
        info: dict with diagnostics (negatives clipped, scaling iterations, etc.)
    """
    rng = np.random.default_rng(seed)
    I = len(demands)
    total_capacity = float(np.sum(capacities))

    # Generate per-customer standard deviations once
    sigma_i = rng.uniform(
        low=sigma_lower * demands, 
        high=sigma_upper * demands
    )

    scenarios = np.zeros((num_scenarios, I), dtype=float)
    total_negatives = 0
    total_scaling_iterations = 0

    for s in range(num_scenarios):
        # Generate scenario with per-customer sigma_i
        scenario = rng.normal(loc=demands, scale=sigma_i)
        
        # Clip negatives
        neg_count = int(np.sum(scenario < 0))
        total_negatives += neg_count
        scenario = np.maximum(scenario, 0.0)
        
        # Enforce capacity constraint with iterative scaling
        scaling_iterations = 0
        D_m = np.sum(scenario)
        while D_m > total_capacity:
            scenario *= 0.9  # Scale down by 10%
            D_m = np.sum(scenario)
            scaling_iterations += 1
            # Safety check to avoid infinite loops
            if scaling_iterations > 100:
                break
        
        total_scaling_iterations += scaling_iterations
        scenarios[s, :] = scenario

    max_scenario_demand = float(np.max(np.sum(scenarios, axis=1)))

    info = {
        "total_negatives": total_negatives,
        "total_draws": int(num_scenarios * I),
        "max_scenario_demand": max_scenario_demand,
        "total_capacity": total_capacity,
        "scaling_factor": 1.0,  # Not applicable for Merve_Normal (per-scenario scaling)
        "total_scaling_iterations": total_scaling_iterations,
        "sigma_lower": sigma_lower,
        "sigma_upper": sigma_upper,
    }
    return scenarios, info


def generate_scenarios_merve_with_covariates(
    demands: np.ndarray,
    capacities: np.ndarray,
    num_scenarios: int = 10,
    sigma_lower: float = 0.1,
    sigma_upper: float = 0.3,
    num_covariates: int = _num_covariates_merve,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Generate demand scenarios using Merve_Normal method and create PCA-based covariates.
    
    1) Generate scenarios with Merve_Normal (identical logic to generate_scenarios_merve_normal).
    2) Build scenario-level covariates Z (S x m) from the scenario–customer matrix via PCA.
    3) Compute per-customer predictiveness metrics (R², MSE, MAPE) using the same
       helpers and validation split as the ARMA method.
    4) Return (scenarios, covariates, info).
    
    Returns:
        scenarios: (S, |I|) array of demand scenarios
        covariates: (S, m) array of PCA-based covariate vectors
        info: dict with diagnostics including predictiveness metrics
    """
    # Step A: Generate demands using Merve_Normal method (unchanged logic)
    scenarios, merve_info = generate_scenarios_merve_normal(
        demands, capacities, num_scenarios, sigma_lower, sigma_upper, seed
    )
    
    S, I = scenarios.shape
    
    # Step B: PCA covariates construction
    # Center across scenarios (per customer): Y - 1*y_bar^T
    Y = scenarios  # Shape: (S, I)
    y_bar = np.mean(Y, axis=0)  # Mean per customer, shape: (I,)
    Y_tilde = Y - y_bar[np.newaxis, :]  # Centered matrix, shape: (S, I)
    
    # Compute SVD: Y_tilde = U * Sigma * V^T
    try:
        U, sigma, Vt = np.linalg.svd(Y_tilde, full_matrices=False)
        
        # Determine effective number of components
        rank_Y_tilde = np.linalg.matrix_rank(Y_tilde)
        m_eff = min(num_covariates, rank_Y_tilde, S)
        
        if m_eff < num_covariates:
            print(f"Warning: Reduced covariates from {num_covariates} to {m_eff} due to rank constraints")
        
        # Take top m_eff components: Z = U_m (orthonormal features)
        Z = U[:, :m_eff]  # Shape: (S, m_eff)
        
    except np.linalg.LinAlgError:
        # Fallback in case of SVD failure
        m_eff = min(num_covariates, S - 1)
        Z = np.random.randn(S, m_eff)
        print(f"Warning: SVD failed, using random covariates with {m_eff} dimensions")
    
    # Step C: Evaluation - compute metrics using proper train/test split
    r2_scores = np.zeros(I)  # R² on training data only
    validation_mse_scores = np.zeros(I)  # MSE on test data
    validation_mape_scores = np.zeros(I)  # MAPE on test data
    
    for i in range(I):
        y_i = scenarios[:, i]  # Demands for customer i across scenarios
        
        # Compute all metrics with consistent train/test split
        train_r2, test_mse, test_mape = _compute_training_and_test_metrics(
            Z, y_i, test_size=0.2, random_state=42 + i
        )
        r2_scores[i] = train_r2  # R² on training data
        validation_mse_scores[i] = test_mse  # MSE on test data
        validation_mape_scores[i] = test_mape  # MAPE on test data
    
    # Step D: Aggregate to info dict (updated for proper train/test split)
    info = merve_info.copy()  # Start with Merve_Normal info
    
    # Add PCA and evaluation metrics
    info.update({
        "num_covariates": m_eff,
        "r2_mean": float(np.mean(r2_scores)),  # R² on training data
        "r2_std": float(np.std(r2_scores)),
        "r2_min": float(np.min(r2_scores)),
        "r2_max": float(np.max(r2_scores)),
        "test_mse_mean": float(np.mean(validation_mse_scores)),  # MSE on test data
        "test_mse_std": float(np.std(validation_mse_scores)),
        "test_mse_min": float(np.min(validation_mse_scores)),
        "test_mse_max": float(np.max(validation_mse_scores)),
        "test_mape_mean": float(np.mean(validation_mape_scores)),  # MAPE on test data
        "test_mape_std": float(np.std(validation_mape_scores)),
        "test_mape_min": float(np.min(validation_mape_scores)),
        "test_mape_max": float(np.max(validation_mape_scores)),
        "validation_mse_std": float(np.std(validation_mse_scores)),
        "validation_mse_min": float(np.min(validation_mse_scores)),
        "validation_mse_max": float(np.max(validation_mse_scores)),
        "validation_mape_mean": float(np.mean(validation_mape_scores)),
        "validation_mape_std": float(np.std(validation_mape_scores)),
        "validation_mape_min": float(np.min(validation_mape_scores)),
        "validation_mape_max": float(np.max(validation_mape_scores)),
        # Store per-customer arrays for Data dataclass compatibility
        "r2_scores": r2_scores,  # R² scores on training data
        "validation_mse_scores": validation_mse_scores,  # MSE scores on test data
        "validation_mape_scores": validation_mape_scores,  # MAPE scores on test data
    })
    
    return scenarios, Z, info


def _create_stable_arma_matrices(d: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create stable ARMA(2,2) matrices for given dimension d.
    
    Returns:
        Phi1, Phi2: AR matrices (d x d)
        Theta1, Theta2: MA matrices (d x d)
        Sigma_U: Innovation covariance matrix (d x d)
    """
    # Create AR matrices with diagonal dominance for stability
    Phi1 = np.zeros((d, d))
    Phi2 = np.zeros((d, d))
    
    # Diagonal elements for stability (ensuring sum < 1)
    phi1_diag = rng.uniform(0.3, 0.5, d)
    phi2_diag = rng.uniform(0.1, 0.3, d)
    
    # Ensure stability condition: diagonal sums < 1
    total_diag = phi1_diag + phi2_diag
    scale_factor = np.where(total_diag >= 0.9, 0.8 / total_diag, 1.0)
    phi1_diag *= scale_factor
    phi2_diag *= scale_factor
    
    np.fill_diagonal(Phi1, phi1_diag)
    np.fill_diagonal(Phi2, phi2_diag)
    
    # Add small off-diagonal terms for weak interaction
    for i in range(d):
        for j in range(d):
            if i != j:
                if rng.random() < 0.3:  # 30% chance of non-zero off-diagonal
                    Phi1[i, j] = rng.uniform(-0.1, 0.1)
                if rng.random() < 0.2:  # 20% chance for Phi2
                    Phi2[i, j] = rng.uniform(-0.05, 0.05)
    
    # Create MA matrices (smaller values than AR)
    Theta1 = rng.uniform(-0.2, 0.2, (d, d))
    Theta2 = rng.uniform(-0.1, 0.1, (d, d))
    
    # Create positive definite innovation covariance matrix
    # Start with diagonal variances
    diag_vars = rng.uniform(0.5, 1.5, d)
    Sigma_U = np.diag(diag_vars)
    
    # Add off-diagonal correlations
    for i in range(d):
        for j in range(i+1, d):
            if rng.random() < 0.4:  # 40% chance of correlation
                corr = rng.uniform(-0.3, 0.3)
                cov = corr * np.sqrt(diag_vars[i] * diag_vars[j])
                Sigma_U[i, j] = cov
                Sigma_U[j, i] = cov
    
    # Check and ensure positive definiteness
    eigenvals = eigvals(Sigma_U)
    if np.any(eigenvals <= 0):
        # Fallback to diagonal matrix if not positive definite
        Sigma_U = np.diag(diag_vars)
    
    return Phi1, Phi2, Theta1, Theta2, Sigma_U


def _generate_arma_features(num_scenarios: int, num_covariates: int, rng: np.random.Generator) -> np.ndarray:
    """Generate feature vectors using ARMA(2,2) process.
    
    Returns:
        features: (num_scenarios, num_covariates) array
    """
    d = num_covariates
    
    # Create ARMA matrices
    Phi1, Phi2, Theta1, Theta2, Sigma_U = _create_stable_arma_matrices(d, rng)
    
    # Initialize arrays for ARMA process
    # We need extra time steps for initialization
    total_steps = num_scenarios + 10  # Extra steps for burn-in
    X = np.zeros((total_steps, d))
    U = np.zeros((total_steps, d))
    
    # Generate innovation sequence
    for t in range(total_steps):
        U[t] = rng.multivariate_normal(np.zeros(d), Sigma_U)
    
    # Initialize first two time steps
    X[0] = rng.multivariate_normal(np.zeros(d), np.eye(d))
    X[1] = rng.multivariate_normal(np.zeros(d), np.eye(d))
    
    # Generate ARMA(2,2) process
    for t in range(2, total_steps):
        X[t] = (Phi1 @ X[t-1] + Phi2 @ X[t-2] + 
                U[t] + Theta1 @ U[t-1] + Theta2 @ U[t-2])
    
    # Return last num_scenarios steps (after burn-in)
    return X[-num_scenarios:]


def _compute_training_and_test_metrics(features: np.ndarray, demands: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[float, float, float]:
    """
    Train regression model and compute R² on training data, MSE and MAPE on test data.
    
    Args:
        features: (n_scenarios, n_features) array
        demands: (n_scenarios,) array - demands for one customer
        test_size: Proportion of data for testing
        random_state: Random seed for train/test split
    
    Returns:
        train_r2: R² score on training set (should be between 0 and 1)
        test_mse: Mean squared error on test set
        test_mape: Mean absolute percentage error on test set (%)
    """
    try:
        if len(features) < 5:  # Need minimum data for train/test split
            return 0.0, float('inf'), float('inf')
            
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, demands, test_size=test_size, random_state=random_state
        )
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Compute R² on training data (should be positive)
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Compute MSE and MAPE on test data
        y_test_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Compute MAPE with robust handling for zero values
        absolute_errors = np.abs(y_test - y_test_pred)
        # For MAPE, avoid division by very small values by using max(|actual|, small_threshold)
        denominator = np.maximum(np.abs(y_test), 1.0)  # Use 1.0 as minimum denominator
        test_mape = np.mean(absolute_errors / denominator) * 100  # Convert to percentage
        
        # Ensure R² is non-negative (it should be by definition on training data)
        train_r2 = max(0.0, train_r2)
        
        return float(train_r2), float(test_mse), float(test_mape)
        
    except Exception:
        return 0.0, float('inf'), float('inf')


def _compute_r2_score(y_true: np.ndarray, X: np.ndarray) -> float:
    """Compute R^2 score for linear regression of y on X."""
    try:
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Solve normal equations: beta = (X'X)^(-1)X'y
        XTX = X_with_intercept.T @ X_with_intercept
        XTy = X_with_intercept.T @ y_true
        
        # Check for singularity
        if np.linalg.cond(XTX) > 1e12:
            return 0.0
            
        beta = np.linalg.solve(XTX, XTy)
        y_pred = X_with_intercept @ beta
        
        # Compute R^2
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
            
        r2 = 1 - (ss_res / ss_tot)
        return float(max(0.0, r2))  # Ensure non-negative
        
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def _compute_regression_metrics(features: np.ndarray, demands: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[float, float, float]:
    """
    Train regression model on 80% of data and compute validation metrics on 20%.
    
    Args:
        features: (n_scenarios, n_features) array
        demands: (n_scenarios,) array - demands for one customer
        test_size: Proportion of data for validation
        random_state: Random seed for train/test split
    
    Returns:
        validation_r2: R² score on validation set
        validation_mse: Mean squared error on validation set
        validation_mape: Mean absolute percentage error on validation set (%)
    """
    try:
        if len(features) < 5:  # Need minimum data for train/test split
            return 0.0, float('inf'), float('inf')
            
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            features, demands, test_size=test_size, random_state=random_state
        )
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions on validation set
        y_pred = model.predict(X_val)
        
        # Compute metrics
        val_r2 = r2_score(y_val, y_pred)
        val_mse = mean_squared_error(y_val, y_pred)
        
        # Compute MAPE with robust handling for zero values
        # Use custom calculation to avoid issues with very small values
        absolute_errors = np.abs(y_val - y_pred)
        # For MAPE, avoid division by very small values by using max(|actual|, small_threshold)
        denominator = np.maximum(np.abs(y_val), 1.0)  # Use 1.0 as minimum denominator
        val_mape = np.mean(absolute_errors / denominator) * 100  # Convert to percentage
        
        return float(val_r2), float(val_mse), float(val_mape)
        
    except Exception:
        return 0.0, float('inf'), float('inf')


def generate_scenarios_arma_covariate(
    demands: np.ndarray,
    capacities: np.ndarray,
    num_scenarios: int = 10,
    num_covariates: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Generate demand scenarios using ARMA(2,2) covariate generation method.
    
    This method:
    1. Generates feature vectors using ARMA(2,2) process
    2. For each customer, generates demands using feature-driven linear + noise model
    3. Rescales demands to match deterministic means
    4. Enforces capacity constraints
    
    Returns:
        scenarios: (S, |I|) array of demand scenarios
        features: (S, d) array of feature vectors
        info: dict with diagnostics
    """
    rng = np.random.default_rng(seed)
    I = len(demands)
    total_capacity = float(np.sum(capacities))
    
    # Generate ARMA(2,2) feature vectors
    features = _generate_arma_features(num_scenarios, num_covariates, rng)
    
    # Initialize demand matrix
    scenarios = np.zeros((num_scenarios, I))
    total_negatives = 0
    r2_scores = np.zeros(I)
    validation_r2_scores = np.zeros(I)
    validation_mse_scores = np.zeros(I)
    validation_mape_scores = np.zeros(I)
    
    # Generate demands for each customer
    for i in range(I):
        # Generate random weight vectors for customer i
        A_i = rng.normal(0, 1, num_covariates)
        B_i = rng.normal(0, 0.5, num_covariates)  # Smaller variance weights
        delta_i = rng.normal(0, 0.1, num_covariates)  # Small offset
        
        # Generate raw demands for all scenarios
        raw_demands = np.zeros(num_scenarios)
        for s in range(num_scenarios):
            # Gaussian noise term
            epsilon = rng.normal(0, 1)
            
            # Feature-driven demand generation
            X_s = features[s]
            linear_term = A_i @ (X_s + delta_i / 4.0)
            variance_term = (B_i @ X_s) * epsilon
            
            raw_demand = linear_term + variance_term
            raw_demands[s] = max(0.0, raw_demand)  # Clip negatives
            
            if raw_demand < 0:
                total_negatives += 1
        
        # Rescale to match deterministic mean
        empirical_mean = np.mean(raw_demands)
        if empirical_mean > 0:
            scaling_factor = demands[i] / empirical_mean
            raw_demands *= scaling_factor
        
        scenarios[:, i] = raw_demands
        
        # Compute validation metrics using train/test split (80% train, 20% validation)
        val_r2, val_mse, val_mape = _compute_regression_metrics(features, raw_demands, test_size=0.2, random_state=42+i)
        validation_r2_scores[i] = val_r2
        validation_mse_scores[i] = val_mse
        validation_mape_scores[i] = val_mape
        
        # For full dataset R², use all data (this gives us the theoretical maximum performance)
        r2_scores[i] = _compute_r2_score(raw_demands, features)
    
    # Enforce capacity constraints
    max_scenario_demand = float(np.max(np.sum(scenarios, axis=1)))
    scaling_factor = 1.0
    
    if max_scenario_demand > total_capacity:
        # Scale all scenarios down with safety factor
        safety = 0.95
        scaling_factor = (total_capacity / max_scenario_demand) * safety
        scenarios *= scaling_factor
        max_scenario_demand *= scaling_factor
    
    info = {
        "total_negatives": total_negatives,
        "total_draws": int(num_scenarios * I),
        "max_scenario_demand": max_scenario_demand,
        "total_capacity": total_capacity,
        "scaling_factor": scaling_factor,
        "num_covariates": num_covariates,
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "r2_min": float(np.min(r2_scores)),  
        "r2_max": float(np.max(r2_scores)),
        "validation_r2_mean": float(np.mean(validation_r2_scores)),
        "validation_r2_std": float(np.std(validation_r2_scores)),
        "validation_r2_min": float(np.min(validation_r2_scores)),
        "validation_r2_max": float(np.max(validation_r2_scores)),
        "validation_mse_mean": float(np.mean(validation_mse_scores)),
        "validation_mse_std": float(np.std(validation_mse_scores)),
        "validation_mse_min": float(np.min(validation_mse_scores)),
        "validation_mse_max": float(np.max(validation_mse_scores)),
        "validation_mape_mean": float(np.mean(validation_mape_scores)),
        "validation_mape_std": float(np.std(validation_mape_scores)),
        "validation_mape_min": float(np.min(validation_mape_scores)),
        "validation_mape_max": float(np.max(validation_mape_scores)),
        "validation_r2_scores": validation_r2_scores,
        "validation_mse_scores": validation_mse_scores,
        "validation_mape_scores": validation_mape_scores,
    }
    
    return scenarios, features, info


def compute_empirical_cv_per_customer(scenarios: np.ndarray) -> np.ndarray:
    """Compute empirical CV per customer across scenarios: std/mean.
    If mean is zero, returns 0 for that customer (to avoid division by zero).
    """
    means = scenarios.mean(axis=0)
    stds = scenarios.std(axis=0, ddof=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        cvs = np.divide(stds, means, out=np.zeros_like(stds), where=means > 0)
    return cvs


def compute_r_overall(scenarios: np.ndarray) -> float:
    """Compute R_overall = average over customers of (max - min)/mean across scenarios.
    Customers with zero mean contribute 0 (or are skipped if all are zero).
    """
    mins = scenarios.min(axis=0)
    maxs = scenarios.max(axis=0)
    means = scenarios.mean(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_i = np.divide(maxs - mins, means, out=np.zeros_like(means), where=means > 0)
    valid = means > 0
    if np.any(valid):
        return float(np.mean(r_i[valid]))
    return 0.0


def create_merve_covariates_excel_tracking(
    tracking_data: List[Dict],
    output_dir: str
) -> str:
    """
    Create Excel file to track MERVE_WITH_COVARIATES method generation results.
    
    Args:
        tracking_data: List of dictionaries containing tracking information for each file
        output_dir: Directory where to save the Excel file
    
    Returns:
        Path to the created Excel file
    """
    # Create DataFrame from tracking data
    df = pd.DataFrame(tracking_data)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"MERVE_WITH_COVARIATES_tracking_{timestamp}.xlsx"
    excel_path = os.path.join(output_dir, excel_filename)
    
    # Create Excel writer with multiple sheets
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Main tracking sheet
        df.to_excel(writer, sheet_name='Tracking_Data', index=False)
        
        # Summary statistics sheet
        if not df.empty:
            summary_data = {
                'Metric': ['Total Files Processed', 'Average R²', 'Average MSE', 'Average MAPE', 
                          'Min R²', 'Max R²', 'Min MSE', 'Max MSE', 'Min MAPE', 'Max MAPE',
                          'Average Scenarios', 'Average Covariates'],
                'Value': [
                    len(df),
                    df['R²'].mean() if 'R²' in df.columns else 0,
                    df['MSE'].mean() if 'MSE' in df.columns else 0,
                    df['MAPE'].mean() if 'MAPE' in df.columns else 0,
                    df['R²'].min() if 'R²' in df.columns else 0,
                    df['R²'].max() if 'R²' in df.columns else 0,
                    df['MSE'].min() if 'MSE' in df.columns else 0,
                    df['MSE'].max() if 'MSE' in df.columns else 0,
                    df['MAPE'].min() if 'MAPE' in df.columns else 0,
                    df['MAPE'].max() if 'MAPE' in df.columns else 0,
                    df['Num_Scenarios'].mean() if 'Num_Scenarios' in df.columns else 0,
                    df['Num_Covariates'].mean() if 'Num_Covariates' in df.columns else 0,
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    return excel_path


def read_dataset(
    file_path: str,
    num_scenarios: int = 10,
    target_cv: float = 0.2,
    method: str = "normal",
    seed: int = 42,
    sigma_lower: Optional[float] = None,
    sigma_upper: Optional[float] = None,
    num_covariates: Optional[int] = None,
) -> Data:
    # Holmberg/OR-Library layout: (capacity, fixed cost) PAIRS per facility, and
    # c_ij is the cost of a customer's whole demand -> unit cost c_ij / d_j.
    # (Fixed 2026-11-01; earlier code mixed the pairs and used c_ij as a unit cost.)
    try:
        from holmberg_format import read_holmberg_instance
    except ImportError:
        from utils.holmberg_format import read_holmberg_instance
    capacities, fixed_costs, demands, shipment_costs = read_holmberg_instance(file_path)
    num_facilities, num_customers = len(capacities), len(demands)

    I = np.arange(num_customers)
    J = np.arange(num_facilities)
    S = np.arange(num_scenarios)

    # Choose appropriate scenario generation method
    covariate_vectors = None
    arma_r2_scores = None
    arma_validation_r2 = None
    arma_validation_mse = None
    arma_validation_mape = None
    
    if method.lower() == "merve_normal":
        # Ensure sigma values are provided for Merve_Normal method
        if sigma_lower is None or sigma_upper is None:
            raise ValueError("sigma_lower and sigma_upper must be provided for Merve_Normal method")
        scenarios, gen_info = generate_scenarios_merve_normal(
            demands, capacities, num_scenarios, sigma_lower, sigma_upper, seed
        )
    elif method.lower() == "arma(2,2)_with_covariate_generation":
        # Ensure num_covariates is provided for ARMA method
        if num_covariates is None:
            num_covariates = 4  # Default value
        scenarios, covariate_vectors, gen_info = generate_scenarios_arma_covariate(
            demands, capacities, num_scenarios, num_covariates, seed
        )
        # Extract R^2 scores and validation metrics for each customer
        arma_r2_scores = gen_info.get("validation_r2_scores", np.zeros(len(demands)))
        arma_validation_r2 = gen_info.get("validation_r2_scores", np.zeros(len(demands)))
        arma_validation_mse = gen_info.get("validation_mse_scores", np.zeros(len(demands)))
        arma_validation_mape = gen_info.get("validation_mape_scores", np.zeros(len(demands)))
    elif method.lower() == "merve_with_covariates":
        if num_covariates is None:
            num_covariates = _num_covariates_merve
        scenarios, covariate_vectors, gen_info = generate_scenarios_merve_with_covariates(
            demands, capacities, num_scenarios, sigma_lower, sigma_upper, num_covariates, seed
        )
        arma_r2_scores = gen_info.get("r2_scores", np.zeros(len(demands)))  # Training R² scores
        arma_validation_r2 = np.zeros(len(demands))  # Not needed anymore (confusing name)
        arma_validation_mse = gen_info.get("validation_mse_scores", np.zeros(len(demands)))  # Test MSE
        arma_validation_mape = gen_info.get("validation_mape_scores", np.zeros(len(demands)))  # Test MAPE
    else:
        scenarios, gen_info = generate_scenarios_cv(
            demands, capacities, num_scenarios, target_cv, method, seed
        )

    max_demand_sum = float(np.max(np.sum(scenarios, axis=1)))

    # Empirical CV diagnostics
    cvs = compute_empirical_cv_per_customer(scenarios)
    empirical_cv_mean = float(np.mean(cvs))
    empirical_cv_std = float(np.std(cvs))
    empirical_cv_min = float(np.min(cvs))
    empirical_cv_max = float(np.max(cvs))

    r_overall = compute_r_overall(scenarios)

    return Data(
        I=I,
        J=J,
        S=S,
        demands=demands,
        scenarios=scenarios,
        capacities=capacities,
        fixed_costs=fixed_costs,
        shipment_costs=shipment_costs,
        max_demand_sum_over_scenario=max_demand_sum,
        distribution=method,
        target_cv=target_cv,
        empirical_cv_mean=empirical_cv_mean,
        empirical_cv_std=empirical_cv_std,
        empirical_cv_min=empirical_cv_min,
        empirical_cv_max=empirical_cv_max,
        num_negatives_clipped=int(gen_info["total_negatives"]),
        total_draws=int(gen_info["total_draws"]),
        scaling_factor_applied=float(gen_info["scaling_factor"]),
        r_overall=r_overall,
        sigma_lower=sigma_lower if sigma_lower is not None else 0.0,
        sigma_upper=sigma_upper if sigma_upper is not None else 0.0,
        covariate_vectors=covariate_vectors,
        num_covariates=num_covariates if num_covariates is not None else 0,
        arma_r2_scores=arma_r2_scores,
        arma_validation_r2=arma_validation_r2,
        arma_validation_mse=arma_validation_mse,
        arma_validation_mape=arma_validation_mape,
    )


def process_files(
    input_dir: str,
    file_list: List[str],
    num_scenarios: int = 10,
    target_cv: float = 0.2,
    method: str = "normal",
    output_format: str = "txt",
    seed: int = 42,
    sigma_lower: Optional[float] = None,
    sigma_upper: Optional[float] = None,
    num_covariates: Optional[int] = None,
):
    """
    Process specific files from `file_list`, generate stochastic scenarios with a
    target CV, Merve_Normal parameters, or ARMA covariate generation, and save them to a named subfolder inside `input_dir`.

    Args:
        input_dir (str): Directory where the original .txt files are located.
        file_list (list): List of file names (e.g., ["p1", "p2"]) without extensions.
        num_scenarios (int): Number of demand scenarios to generate.
        target_cv (float): Target coefficient of variation (CV = std/mean) for CV-based methods.
        method (str): Distribution ("normal", "uniform", "gamma", "lognormal", "merve_normal", "arma(2,2)_with_covariate_generation", or "merve_with_covariates").
        output_format (str): "txt" or "pkl".
        seed (int): Random seed for reproducible results.
        sigma_lower (float): Lower bound for sigma_i in Merve_Normal method.
        sigma_upper (float): Upper bound for sigma_i in Merve_Normal method.
        num_covariates (int): Number of covariates for ARMA(2,2) method.
    """
    if method.lower() == "merve_normal":
        folder_name = f"scenarios_s{num_scenarios}_sigma{sigma_lower}_{sigma_upper}_merve_normal"
    elif method.lower() == "arma(2,2)_with_covariate_generation":
        folder_name = f"scenarios_s{num_scenarios}_cov{num_covariates}_arma_with_covariates"
    elif method.lower() == "merve_with_covariates":
        folder_name = f"scenarios_s{num_scenarios}_sigma{sigma_lower}_{sigma_upper}_cov{num_covariates}_merve_with_covariates"
    elif method.lower() in ["normal", "uniform", "gamma", "lognormal"]:
        folder_name = f"scenarios_s{num_scenarios}_cv{target_cv}_{method.lower()}"
    else:
        folder_name = f"scenarios_s{num_scenarios}_cv{target_cv}_dist{method}"
    output_dir = os.path.join(input_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Excel tracking for MERVE_WITH_COVARIATES method
    tracking_data = []
    
    for fname in file_list:
        full_path = os.path.join(input_dir, fname)
        if not os.path.exists(full_path):
            print(f"⚠️ File not found: {fname}")
            continue

        print(f"🔄 Processing: {fname}")
        try:
            data = read_dataset(full_path, num_scenarios, target_cv, method, seed, sigma_lower, sigma_upper, num_covariates)

            total_capacity = float(np.sum(data.capacities))
            print(f"   📊 Total capacity: {total_capacity}")
            print(f"   📊 Max scenario demand: {data.max_demand_sum_over_scenario:.2f}")
            print(f"   ✅ Feasibility ratio: {data.max_demand_sum_over_scenario/total_capacity:.4f}")
            
            if method.lower() == "merve_normal":
                print(f"   🧮 Sigma range: [{data.sigma_lower}, {data.sigma_upper}] | Dist: {data.distribution}")
            elif method.lower() == "arma(2,2)_with_covariate_generation":
                print(f"   🧮 Num covariates: {data.num_covariates} | Dist: {data.distribution}")
                if data.arma_r2_scores is not None and len(data.arma_r2_scores) > 0:
                    avg_r2 = float(np.mean(data.arma_r2_scores))
                    print(f"   📈 Average R² (full data): {avg_r2:.4f}")
                if data.arma_validation_r2 is not None and len(data.arma_validation_r2) > 0:
                    avg_val_r2 = float(np.mean(data.arma_validation_r2))
                    print(f"   🎯 Average validation R²: {avg_val_r2:.4f}")
                if data.arma_validation_mse is not None and len(data.arma_validation_mse) > 0:
                    avg_val_mse = float(np.mean(data.arma_validation_mse))
                    print(f"   📉 Average validation MSE: {avg_val_mse:.2f}")
                if data.arma_validation_mape is not None and len(data.arma_validation_mape) > 0:
                    avg_val_mape = float(np.mean(data.arma_validation_mape))
                    print(f"   📊 Average validation error: {avg_val_mape:.1f}%")
            elif method.lower() == "merve_with_covariates":
                print(f"   🧮 Num covariates: {data.num_covariates} | Dist: {data.distribution}")
                if data.arma_r2_scores is not None and len(data.arma_r2_scores) > 0:
                    avg_r2 = float(np.mean(data.arma_r2_scores))
                    print(f"   📈 Average R² (training data): {avg_r2:.4f}")
                # Note: no validation R² anymore (was confusing)
                if data.arma_validation_mse is not None and len(data.arma_validation_mse) > 0:
                    avg_val_mse = float(np.mean(data.arma_validation_mse))
                    print(f"   📉 Average test MSE: {avg_val_mse:.2f}")
                if data.arma_validation_mape is not None and len(data.arma_validation_mape) > 0:
                    avg_val_mape = float(np.mean(data.arma_validation_mape))
                    print(f"   📊 Average test error: {avg_val_mape:.1f}%")
            else:
                print(f"   🧮 Target CV: {data.target_cv} | Dist: {data.distribution}")
            
            print(
                f"   📐 Empirical CV across customers: mean={data.empirical_cv_mean:.4f}, "
                f"std={data.empirical_cv_std:.4f}, min={data.empirical_cv_min:.4f}, max={data.empirical_cv_max:.4f}"
            )
            print(f"   🌐 R_overall (avg normalized range): {data.r_overall:.4f}")
            print(
                f"   ⚠️ Negatives clipped: {data.num_negatives_clipped} / {data.total_draws}"
            )
            
            if method.lower() not in ["merve_normal"] and data.scaling_factor_applied != 1.0:
                print(f"   🔧 Scaling factor applied (capacity): {data.scaling_factor_applied:.6f}")
            elif method.lower() == "merve_normal":
                print(f"   🔧 Merve_Normal: Per-scenario capacity scaling applied")

            # Collect tracking data for MERVE_WITH_COVARIATES method
            if method.lower() == "merve_with_covariates":
                # Calculate metrics
                avg_r2 = float(np.mean(data.arma_r2_scores)) if data.arma_r2_scores is not None else 0.0
                avg_mse = float(np.mean(data.arma_validation_mse)) if data.arma_validation_mse is not None else 0.0
                avg_mape = float(np.mean(data.arma_validation_mape)) if data.arma_validation_mape is not None else 0.0
                
                # Create dictionary with other parameters for the "other_parameters" column
                other_params = {
                    'empirical_cv_mean': data.empirical_cv_mean,
                    'empirical_cv_std': data.empirical_cv_std,
                    'empirical_cv_min': data.empirical_cv_min,
                    'empirical_cv_max': data.empirical_cv_max,
                    'r_overall': data.r_overall,
                    'num_negatives_clipped': data.num_negatives_clipped,
                    'total_draws': data.total_draws,
                    'scaling_factor_applied': data.scaling_factor_applied,
                    'max_demand_sum_over_scenario': data.max_demand_sum_over_scenario,
                    'total_capacity': total_capacity,
                    'validation_r2_mean': float(np.mean(data.arma_validation_r2)) if data.arma_validation_r2 is not None else 0.0,
                    'validation_r2_std': float(np.std(data.arma_validation_r2)) if data.arma_validation_r2 is not None else 0.0,
                    'validation_mse_mean': avg_mse,
                    'validation_mse_std': float(np.std(data.arma_validation_mse)) if data.arma_validation_mse is not None else 0.0,
                    'validation_mape_mean': avg_mape,
                    'validation_mape_std': float(np.std(data.arma_validation_mape)) if data.arma_validation_mape is not None else 0.0,
                }
                
                # Add tracking row
                tracking_row = {
                    'Original_File_Name': fname,
                    'Num_First_Stage_Solutions': len(data.J),  # Number of facilities
                    'Num_Second_Stage_Solutions': len(data.I),  # Number of customers  
                    'Num_Scenarios': num_scenarios,
                    'Folder_Name': folder_name,
                    'Sigma_Lower': sigma_lower if sigma_lower is not None else 0.0,
                    'Sigma_Upper': sigma_upper if sigma_upper is not None else 0.0,
                    'Num_Covariates': data.num_covariates,
                    'R²': avg_r2,
                    'MSE': avg_mse,
                    'MAPE': avg_mape,
                    'Date_Time_Generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Seed_Value': seed,
                    'Other_Parameters': str(other_params)  # Convert dict to string for Excel
                }
                tracking_data.append(tracking_row)

            if output_format == "txt":
                out_path = os.path.join(output_dir, f"{fname}_stochastic.txt")

                with open(out_path, "w") as f:
                    f.write(f"# Stochastic CFLP Instance generated from {fname}\n")
                    f.write(f"# Parameters\n")
                    f.write(f"#   scenarios = {num_scenarios}\n")
                    f.write(f"#   target_cv = {target_cv if method.lower() not in ['merve_normal', 'arma(2,2)_with_covariate_generation'] else 'null'}\n")
                    f.write(f"#   sigma_lower = {sigma_lower if sigma_lower is not None else 'null'}\n")
                    f.write(f"#   sigma_upper = {sigma_upper if sigma_upper is not None else 'null'}\n")
                    f.write(f"#   num_covariates = {num_covariates if num_covariates is not None else 'null'}\n")
                    f.write(f"#   seed = {seed}\n")
                    f.write(f"#   distribution = {method}\n\n")

                    f.write(f"{len(data.J)} {len(data.I)} {len(data.S)}\n\n")

                    f.write("# Facility capacities\n")
                    f.write(" ".join(map(str, data.capacities.astype(int))) + "\n\n")

                    f.write("# Fixed costs\n")
                    f.write(" ".join(map(lambda x: str(int(x)) if float(x).is_integer() else str(x), data.fixed_costs)) + "\n\n")

                    f.write("# Original demands (means)\n")
                    f.write(" ".join(map(str, data.demands)) + "\n\n")

                    f.write("# Shipment costs (customers x facilities)\n")
                    for row in data.shipment_costs:
                        f.write(" ".join(map(str, row)) + "\n")
                    f.write("\n")

                    f.write("# Demand scenarios\n")
                    for idx, scenario in enumerate(data.scenarios):
                        f.write(f"# Scenario {idx + 1}\n")
                        f.write(" ".join(map(str, scenario)) + "\n")
                    f.write("\n")

                    # Add covariate vectors section for ARMA method
                    if method.lower() in ["arma(2,2)_with_covariate_generation", "merve_with_covariates"] and data.covariate_vectors is not None:
                        covariate_type = "ARMA-generated covariates" if method.lower() == "arma(2,2)_with_covariate_generation" else "PCA-generated covariates"
                        f.write(f"# Covariate vectors ({covariate_type})\n")
                        f.write(f"# Number of covariates: {data.num_covariates}\n")
                        for idx, covariate_vec in enumerate(data.covariate_vectors):
                            f.write(f"# Covariate vector for scenario {idx + 1}\n")
                            f.write(" ".join(map(str, covariate_vec)) + "\n")
                        f.write("\n")

                    f.write("# Diagnostics (single-file dispersion summaries)\n")
                    f.write(f"#   empirical_cv_mean = {data.empirical_cv_mean}\n")
                    f.write(f"#   r_overall = {data.r_overall}\n")
                    f.write(f"#   negatives_clipped = {data.num_negatives_clipped} of {data.total_draws}\n")
                    f.write(f"#   scaling_factor_applied = {data.scaling_factor_applied if method.lower() != 'merve_normal' else 'null'}\n")
                    f.write(f"#   sigma_lower = {data.sigma_lower if data.sigma_lower != 0.0 else 'null'}\n")
                    f.write(f"#   sigma_upper = {data.sigma_upper if data.sigma_upper != 0.0 else 'null'}\n")
                    f.write(f"#   num_covariates = {data.num_covariates if data.num_covariates > 0 else 'null'}\n")
                    if method.lower() in ["arma(2,2)_with_covariate_generation", "merve_with_covariates"] and data.arma_r2_scores is not None:
                        avg_r2 = float(np.mean(data.arma_r2_scores))
                        method_prefix = "arma" if method.lower() == "arma(2,2)_with_covariate_generation" else "merve_pca"
                        f.write(f"#   {method_prefix}_avg_r2_score = {avg_r2:.6f}\n")
                        if data.arma_validation_r2 is not None:
                            avg_val_r2 = float(np.mean(data.arma_validation_r2))
                            f.write(f"#   {method_prefix}_validation_r2_mean = {avg_val_r2:.6f}\n")
                        if data.arma_validation_mse is not None:
                            avg_val_mse = float(np.mean(data.arma_validation_mse))
                            f.write(f"#   {method_prefix}_validation_mse_mean = {avg_val_mse:.6f}\n")
                        if data.arma_validation_mape is not None:
                            avg_val_mape = float(np.mean(data.arma_validation_mape))
                            f.write(f"#   {method_prefix}_validation_mape_mean = {avg_val_mape:.2f}%\n")
                    f.write(f"#   max_demand_sum_over_scenario = {data.max_demand_sum_over_scenario}\n")

            elif output_format == "pkl":
                out_path = os.path.join(output_dir, f"{fname}.pkl")
                with open(out_path, "wb") as f:
                    pickle.dump(data, f)

            else:
                raise ValueError("Invalid output_format. Use 'txt' or 'pkl'.")

        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")

    print(f"\n✅ All selected files processed and saved to: {output_dir}")
    
    # Create Excel tracking file for MERVE_WITH_COVARIATES method
    if method.lower() == "merve_with_covariates" and tracking_data:
        try:
            excel_path = create_merve_covariates_excel_tracking(tracking_data, input_dir)
            print(f"📊 Excel tracking file created: {excel_path}")
        except Exception as e:
            print(f"⚠️ Failed to create Excel tracking file: {e}")


# ======================
# Run the function here
# ======================
if __name__ == "__main__":
    selected_files = [
        "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
        "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20",
        "p21", "p22", "p23", "p24", "p25", "p26", "p27", "p28", "p29", "p30",
        "p31", "p32", "p33", "p34", "p35", "p36", "p37", "p38", "p39", "p40",
        "p41", "p42", "p43", "p44", "p45", "p46", "p47", "p48", "p49", "p50",
        "p51", "p52", "p53", "p54", "p55", "p56", "p57", "p58", "p59", "p60",
        "p61", "p62", "p63", "p64", "p65", "p66", "p67", "p68", "p69", "p70",
        "p71"
    ]

    print("🚀 Generating SCFLP stochastic datasets (CV-driven, Merve_Normal, and ARMA)...")
    
    # Configure here: any list of scenario counts you want
    scenarios_to_generate = [300,500,700,900,1100,1300]  # e.g., [100, 400, 800]
    
    # CV-driven methods
    cvs_to_use = [0.1, 0.2, 0.3] #0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    cv_methods_to_use = ["merve_with_covariates"]
    
    # Merve_Normal method parameters
    merve_normal_params = [
        (0.1, 0.3)  # Default
        #(0.05, 0.25),  # Lower variability
        #(0.15, 0.4),   # Higher variability
    ]
    
    # ARMA(2,2) covariate generation method parameters
    arma_covariate_params = [
        3,  # Default number of covariates
        # 6,  # Alternative: more covariates
        # 8,  # Alternative: even more covariates
    ]
    
    # MERVE_WITH_COVARIATES method parameters  
    merve_covariate_params = [
        {
            "sigma_lower": 0.1,
            "sigma_upper": 0.3,
            "num_covariates": 3
        },
        # Alternative parameter sets:
        # {
        #     "sigma_lower": 0.05,
        #     "sigma_upper": 0.25,
        #     "num_covariates": 3
        # },
        # {
        #     "sigma_lower": 0.1,
        #     "sigma_upper": 0.3,
        #     "num_covariates": 5
        # },
    ]

    # Process CV-driven methods (traditional CV methods like uniform, normal, gamma, lognormal)
    cv_driven_methods = [m for m in cv_methods_to_use if m not in ["merve_normal", "arma(2,2)_with_covariate_generation", "merve_with_covariates"]]
    if cvs_to_use and cv_driven_methods:
        for num_scenarios in scenarios_to_generate:
            for target_cv in cvs_to_use:
                for method in cv_driven_methods:
                    print(f"\n📋 Processing CV-driven: scenarios={num_scenarios}, target_cv={target_cv}, dist={method}")
                    process_files(
                        input_dir="/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP",
                        file_list=selected_files,
                        num_scenarios=num_scenarios,
                        target_cv=target_cv,
                        method=method,
                        output_format="txt",  # or "pkl"
                        seed=42,
                        sigma_lower=None,  # Not used for CV methods
                        sigma_upper=None,  # Not used for CV methods
                    )
    
    # Process Merve_Normal method
    if "merve_normal" in cv_methods_to_use and merve_normal_params:
        for num_scenarios in scenarios_to_generate:
            for sigma_lower, sigma_upper in merve_normal_params:
                print(f"\n📋 Processing Merve_Normal: scenarios={num_scenarios}, sigma_range=[{sigma_lower}, {sigma_upper}]")
                process_files(
                    input_dir="/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP",
                    file_list=selected_files,
                    num_scenarios=num_scenarios,
                    target_cv=0.0,  # Not used for Merve_Normal
                    method="merve_normal",
                    output_format="txt",  # or "pkl"
                    seed=42,
                    sigma_lower=sigma_lower,
                    sigma_upper=sigma_upper,
                )
    
    # Process ARMA(2,2) covariate generation method
    if "arma(2,2)_with_covariate_generation" in cv_methods_to_use and arma_covariate_params:
        for num_scenarios in scenarios_to_generate:
            for num_covariates in arma_covariate_params:
                print(f"\n📋 Processing ARMA(2,2)_with_covariate_generation: scenarios={num_scenarios}, covariates={num_covariates}")
                process_files(
                    input_dir="/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP",
                    file_list=selected_files,
                    num_scenarios=num_scenarios,
                    target_cv=0.0,  # Not used for ARMA method
                    method="arma(2,2)_with_covariate_generation",
                    output_format="txt",  # or "pkl"
                    seed=42,
                    sigma_lower=None,  # Not used for ARMA method
                    sigma_upper=None,  # Not used for ARMA method
                    num_covariates=num_covariates,
                )
    
    # Process MERVE_WITH_COVARIATES method
    if "merve_with_covariates" in cv_methods_to_use and merve_covariate_params:
        for num_scenarios in scenarios_to_generate:
            for params in merve_covariate_params:
                sigma_lower = params["sigma_lower"]
                sigma_upper = params["sigma_upper"]
                num_covariates = params["num_covariates"]
                print(f"\n📋 Processing MERVE_WITH_COVARIATES: scenarios={num_scenarios}, sigma_range=[{sigma_lower}, {sigma_upper}], covariates={num_covariates}")
                process_files(
                    input_dir="/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP",
                    file_list=selected_files,
                    num_scenarios=num_scenarios,
                    target_cv=0.0,  # Not used for MERVE_WITH_COVARIATES method
                    method="merve_with_covariates",
                    output_format="txt",  # or "pkl"
                    seed=42,
                    sigma_lower=sigma_lower,
                    sigma_upper=sigma_upper,
                    num_covariates=num_covariates,
                )
