"""Statistical utility functions for TMB prediction project.

Provides wrappers for OLS, robust regression, Firth logistic regression,
progressive model comparison, and diagnostic computations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# OLS helpers
# ---------------------------------------------------------------------------


def fit_ols_formula(
    df: pd.DataFrame, formula: str
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit an OLS model using a statsmodels formula.

    Args:
        df: DataFrame containing model variables.
        formula: Patsy formula string (e.g. 'log_tmb ~ age + C(sex)').

    Returns:
        Fitted OLS results object.
    """
    model = sm.OLS.from_formula(formula, data=df)
    return model.fit()


def progressive_regression(
    df: pd.DataFrame,
    response: str,
    predictor_blocks: list[tuple[str, str]],
) -> pd.DataFrame:
    """Fit a sequence of nested OLS models, adding predictors progressively.

    Args:
        df: DataFrame containing all variables.
        response: Response variable name.
        predictor_blocks: List of (block_name, formula_term) tuples to add
            sequentially. Each term is appended to the formula.

    Returns:
        Summary DataFrame with columns: model, formula, n, adj_r2,
        delta_adj_r2, aic, bic, f_pvalue.
    """
    rows: list[dict[str, Any]] = []
    current_terms: list[str] = []
    prev_result = None

    for block_name, term in predictor_blocks:
        current_terms.append(term)
        formula = f"{response} ~ " + " + ".join(current_terms)

        try:
            result = fit_ols_formula(df, formula)
        except Exception as exc:
            print(f"  Warning: failed to fit '{formula}': {exc}")
            continue

        adj_r2 = result.rsquared_adj
        prev_adj_r2 = rows[-1]["adj_r2"] if rows else 0.0

        # Partial F-test vs previous model
        f_pval = np.nan
        if prev_result is not None:
            try:
                f_test = partial_f_test(prev_result, result)
                f_pval = f_test["p_value"]
            except Exception:
                pass

        rows.append(
            {
                "model": block_name,
                "formula": formula,
                "n": int(result.nobs),
                "adj_r2": adj_r2,
                "delta_adj_r2": adj_r2 - prev_adj_r2,
                "aic": result.aic,
                "bic": result.bic,
                "f_pvalue": f_pval,
            }
        )
        prev_result = result

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def partial_f_test(
    reduced: Any,
    full: Any,
) -> dict[str, float]:
    """Partial F-test comparing a reduced model to a full (nested) model.

    Args:
        reduced: Fitted reduced OLS model.
        full: Fitted full OLS model (superset of reduced).

    Returns:
        Dict with 'f_stat', 'df_num', 'df_denom', 'p_value'.
    """
    rss_reduced = reduced.ssr
    rss_full = full.ssr
    df_num = full.df_model - reduced.df_model
    df_denom = full.df_resid

    if df_num <= 0 or df_denom <= 0:
        return {
            "f_stat": np.nan,
            "df_num": df_num,
            "df_denom": df_denom,
            "p_value": np.nan,
        }

    f_stat = ((rss_reduced - rss_full) / df_num) / (rss_full / df_denom)
    p_value = 1 - sp_stats.f.cdf(f_stat, df_num, df_denom)
    return {
        "f_stat": f_stat,
        "df_num": df_num,
        "df_denom": df_denom,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------


def compute_vif(
    df: pd.DataFrame,
    predictors: list[str],
) -> pd.DataFrame:
    """Compute Variance Inflation Factors for a set of predictors.

    Args:
        df: DataFrame with predictor columns (numeric, no NAs).
        predictors: List of predictor column names.

    Returns:
        DataFrame with 'variable' and 'VIF' columns.
    """
    X = df[predictors].dropna()
    X = sm.add_constant(X)
    vif_data = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif_data.append(
            {"variable": col, "VIF": variance_inflation_factor(X.values, i)}
        )
    return pd.DataFrame(vif_data)


# ---------------------------------------------------------------------------
# Tidy model output
# ---------------------------------------------------------------------------


def extract_model_summary(result: Any) -> pd.DataFrame:
    """Extract a tidy coefficient table from a statsmodels result.

    Args:
        result: Fitted statsmodels results object.

    Returns:
        DataFrame with columns: variable, coef, std_err, t_stat,
        p_value, ci_lower, ci_upper.
    """
    summary = pd.DataFrame(
        {
            "variable": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "t_stat": result.tvalues.values,
            "p_value": result.pvalues.values,
        }
    )
    ci = result.conf_int()
    summary["ci_lower"] = ci.iloc[:, 0].values
    summary["ci_upper"] = ci.iloc[:, 1].values
    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Robust regression
# ---------------------------------------------------------------------------


def fit_robust_regression(
    df: pd.DataFrame,
    formula: str,
    m_estimator: str = "huber",
) -> Any:
    """Fit a robust linear model (RLM) using a statsmodels formula.

    Args:
        df: DataFrame containing model variables.
        formula: Patsy formula string.
        m_estimator: M-estimator type ('huber' or 'bisquare').

    Returns:
        Fitted RLM results object.
    """
    norm = (
        sm.robust.norms.HuberT()
        if m_estimator == "huber"
        else sm.robust.norms.TukeyBiweight()
    )
    model = sm.RLM.from_formula(formula, data=df, M=norm)
    return model.fit()


# ---------------------------------------------------------------------------
# Influence diagnostics
# ---------------------------------------------------------------------------


def cooks_distance(result: Any) -> pd.Series:
    """Compute Cook's distance for an OLS model.

    Args:
        result: Fitted OLS results object.

    Returns:
        Series of Cook's distance values.
    """
    influence = result.get_influence()
    cooks_d, _ = influence.cooks_distance
    return pd.Series(cooks_d, name="cooks_distance")


# ---------------------------------------------------------------------------
# Firth logistic regression
# ---------------------------------------------------------------------------


def _logistic(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute logistic probabilities, clipping for numerical stability."""
    z = X @ beta
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def fit_firth_logistic(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Fit Firth's bias-corrected logistic regression.

    Implements the penalized maximum likelihood approach of Firth (1993)
    using iteratively reweighted least squares with a Jeffreys-prior
    bias correction.

    Args:
        X: Feature matrix (no intercept column — one is prepended).
        y: Binary response vector.
        max_iter: Maximum number of IRLS iterations.
        tol: Convergence tolerance on the coefficient change norm.

    Returns:
        Dict with keys: 'coef', 'intercept', 'predictions',
        'probabilities', 'feature_names', 'n_iter'.
    """
    feature_names = (
        X.columns.tolist()
        if hasattr(X, "columns")
        else [f"x{i}" for i in range(X.shape[1])]
    )

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n, p = X_arr.shape

    # Prepend intercept column
    X_design = np.column_stack([np.ones(n), X_arr])
    k = X_design.shape[1]  # p + 1

    beta = np.zeros(k)

    for iteration in range(max_iter):
        pi = _logistic(X_design, beta)
        pi = np.clip(pi, 1e-10, 1 - 1e-10)

        W = np.diag(pi * (1 - pi))
        XtWX = X_design.T @ W @ X_design

        # Hat matrix diagonal for Firth correction
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)

        W_sqrt = np.diag(np.sqrt(pi * (1 - pi)))
        H = W_sqrt @ X_design @ XtWX_inv @ X_design.T @ W_sqrt
        h = np.diag(H)

        # Modified score (Firth correction)
        U = X_design.T @ (y_arr - pi + h * (0.5 - pi))

        # Newton step
        delta = XtWX_inv @ U
        beta += delta

        if np.linalg.norm(delta) < tol:
            break

    pi_final = _logistic(X_design, beta)

    return {
        "coef": beta[1:].reshape(1, -1),
        "intercept": beta[0],
        "predictions": (pi_final >= 0.5).astype(int),
        "probabilities": pi_final,
        "feature_names": feature_names,
        "n_iter": iteration + 1,
    }


# ---------------------------------------------------------------------------
# Likelihood ratio test
# ---------------------------------------------------------------------------


def likelihood_ratio_test(
    ll_reduced: float,
    ll_full: float,
    df_diff: int,
) -> dict[str, float]:
    """Perform a likelihood ratio test between nested models.

    Args:
        ll_reduced: Log-likelihood of the reduced model.
        ll_full: Log-likelihood of the full model.
        df_diff: Difference in degrees of freedom.

    Returns:
        Dict with 'chi2', 'df', 'p_value'.
    """
    chi2 = -2 * (ll_reduced - ll_full)
    p_value = 1 - sp_stats.chi2.cdf(chi2, df_diff)
    return {"chi2": chi2, "df": df_diff, "p_value": p_value}
