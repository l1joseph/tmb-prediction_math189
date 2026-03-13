"""Reusable visualization functions for TMB prediction project.

Provides consistent styling and common plot types used across notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def set_style() -> None:
    """Apply project-wide matplotlib/seaborn style settings."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )


def save_fig(
    fig: Figure,
    name: str,
    figures_dir: Path | str = "figures",
) -> None:
    """Save a figure as both PNG and PDF.

    Args:
        fig: Matplotlib figure object.
        name: Base filename (without extension).
        figures_dir: Directory to save into.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png")
    fig.savefig(figures_dir / f"{name}.pdf")
    print(f"  Saved: {name}.png, {name}.pdf")


# ---------------------------------------------------------------------------
# TMB distribution plots
# ---------------------------------------------------------------------------


def plot_tmb_distribution(
    df: pd.DataFrame,
    tmb_col: str = "tmb",
    log_tmb_col: str = "log_tmb",
) -> Figure:
    """Side-by-side histograms of raw TMB and log-transformed TMB.

    Annotates each panel with skewness.

    Args:
        df: DataFrame containing TMB columns.
        tmb_col: Column name for raw TMB.
        log_tmb_col: Column name for log-transformed TMB.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw TMB
    data_raw = df[tmb_col].dropna()
    axes[0].hist(data_raw, bins=80, edgecolor="white", alpha=0.8, color="#4C72B0")
    skew_raw = sp_stats.skew(data_raw)
    axes[0].axvline(10, color="red", ls="--", lw=1.5, label="FDA cutoff (10 mut/Mb)")
    axes[0].set_title("Raw TMB Distribution")
    axes[0].set_xlabel("TMB (mutations / Mb)")
    axes[0].set_ylabel("Count")
    axes[0].annotate(
        f"Skewness = {skew_raw:.2f}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
    )
    axes[0].legend()

    # Log TMB
    data_log = df[log_tmb_col].dropna()
    axes[1].hist(data_log, bins=60, edgecolor="white", alpha=0.8, color="#55A868")
    skew_log = sp_stats.skew(data_log)
    axes[1].axvline(
        np.log1p(10), color="red", ls="--", lw=1.5, label="FDA cutoff (log scale)"
    )
    axes[1].set_title("Log-Transformed TMB Distribution")
    axes[1].set_xlabel("log(1 + TMB)")
    axes[1].set_ylabel("Count")
    axes[1].annotate(
        f"Skewness = {skew_log:.2f}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
    )
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_tmb_by_cancer_type(
    df: pd.DataFrame,
    kind: str = "box",
    tmb_col: str = "log_tmb",
    type_col: str = "cancer_type",
) -> Figure:
    """Box or violin plot of TMB by cancer type, sorted by median.

    Args:
        df: DataFrame with TMB and cancer type columns.
        kind: 'box' or 'violin'.
        tmb_col: TMB column to plot.
        type_col: Cancer type column.

    Returns:
        Matplotlib Figure.
    """
    data = df[[type_col, tmb_col]].dropna()
    order = data.groupby(type_col)[tmb_col].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(16, 7))
    if kind == "violin":
        sns.violinplot(
            data=data, x=type_col, y=tmb_col, order=order, ax=ax, cut=0, scale="width"
        )
    else:
        sns.boxplot(
            data=data,
            x=type_col,
            y=tmb_col,
            order=order,
            ax=ax,
            fliersize=2,
            showfliers=True,
        )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
    ax.set_xlabel("Cancer Type")
    ax.set_ylabel(tmb_col.replace("_", " ").title())
    ax.set_title(f"{tmb_col.replace('_', ' ').title()} by Cancer Type")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def plot_residual_diagnostics(
    fitted: np.ndarray,
    residuals: np.ndarray,
    model_name: str = "",
    qq_residuals: np.ndarray | None = None,
    qq_compare_residuals: np.ndarray | None = None,
    qq_compare_label: str = "Comparison",
    qq_show_ci: bool = True,
    qq_ci_level: float = 0.95,
    qq_max_points: int = 4000,
    qq_random_state: int = 42,
) -> Figure:
    """Four-panel residual diagnostic plot.

    Panels: residuals vs fitted, Q-Q plot, scale-location, residual histogram.
    Q-Q can optionally use externally studentized residuals and overlay
    a comparison set (e.g., hypermutators excluded).

    Args:
        fitted: Fitted/predicted values.
        residuals: Residuals from the model.
        model_name: Optional title prefix.
        qq_residuals: Optional residual vector used only for Q-Q panel.
            If None, uses ``residuals``.
        qq_compare_residuals: Optional second residual vector to overlay
            in the Q-Q panel.
        qq_compare_label: Legend label for ``qq_compare_residuals``.
        qq_show_ci: Whether to draw a simulation-based confidence envelope
            for the primary Q-Q series.
        qq_ci_level: Confidence level for Q-Q envelope.
        qq_max_points: Max points displayed in Q-Q scatter (subsampled).
        qq_random_state: Random seed for reproducible Q-Q subsampling.

    Returns:
        Matplotlib Figure.
    """
    def _to_clean_array(arr: np.ndarray) -> np.ndarray:
        vals = np.asarray(arr, dtype=float).reshape(-1)
        return vals[np.isfinite(vals)]

    def _qq_components(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        y = np.sort(_to_clean_array(arr))
        n = len(y)
        if n < 2:
            # Degenerate case: return a minimal identity mapping.
            x = np.array([0.0])
            return x, np.array([y[0] if n == 1 else 0.0]), 1.0, 0.0
        probs = (np.arange(1, n + 1) - 0.5) / n
        x = sp_stats.norm.ppf(probs)
        slope, intercept = np.polyfit(x, y, 1)
        return x, y, float(slope), float(intercept)

    def _qq_envelope(
        n: int,
        slope: float,
        intercept: float,
        ci_level: float,
        random_state: int,
        n_sim: int = 300,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        sims = rng.standard_normal((n_sim, n))
        sims.sort(axis=1)
        probs = (np.arange(1, n + 1) - 0.5) / n
        x = sp_stats.norm.ppf(probs)
        alpha = 1 - ci_level
        lo = np.quantile(sims, alpha / 2, axis=0)
        hi = np.quantile(sims, 1 - alpha / 2, axis=0)
        # Match the fitted reference line scale/location.
        lo = intercept + slope * lo
        hi = intercept + slope * hi
        return x, lo, hi

    residuals_arr = np.asarray(residuals, dtype=float).reshape(-1)
    fitted_arr = np.asarray(fitted, dtype=float).reshape(-1)
    finite_main = np.isfinite(fitted_arr) & np.isfinite(residuals_arr)
    resid_clean = residuals_arr[np.isfinite(residuals_arr)]
    resid_std = resid_clean.std()
    std_resid = (
        (resid_clean - resid_clean.mean()) / resid_std
        if resid_std > 0
        else np.zeros_like(resid_clean)
    )
    sqrt_abs_std = np.sqrt(np.abs(std_resid))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    title_prefix = f"{model_name} — " if model_name else ""

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(
        fitted_arr[finite_main], residuals_arr[finite_main], alpha=0.3, s=10, color="#4C72B0"
    )
    axes[0, 0].axhline(0, color="red", ls="--", lw=1)
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title(f"{title_prefix}Residuals vs Fitted")
    # Add lowess trend
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smooth = lowess(residuals_arr[finite_main], fitted_arr[finite_main], frac=0.3)
        axes[0, 0].plot(smooth[:, 0], smooth[:, 1], color="red", lw=2)
    except ImportError:
        pass

    # 2. Q-Q Plot (supports studentized residuals + comparison overlay)
    qq_base = _to_clean_array(qq_residuals if qq_residuals is not None else residuals)
    x, y, slope, intercept = _qq_components(qq_base)
    rng = np.random.default_rng(qq_random_state)
    if len(x) > qq_max_points:
        idx = np.sort(rng.choice(len(x), size=qq_max_points, replace=False))
        x_plot, y_plot = x[idx], y[idx]
    else:
        x_plot, y_plot = x, y
    axes[0, 1].scatter(x_plot, y_plot, s=10, alpha=0.35, color="#4C72B0", label="Primary")
    axes[0, 1].plot(x, intercept + slope * x, color="red", lw=2, label="Reference line")

    if qq_show_ci and len(x) > 10:
        x_ci, lo, hi = _qq_envelope(
            n=len(x),
            slope=slope,
            intercept=intercept,
            ci_level=qq_ci_level,
            random_state=qq_random_state,
        )
        axes[0, 1].fill_between(
            x_ci,
            lo,
            hi,
            color="red",
            alpha=0.12,
            linewidth=0,
            label=f"{int(qq_ci_level * 100)}% envelope",
        )

    if qq_compare_residuals is not None:
        x2, y2, slope2, intercept2 = _qq_components(qq_compare_residuals)
        if len(x2) > qq_max_points:
            idx2 = np.sort(rng.choice(len(x2), size=qq_max_points, replace=False))
            x2_plot, y2_plot = x2[idx2], y2[idx2]
        else:
            x2_plot, y2_plot = x2, y2
        axes[0, 1].scatter(
            x2_plot,
            y2_plot,
            s=10,
            alpha=0.35,
            color="#55A868",
            label=qq_compare_label,
        )
        axes[0, 1].plot(x2, intercept2 + slope2 * x2, color="#1F7A4C", lw=1.8, alpha=0.9)

    axes[0, 1].set_xlabel("Theoretical Quantiles")
    axes[0, 1].set_ylabel("Sample Quantiles")
    axes[0, 1].set_title(f"{title_prefix}Normal Q-Q")
    axes[0, 1].legend(loc="best", fontsize=8)

    # 3. Scale-Location
    axes[1, 0].scatter(
        fitted_arr[finite_main], sqrt_abs_std, alpha=0.3, s=10, color="#55A868"
    )
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel(r"$\sqrt{|Standardized\ Residuals|}$")
    axes[1, 0].set_title(f"{title_prefix}Scale-Location")
    try:
        smooth2 = lowess(sqrt_abs_std, fitted_arr[finite_main], frac=0.3)
        axes[1, 0].plot(smooth2[:, 0], smooth2[:, 1], color="red", lw=2)
    except Exception:
        pass

    # 4. Standardized Residuals histogram
    axes[1, 1].hist(std_resid, bins=50, edgecolor="white", alpha=0.8, color="#C44E52")
    axes[1, 1].set_xlabel("Standardized Residuals")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title(f"{title_prefix}Residual Distribution")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# VIF bar chart
# ---------------------------------------------------------------------------


def plot_vif_bar(vif_df: pd.DataFrame) -> Figure:
    """Horizontal bar chart of Variance Inflation Factors.

    Args:
        vif_df: DataFrame with 'variable' and 'VIF' columns.

    Returns:
        Matplotlib Figure.
    """
    vif_sorted = vif_df.sort_values("VIF", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(vif_sorted) * 0.4)))
    ax.barh(vif_sorted["variable"], vif_sorted["VIF"], color="#4C72B0", alpha=0.8)
    ax.axvline(5, color="orange", ls="--", lw=1.5, label="VIF = 5")
    ax.axvline(10, color="red", ls="--", lw=1.5, label="VIF = 10")
    ax.set_xlabel("VIF")
    ax.set_title("Variance Inflation Factors")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Coefficient forest plot
# ---------------------------------------------------------------------------


def plot_coefficient_comparison(
    models: dict[str, pd.DataFrame],
    variables: list[str] | None = None,
) -> Figure:
    """Forest plot comparing coefficients across progressive models.

    Args:
        models: Dict mapping model name to tidy coefficient DataFrames
            with columns 'variable', 'coef', 'ci_lower', 'ci_upper'.
        variables: Subset of variables to plot. If None, uses all from
            the last model.

    Returns:
        Matplotlib Figure.
    """
    model_names = list(models.keys())
    if variables is None:
        last_df = models[model_names[-1]]
        variables = [
            v for v in last_df["variable"].tolist() if not v.startswith("C(cancer_type")
        ]

    n_vars = len(variables)
    n_models = len(model_names)
    fig, ax = plt.subplots(figsize=(10, max(4, n_vars * 0.6)))

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    offsets = np.linspace(-0.15, 0.15, n_models)

    for j, (mname, mdf) in enumerate(models.items()):
        for i, var in enumerate(variables):
            row = mdf[mdf["variable"] == var]
            if row.empty:
                continue
            coef = row["coef"].values[0]
            ci_lo = row["ci_lower"].values[0]
            ci_hi = row["ci_upper"].values[0]
            ax.errorbar(
                coef,
                i + offsets[j],
                xerr=[[coef - ci_lo], [ci_hi - coef]],
                fmt="o",
                color=colors[j],
                label=mname if i == 0 else "",
                capsize=3,
                markersize=5,
            )

    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(variables)
    ax.set_xlabel("Coefficient")
    ax.set_title("Coefficient Comparison Across Models")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Classification plots
# ---------------------------------------------------------------------------


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "Model",
) -> Figure:
    """ROC curve with AUC annotation.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for the positive class.
        label: Legend label.

    Returns:
        Matplotlib Figure.
    """
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> Figure:
    """Annotated heatmap confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class labels for axes.

    Returns:
        Matplotlib Figure.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    labels = labels or ["TMB-Low", "TMB-High"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig
