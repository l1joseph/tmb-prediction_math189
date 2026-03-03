"""Preprocessing module for TMB prediction project.

Handles dataset merging, TMB computation, feature engineering,
and complete-case subsetting for modeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Barcode extraction
# ---------------------------------------------------------------------------


def extract_patient_barcode(sample_id: str) -> str:
    """Extract the 12-character TCGA patient barcode from a sample ID.

    Args:
        sample_id: Full TCGA sample barcode (e.g. 'TCGA-AB-1234-01').

    Returns:
        First 12 characters (e.g. 'TCGA-AB-1234').
    """
    return str(sample_id)[:12]


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_datasets(
    cdr_df: pd.DataFrame,
    cbio_df: pd.DataFrame,
    aneuploidy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the three TCGA datasets on patient barcode.

    Uses left joins starting from the cBioPortal data (which has
    mutation counts) to preserve the maximum number of samples
    with TMB information.

    Args:
        cdr_df: TCGA-CDR clinical data.
        cbio_df: cBioPortal clinical data (wide format).
        aneuploidy_df: Taylor et al. aneuploidy/WGD data.

    Returns:
        Merged DataFrame with a ``patient_barcode`` column.
    """
    # Ensure patient barcode columns exist
    if "patient_barcode" not in cbio_df.columns:
        if "patient_id" in cbio_df.columns:
            cbio_df = cbio_df.copy()
            cbio_df["patient_barcode"] = cbio_df["patient_id"].apply(
                extract_patient_barcode
            )
        elif "sample_id" in cbio_df.columns:
            cbio_df = cbio_df.copy()
            cbio_df["patient_barcode"] = cbio_df["sample_id"].apply(
                extract_patient_barcode
            )

    if "patient_barcode" not in cdr_df.columns:
        cdr_df = cdr_df.copy()
        for col in ["patient_barcode", "bcr_patient_barcode"]:
            if col in cdr_df.columns:
                cdr_df = cdr_df.rename(columns={col: "patient_barcode"})
                break

    if "patient_barcode" not in aneuploidy_df.columns:
        aneuploidy_df = aneuploidy_df.copy()
        sample_col = aneuploidy_df.columns[0]
        aneuploidy_df["patient_barcode"] = (
            aneuploidy_df[sample_col].astype(str).str[:12]
        )

    # De-duplicate cBioPortal to one row per patient (keep first sample)
    cbio_dedup = cbio_df.drop_duplicates(subset="patient_barcode", keep="first")

    # De-duplicate aneuploidy to one row per patient
    aneuploidy_dedup = aneuploidy_df.drop_duplicates(
        subset="patient_barcode", keep="first"
    )

    # Left join: cBioPortal ← CDR ← aneuploidy
    merged = cbio_dedup.merge(
        cdr_df, on="patient_barcode", how="left", suffixes=("", "_cdr")
    )
    merged = merged.merge(
        aneuploidy_dedup,
        on="patient_barcode",
        how="left",
        suffixes=("", "_taylor"),
    )

    return merged


# ---------------------------------------------------------------------------
# TMB computation
# ---------------------------------------------------------------------------

EXOME_SIZE_MB: float = 30.0


def compute_tmb(
    df: pd.DataFrame,
    mutation_col: str = "mutation_count",
    exome_size: float = EXOME_SIZE_MB,
) -> pd.DataFrame:
    """Compute TMB, log-TMB, and TMB-high indicator.

    Args:
        df: DataFrame with a mutation count column.
        mutation_col: Name of the column with raw mutation counts.
        exome_size: Exome capture size in megabases (default 30 Mb).

    Returns:
        DataFrame with added columns: ``tmb``, ``log_tmb``, ``tmb_high``.
    """
    df = df.copy()
    df["tmb"] = df[mutation_col] / exome_size
    df["log_tmb"] = np.log1p(df["tmb"])
    df["tmb_high"] = (df["tmb"] >= 10).astype(int)
    # Flag hypermutators (TMB > 50 mut/Mb — POLE/MSI-H outliers)
    df["hypermutator"] = (df["tmb"] > 50).astype(int)
    return df


# ---------------------------------------------------------------------------
# Cleaning & encoding
# ---------------------------------------------------------------------------


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged data and engineer features for modeling.

    Operations:
    - Coerce numeric columns to proper dtypes
    - Binarize MSI status (MANTIS score >= 0.4 → MSI-H)
    - Derive WGD status from genome doubling annotations
    - Standardize categorical columns

    Args:
        df: Merged DataFrame from ``merge_datasets``.

    Returns:
        Cleaned DataFrame ready for analysis.
    """
    df = df.copy()

    # --- Numeric coercion ---
    numeric_cols = [
        "mutation_count",
        "tmb",
        "log_tmb",
        "fraction_genome_altered",
        "aneuploidy_score",
        "msi_score_mantis",
        "msi_sensor_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Age ---
    age_candidates = [
        "age_at_initial_pathologic_diagnosis",
        "age_at_diagnosis",
        "AGE",
        "age",
    ]
    for col in age_candidates:
        if col in df.columns:
            df["age_at_diagnosis"] = pd.to_numeric(df[col], errors="coerce")
            break

    # --- Sex ---
    sex_candidates = ["gender", "sex", "GENDER", "SEX"]
    for col in sex_candidates:
        if col in df.columns:
            df["sex"] = df[col].astype(str).str.upper().str.strip()
            break

    # --- Cancer type normalization ---
    # Prefer the cBioPortal cancer_type; fall back to CDR "type"
    if "cancer_type" not in df.columns:
        for col in ["type", "cancer_type_cdr"]:
            if col in df.columns:
                df["cancer_type"] = df[col]
                break

    if "cancer_type" in df.columns:
        df["cancer_type"] = df["cancer_type"].astype(str).str.strip()

    # --- MSI binarization ---
    if "msi_score_mantis" in df.columns:
        df["msi_status"] = np.where(
            df["msi_score_mantis"].isna(),
            np.nan,
            np.where(df["msi_score_mantis"] >= 0.4, "MSI-H", "MSS"),
        )
    elif "msi_sensor_score" in df.columns:
        # Fallback: MSIsensor score >= 3.5 used in some studies
        df["msi_status"] = np.where(
            df["msi_sensor_score"].isna(),
            np.nan,
            np.where(df["msi_sensor_score"] >= 3.5, "MSI-H", "MSS"),
        )

    # --- WGD status ---
    wgd_candidates = [
        "Genome_doublings",
        "Genome doublings",
        "genome_doublings",
        "WGD",
        "wgd",
    ]
    for col in wgd_candidates:
        if col in df.columns:
            df["wgd_status"] = (
                pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
            )
            break

    return df


# ---------------------------------------------------------------------------
# Complete-case subsetting
# ---------------------------------------------------------------------------


def get_model_df(
    df: pd.DataFrame,
    predictors: list[str],
    response: str,
) -> pd.DataFrame:
    """Subset to complete cases for a given set of model variables.

    Args:
        df: Full DataFrame.
        predictors: List of predictor column names.
        response: Name of the response column.

    Returns:
        DataFrame with no missing values in the specified columns.
    """
    cols = [response] + predictors
    available = [c for c in cols if c in df.columns]
    missing = set(cols) - set(available)
    if missing:
        print(f"  Warning: columns not found in data: {missing}")
    return df[available].dropna().reset_index(drop=True)
