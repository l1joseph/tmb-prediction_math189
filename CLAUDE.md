# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predicting Tumor Mutational Burden (TMB) from clinical and genomic features across cancer types using TCGA Pan-Cancer Atlas data. TMB (somatic mutations per megabase) is a key immunotherapy biomarker; the FDA-approved threshold is ≥10 mut/Mb for pembrolizumab eligibility.

## Data Sources

Three TCGA Pan-Cancer Atlas datasets, merged by TCGA patient barcode:

1. **TCGA-CDR clinical data** (Liu et al. 2018): demographics (age, sex, race), tumor staging, survival — from GitHub mirror at `GerkeLab/TCGAclinical` or Cell supplementary files
2. **Mutation counts** from cBioPortal (`TCGA PanCancer Atlas` study): per-sample total mutation counts; normalize by ~30 Mb exome capture size to compute TMB (mut/Mb)
3. **Copy number/aneuploidy data** (Taylor et al. 2018): aneuploidy score, fraction genome altered, whole-genome doubling (WGD) status, microsatellite instability (MSI) classification — from GDC publications page

## Analysis Pipeline

### Phase 1: Data Preparation
- Merge three datasets on TCGA patient barcode
- Compute TMB = mutation_count / 30 (mut/Mb)
- Log-transform TMB (`log_TMB`) to address right-skew
- Binarize TMB: `TMB_high = 1 if TMB ≥ 10 else 0`

### Phase 2: EDA
- TMB distribution across cancer types (boxplots, violin plots)
- Assess need for log-transformation via histograms and skewness

### Phase 3: Linear Regression
- Simple: `log(TMB) ~ age`
- Progressive multiple regression adding: cancer type, sex, MSI status, aneuploidy score, fraction genome altered
- Track adjusted R² and coefficient significance at each step
- **Interaction terms**: `age × cancer_type` (clock-like mutation rates vary by tissue), `MSI × aneuploidy` (MSI-high has distinct mutational landscape)
- Regressing out cancer type to reveal subtler within-type predictor–TMB relationships

### Phase 4: Logistic Regression
- Predict `TMB_high` from same predictor set
- Apply **Firth's bias-corrected logistic regression** (`firthlogist` or `statsmodels` with penalized likelihood) for quasi-complete separation in rare cancer subtypes

### Phase 5: Diagnostics
- Residual plots (homoscedasticity), Q-Q plots (normality), VIF (multicollinearity)
- Robust regression for influential observations (melanoma, endometrial known outliers)

## Key Variables

| Variable | Type | Role |
|---|---|---|
| `log_TMB` | continuous | response (linear regression) |
| `TMB_high` | binary (≥10 mut/Mb) | response (logistic regression) |
| `age_at_diagnosis` | continuous | predictor |
| `cancer_type` | categorical (33 types) | predictor |
| `sex` | categorical | predictor |
| `MSI_status` | categorical | predictor |
| `aneuploidy_score` | continuous | predictor |
| `fraction_genome_altered` | continuous | predictor |
| `WGD_status` | binary | predictor |

## Python Environment

```bash
conda create -n tmb python=3.11
conda activate tmb
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn jupyter openpyxl
pip install firthlogist  # Firth's penalized logistic regression
```

## Conventions

- Use Jupyter notebooks in `notebooks/` for analysis
- Use `src/` for reusable Python modules (data loading, preprocessing, plotting utilities)
- Follow Google-style docstrings and type hints on all function signatures
- Use `pathlib.Path` for file paths
- Format with `black .` (88 char line length), lint with `ruff check .`
- This is a Math 189 (UCSD) course project — prioritize interpretability and statistical rigor over predictive performance
