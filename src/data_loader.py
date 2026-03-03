"""Data acquisition module for TCGA Pan-Cancer Atlas datasets.

Downloads and loads three datasets with local caching:
1. TCGA-CDR clinical data (Liu et al. 2018)
2. Mutation counts + genomic features from cBioPortal API
3. Aneuploidy/WGD data (Taylor et al. 2018)
"""

from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TCGA_CDR_URL = "https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"

ANEUPLOIDY_URL = "https://api.gdc.cancer.gov/data/4c35f34f-b0f3-4891-8794-4840dd748aad"

ABSOLUTE_PURITY_URL = (
    "https://api.gdc.cancer.gov/data/4f277128-f793-4354-a13d-30cc7fe9f6b5"
)

CBIOPORTAL_BASE = "https://www.cbioportal.org/api"

# 32 TCGA PanCancer Atlas study IDs on cBioPortal
PANCAN_STUDY_IDS: list[str] = [
    "acc_tcga_pan_can_atlas_2018",
    "blca_tcga_pan_can_atlas_2018",
    "brca_tcga_pan_can_atlas_2018",
    "cesc_tcga_pan_can_atlas_2018",
    "chol_tcga_pan_can_atlas_2018",
    "coadread_tcga_pan_can_atlas_2018",
    "dlbc_tcga_pan_can_atlas_2018",
    "esca_tcga_pan_can_atlas_2018",
    "gbm_tcga_pan_can_atlas_2018",
    "hnsc_tcga_pan_can_atlas_2018",
    "kich_tcga_pan_can_atlas_2018",
    "kirc_tcga_pan_can_atlas_2018",
    "kirp_tcga_pan_can_atlas_2018",
    "laml_tcga_pan_can_atlas_2018",
    "lgg_tcga_pan_can_atlas_2018",
    "lihc_tcga_pan_can_atlas_2018",
    "luad_tcga_pan_can_atlas_2018",
    "lusc_tcga_pan_can_atlas_2018",
    "meso_tcga_pan_can_atlas_2018",
    "ov_tcga_pan_can_atlas_2018",
    "paad_tcga_pan_can_atlas_2018",
    "pcpg_tcga_pan_can_atlas_2018",
    "prad_tcga_pan_can_atlas_2018",
    "sarc_tcga_pan_can_atlas_2018",
    "skcm_tcga_pan_can_atlas_2018",
    "stad_tcga_pan_can_atlas_2018",
    "tgct_tcga_pan_can_atlas_2018",
    "thca_tcga_pan_can_atlas_2018",
    "thym_tcga_pan_can_atlas_2018",
    "ucec_tcga_pan_can_atlas_2018",
    "ucs_tcga_pan_can_atlas_2018",
    "uvm_tcga_pan_can_atlas_2018",
]

CBIOPORTAL_ATTRIBUTES: list[str] = [
    "MUTATION_COUNT",
    "FRACTION_GENOME_ALTERED",
    "ANEUPLOIDY_SCORE",
    "MSI_SCORE_MANTIS",
    "MSI_SENSOR_SCORE",
    "CANCER_TYPE",
    "CANCER_TYPE_DETAILED",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path, description: str = "") -> Path:
    """Download a file with a progress bar, skipping if it already exists."""
    if dest.exists():
        print(f"  [cached] {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {description or dest.name} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, disable=total == 0
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            pbar.update(len(chunk))
    return dest


# ---------------------------------------------------------------------------
# Dataset 1 — TCGA-CDR clinical
# ---------------------------------------------------------------------------


def download_tcga_cdr(raw_dir: Path, force: bool = False) -> Path:
    """Download TCGA-CDR clinical Excel file from GDC.

    Args:
        raw_dir: Directory to save the downloaded file.
        force: Re-download even if cached.

    Returns:
        Path to the downloaded xlsx file.
    """
    dest = raw_dir / "TCGA-CDR.xlsx"
    if force and dest.exists():
        dest.unlink()
    return _download_file(TCGA_CDR_URL, dest, "TCGA-CDR clinical data")


def load_tcga_cdr(raw_dir: Path) -> pd.DataFrame:
    """Load TCGA-CDR clinical data from the cached Excel file.

    Args:
        raw_dir: Directory containing the downloaded file.

    Returns:
        DataFrame with clinical data indexed by patient barcode.
    """
    path = raw_dir / "TCGA-CDR.xlsx"
    if not path.exists():
        path = download_tcga_cdr(raw_dir)
    # The TCGA-CDR file has a sheet named "TCGA-CDR"
    df = pd.read_excel(path, sheet_name="TCGA-CDR", engine="openpyxl")
    # Standardize barcode column
    if "bcr_patient_barcode" in df.columns:
        df = df.rename(columns={"bcr_patient_barcode": "patient_barcode"})
    return df


# ---------------------------------------------------------------------------
# Dataset 2 — cBioPortal clinical data
# ---------------------------------------------------------------------------


def download_cbioportal_clinical(
    raw_dir: Path,
    study_ids: list[str] | None = None,
    attributes: list[str] | None = None,
    force: bool = False,
) -> Path:
    """Fetch clinical attributes from cBioPortal API for PanCancer studies.

    Queries the ``/clinical-data`` endpoint for each study, collecting
    sample-level attributes. Results are cached as a single TSV file.

    Args:
        raw_dir: Directory to save the cached TSV.
        study_ids: List of cBioPortal study IDs. Defaults to all 32 PanCan.
        attributes: Clinical attribute IDs to fetch. Defaults to standard set.
        force: Re-download even if cached.

    Returns:
        Path to the cached TSV file.
    """
    dest = raw_dir / "cbioportal_clinical.tsv"
    if dest.exists() and not force:
        print(f"  [cached] {dest.name}")
        return dest

    study_ids = study_ids or PANCAN_STUDY_IDS
    attributes = attributes or CBIOPORTAL_ATTRIBUTES

    all_rows: list[dict] = []
    for study_id in tqdm(study_ids, desc="cBioPortal studies"):
        url = f"{CBIOPORTAL_BASE}/studies/{study_id}/clinical-data"
        params = {
            "clinicalDataType": "SAMPLE",
            "projection": "SUMMARY",
        }
        headers = {"Accept": "application/json"}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"  Warning: failed to fetch {study_id}: {exc}")
            continue

        for record in data:
            attr_id = record.get("clinicalAttributeId", "")
            if attr_id in attributes:
                all_rows.append(
                    {
                        "sample_id": record.get("sampleId", ""),
                        "patient_id": record.get("patientId", ""),
                        "study_id": study_id,
                        "attribute": attr_id,
                        "value": record.get("value", ""),
                    }
                )

    long_df = pd.DataFrame(all_rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(dest, sep="\t", index=False)
    print(f"  Saved {len(long_df)} records to {dest.name}")
    return dest


def load_cbioportal_clinical(raw_dir: Path) -> pd.DataFrame:
    """Load cached cBioPortal clinical data and pivot to wide format.

    Args:
        raw_dir: Directory containing the cached TSV.

    Returns:
        DataFrame with one row per sample, columns for each attribute.
    """
    path = raw_dir / "cbioportal_clinical.tsv"
    if not path.exists():
        path = download_cbioportal_clinical(raw_dir)

    long_df = pd.read_csv(path, sep="\t", dtype=str)

    # Pivot from long to wide: one row per (sample_id, patient_id, study_id)
    wide_df = long_df.pivot_table(
        index=["sample_id", "patient_id", "study_id"],
        columns="attribute",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide_df.columns.name = None  # remove pivot artifact

    # Rename to snake_case
    rename_map = {
        "MUTATION_COUNT": "mutation_count",
        "FRACTION_GENOME_ALTERED": "fraction_genome_altered",
        "ANEUPLOIDY_SCORE": "aneuploidy_score",
        "MSI_SCORE_MANTIS": "msi_score_mantis",
        "MSI_SENSOR_SCORE": "msi_sensor_score",
        "CANCER_TYPE": "cancer_type",
        "CANCER_TYPE_DETAILED": "cancer_type_detailed",
    }
    wide_df = wide_df.rename(columns=rename_map)

    # Coerce numeric columns
    for col in [
        "mutation_count",
        "fraction_genome_altered",
        "aneuploidy_score",
        "msi_score_mantis",
        "msi_sensor_score",
    ]:
        if col in wide_df.columns:
            wide_df[col] = pd.to_numeric(wide_df[col], errors="coerce")

    return wide_df


# ---------------------------------------------------------------------------
# Dataset 3 — Aneuploidy / WGD (Taylor et al. 2018)
# ---------------------------------------------------------------------------


def download_aneuploidy_data(raw_dir: Path, force: bool = False) -> dict[str, Path]:
    """Download aneuploidy and ABSOLUTE purity/ploidy files from GDC.

    Args:
        raw_dir: Directory to save downloaded files.
        force: Re-download even if cached.

    Returns:
        Dict mapping dataset name to local file path.
    """
    paths: dict[str, Path] = {}

    aneuploidy_dest = raw_dir / "taylor_aneuploidy.tsv"
    if force and aneuploidy_dest.exists():
        aneuploidy_dest.unlink()
    paths["aneuploidy"] = _download_file(
        ANEUPLOIDY_URL, aneuploidy_dest, "Taylor aneuploidy scores"
    )

    absolute_dest = raw_dir / "taylor_absolute_purity.tsv"
    if force and absolute_dest.exists():
        absolute_dest.unlink()
    paths["absolute"] = _download_file(
        ABSOLUTE_PURITY_URL, absolute_dest, "ABSOLUTE purity/ploidy"
    )

    return paths


def load_aneuploidy_data(raw_dir: Path) -> pd.DataFrame:
    """Load and merge aneuploidy + ABSOLUTE purity data.

    Args:
        raw_dir: Directory containing downloaded files.

    Returns:
        DataFrame with aneuploidy score, WGD status, purity, ploidy per sample.
    """
    aneuploidy_path = raw_dir / "taylor_aneuploidy.tsv"
    absolute_path = raw_dir / "taylor_absolute_purity.tsv"

    if not aneuploidy_path.exists() or not absolute_path.exists():
        download_aneuploidy_data(raw_dir)

    # Load aneuploidy scores (tab-separated text from GDC)
    aneuploidy_df = pd.read_csv(aneuploidy_path, sep="\t")

    # Load ABSOLUTE purity/ploidy (tab-separated text from GDC)
    absolute_df = pd.read_csv(absolute_path, sep="\t")

    # Standardize column names — these files use varying conventions
    # Aneuploidy file typically has: Sample, Aneuploidy Score, etc.
    aneuploidy_df.columns = aneuploidy_df.columns.str.strip()
    absolute_df.columns = absolute_df.columns.str.strip()

    # Extract patient barcode from sample IDs
    for df in [aneuploidy_df, absolute_df]:
        sample_col = None
        for candidate in ["Sample", "sample", "SAMPLE", "Name", "name"]:
            if candidate in df.columns:
                sample_col = candidate
                break
        if sample_col is None:
            # Use first column as sample identifier
            sample_col = df.columns[0]
        df["patient_barcode"] = df[sample_col].astype(str).str[:12]

    # Merge on patient barcode
    merged = aneuploidy_df.merge(
        absolute_df, on="patient_barcode", how="outer", suffixes=("", "_abs")
    )

    return merged


if __name__ == "__main__":
    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)

    print("=== Downloading TCGA-CDR ===")
    download_tcga_cdr(raw)

    print("\n=== Downloading cBioPortal clinical data ===")
    download_cbioportal_clinical(raw)

    print("\n=== Downloading aneuploidy data ===")
    download_aneuploidy_data(raw)

    print("\nAll downloads complete.")
