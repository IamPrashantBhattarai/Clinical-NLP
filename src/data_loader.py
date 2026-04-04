"""
data_loader.py
Purpose: Load and merge MIMIC-IV (or synthetic) data tables for clinical NLP analysis.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_discharge_notes(
    filepath: str,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load discharge summaries CSV.

    Parameters
    ----------
    filepath : str
        Path to discharge notes CSV (MIMIC or synthetic).
    sample_size : int, optional
        If provided, randomly sample this many rows for development.
    random_seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        Filtered to 'Discharge summary' note_type with clean column names.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Notes file not found: {path.resolve()}")

    logger.info("Loading discharge notes from %s ...", path)
    df = pd.read_csv(filepath, low_memory=False)

    required = {"note_id", "subject_id", "hadm_id", "note_type", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in notes file: {missing}")

    df = df[df["note_type"] == "Discharge summary"].copy()
    logger.info("Discharge summaries found: %d", len(df))

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
        logger.info("Sampled %d notes (seed=%d)", sample_size, random_seed)

    for col in ["charttime", "storetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df.reset_index(drop=True)


def load_admissions(filepath: str) -> pd.DataFrame:
    """
    Load admissions table.

    Parameters
    ----------
    filepath : str
        Path to admissions CSV.

    Returns
    -------
    pd.DataFrame
        Admissions with parsed datetime columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Admissions file not found: {path.resolve()}")

    logger.info("Loading admissions from %s ...", path)
    df = pd.read_csv(filepath, low_memory=False)

    for col in ["admittime", "dischtime", "edregtime", "edouttime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    logger.info("Admissions loaded: %d rows", len(df))
    return df


def load_patients(filepath: str) -> pd.DataFrame:
    """
    Load patients table.

    Parameters
    ----------
    filepath : str
        Path to patients CSV.

    Returns
    -------
    pd.DataFrame
        Patients table.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Patients file not found: {path.resolve()}")

    logger.info("Loading patients from %s ...", path)
    df = pd.read_csv(filepath, low_memory=False)
    logger.info("Patients loaded: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Label creation
# ---------------------------------------------------------------------------

def create_readmission_label(
    admissions_df: pd.DataFrame,
    window_days: int = 30,
) -> pd.DataFrame:
    """
    Add a binary 30-day readmission label to the admissions DataFrame.

    Rules:
    - Exclude admissions where the patient died (hospital_expire_flag == 1).
    - Exclude the last admission for each patient (no follow-up available).
    - Label = 1 if the same patient was readmitted within `window_days` of discharge.

    Parameters
    ----------
    admissions_df : pd.DataFrame
        Admissions table with at least: subject_id, hadm_id, admittime,
        dischtime, hospital_expire_flag.
    window_days : int
        Readmission window in days (default 30).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new column 'readmission_30day':
            1  = readmitted within window
            0  = not readmitted
           -1  = excluded (patient died or last admission)
    """
    df = admissions_df.copy()
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    df = df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    if "readmission_30day" in df.columns:
        logger.info("readmission_30day column already present — skipping label creation.")
        return df

    flags = []
    for _, grp in df.groupby("subject_id", sort=False):
        grp = grp.reset_index(drop=True)
        group_flags = []
        for i in range(len(grp)):
            if grp.loc[i, "hospital_expire_flag"] == 1:
                group_flags.append(-1)
            elif i == len(grp) - 1:
                group_flags.append(-1)
            else:
                days_gap = (grp.loc[i + 1, "admittime"] - grp.loc[i, "dischtime"]).days
                group_flags.append(1 if days_gap <= window_days else 0)
        flags.extend(group_flags)

    df["readmission_30day"] = flags
    eligible = df[df["readmission_30day"] >= 0]
    rate = eligible["readmission_30day"].mean() * 100
    logger.info(
        "Readmission label created — eligible: %d, readmit rate: %.1f%%",
        len(eligible), rate,
    )
    return df


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_dataset(
    notes_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    patients_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge notes, admissions, and patients into one analysis-ready DataFrame.

    Derived columns added:
    - 'age_group'  : '<40', '40-65', '65+'
    - 'los_days'   : length of stay in days

    Parameters
    ----------
    notes_df : pd.DataFrame
    admissions_df : pd.DataFrame
    patients_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Merged DataFrame, one row per note.
    """
    logger.info("Merging notes + admissions + patients ...")

    df = notes_df.merge(admissions_df, on=["subject_id", "hadm_id"], how="inner")
    df = df.merge(patients_df, on="subject_id", how="left")

    # Length of stay
    adm = pd.to_datetime(df["admittime"])
    dis = pd.to_datetime(df["dischtime"])
    df["los_days"] = (dis - adm).dt.total_seconds() / 86400
    df["los_days"] = df["los_days"].clip(lower=0).round(2)

    # Age group
    age_col = "anchor_age" if "anchor_age" in df.columns else None
    if age_col:
        df["age_group"] = pd.cut(
            df[age_col],
            bins=[0, 39, 64, 120],
            labels=["<40", "40-65", "65+"],
        )
    else:
        logger.warning("anchor_age column not found — age_group not created.")

    logger.info("Merged dataset shape: %s", df.shape)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def get_data_summary(merged_df: pd.DataFrame) -> dict:
    """
    Print and return summary statistics of the merged dataset.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Output of merge_dataset().

    Returns
    -------
    dict
        Summary statistics dictionary.
    """
    df = merged_df.copy()

    total_notes = len(df)
    unique_patients = df["subject_id"].nunique()
    unique_admissions = df["hadm_id"].nunique() if "hadm_id" in df.columns else None

    eligible = df[df["readmission_30day"] >= 0] if "readmission_30day" in df.columns else df
    readmit_rate = (
        eligible["readmission_30day"].mean() * 100
        if "readmission_30day" in df.columns else None
    )

    avg_note_words = df["text"].dropna().apply(lambda t: len(str(t).split())).mean()

    summary = {
        "total_notes": total_notes,
        "unique_patients": unique_patients,
        "unique_admissions": unique_admissions,
        "readmission_rate_pct": round(readmit_rate, 2) if readmit_rate is not None else None,
        "avg_note_length_words": round(avg_note_words, 1),
    }

    if "gender" in df.columns:
        summary["gender_distribution"] = df["gender"].value_counts().to_dict()

    if "anchor_age" in df.columns:
        summary["age_stats"] = {
            "mean": round(df["anchor_age"].mean(), 1),
            "median": round(df["anchor_age"].median(), 1),
            "min": int(df["anchor_age"].min()),
            "max": int(df["anchor_age"].max()),
        }

    if "age_group" in df.columns:
        summary["age_group_distribution"] = df["age_group"].value_counts().to_dict()

    if "insurance" in df.columns:
        summary["insurance_distribution"] = df["insurance"].value_counts().to_dict()

    # Print report
    separator = "-" * 50
    print(separator)
    print("DATASET SUMMARY")
    print(separator)
    print(f"  Total notes          : {summary['total_notes']:,}")
    print(f"  Unique patients      : {summary['unique_patients']:,}")
    print(f"  Unique admissions    : {summary['unique_admissions']:,}")
    if readmit_rate is not None:
        print(f"  Readmission rate     : {summary['readmission_rate_pct']:.1f}%")
    print(f"  Avg note length      : {summary['avg_note_length_words']:.0f} words")

    if "gender_distribution" in summary:
        print(f"\n  Gender distribution  : {summary['gender_distribution']}")

    if "age_stats" in summary:
        s = summary["age_stats"]
        print(f"  Age (mean/median)    : {s['mean']} / {s['median']} yrs  [{s['min']}–{s['max']}]")

    if "age_group_distribution" in summary:
        print(f"  Age groups           : {summary['age_group_distribution']}")

    if "insurance_distribution" in summary:
        print(f"  Insurance types      : {summary['insurance_distribution']}")

    print(separator)
    return summary


# ---------------------------------------------------------------------------
# Convenience loader (uses config paths)
# ---------------------------------------------------------------------------

def load_all(config: dict, use_synthetic: bool = True) -> pd.DataFrame:
    """
    One-call loader: reads all three tables, creates labels, merges, returns
    a ready-to-use DataFrame.

    Parameters
    ----------
    config : dict
        Loaded config.yaml dict (top-level).
    use_synthetic : bool
        If True, uses synthetic_* paths; otherwise uses mimic_* paths.

    Returns
    -------
    pd.DataFrame
        Merged, labelled dataset.
    """
    data_cfg = config["data"]
    prefix = "synthetic" if use_synthetic else "mimic"

    notes_df = load_discharge_notes(
        filepath=data_cfg[f"{prefix}_notes_path"],
        sample_size=data_cfg.get("sample_size") if not use_synthetic else None,
        random_seed=data_cfg.get("random_seed", 42),
    )
    admissions_df = load_admissions(data_cfg[f"{prefix}_admissions_path"])
    patients_df = load_patients(data_cfg[f"{prefix}_patients_path"])

    # Real MIMIC data won't have the label pre-computed
    if "readmission_30day" not in admissions_df.columns:
        admissions_df = create_readmission_label(admissions_df)

    merged = merge_dataset(notes_df, admissions_df, patients_df)
    return merged
