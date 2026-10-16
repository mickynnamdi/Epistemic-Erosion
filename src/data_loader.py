"""Load and join MIMIC-IV tables into unified hospitalization episodes."""

import gzip
import csv
import pandas as pd
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import (
    DISCHARGE_NOTES, RADIOLOGY_NOTES, ADMISSIONS, SERVICES,
    DIAGNOSES_ICD, D_ICD_DIAGNOSES, PATIENTS, ICUSTAYS, PRESCRIPTIONS,
)


# ---------------------------------------------------------------------------
# Service code → specialty mapping (MIMIC-IV convention)
# ---------------------------------------------------------------------------
SERVICE_TO_SPECIALTY = {
    "MED": "general_medicine",
    "CMED": "cardiology",
    "OMED": "general_medicine",  # observation medicine
    "SURG": "surgery",
    "ORTHO": "surgery",
    "NSURG": "surgery",
    "CSURG": "surgery",
    "VSURG": "surgery",
    "TSURG": "surgery",
    "TRAUM": "surgery",
    "NMED": "neurology",
    "OBS": "obstetrics",
    "GYN": "gynecology",
    "GU": "urology",
    "PSYCH": "psychiatry",
    "ENT": "ent",
    "DENT": "dental",
}

# Target specialties for the study
TARGET_SPECIALTIES = [
    "cardiology",
    "general_medicine",
    "surgery",
    "neurology",
]


def load_csv_gz(path: Path, usecols: Optional[list] = None) -> pd.DataFrame:
    """Load a gzipped CSV into a DataFrame."""
    return pd.read_csv(path, compression="gzip", usecols=usecols, low_memory=False)


def load_discharge_notes() -> pd.DataFrame:
    """Load discharge notes with text."""
    print("Loading discharge notes...")
    df = load_csv_gz(DISCHARGE_NOTES, usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"])
    df["hadm_id"] = df["hadm_id"].astype("Int64")
    print(f"  Loaded {len(df):,} discharge notes")
    return df


def load_radiology_notes() -> pd.DataFrame:
    """Load radiology notes (used as supplementary input for generation tasks)."""
    print("Loading radiology notes...")
    df = load_csv_gz(RADIOLOGY_NOTES, usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"])
    df["hadm_id"] = df["hadm_id"].astype("Int64")
    print(f"  Loaded {len(df):,} radiology notes")
    return df


def load_admissions() -> pd.DataFrame:
    """Load admissions table."""
    print("Loading admissions...")
    df = load_csv_gz(ADMISSIONS, usecols=[
        "subject_id", "hadm_id", "admittime", "dischtime",
        "admission_type", "hospital_expire_flag",
    ])
    df["hadm_id"] = df["hadm_id"].astype("Int64")
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400
    print(f"  Loaded {len(df):,} admissions")
    return df


def load_services() -> pd.DataFrame:
    """Load services table and map to specialty."""
    print("Loading services...")
    df = load_csv_gz(SERVICES, usecols=["subject_id", "hadm_id", "transfertime", "curr_service"])
    df["hadm_id"] = df["hadm_id"].astype("Int64")
    df["specialty"] = df["curr_service"].map(SERVICE_TO_SPECIALTY).fillna("other")
    # Take the first (admitting) service per admission
    df = df.sort_values("transfertime").drop_duplicates(subset=["hadm_id"], keep="first")
    print(f"  Loaded services for {len(df):,} admissions")
    return df[["hadm_id", "curr_service", "specialty"]]


def load_diagnoses() -> pd.DataFrame:
    """Load diagnoses with ICD descriptions. Returns primary diagnosis per admission."""
    print("Loading diagnoses...")
    diag = load_csv_gz(DIAGNOSES_ICD, usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
    d_icd = load_csv_gz(D_ICD_DIAGNOSES, usecols=["icd_code", "icd_version", "long_title"])
    diag["hadm_id"] = diag["hadm_id"].astype("Int64")

    # Merge to get diagnosis names
    diag = diag.merge(d_icd, on=["icd_code", "icd_version"], how="left")

    # Primary diagnosis (seq_num == 1)
    primary = diag[diag["seq_num"] == 1][["hadm_id", "icd_code", "long_title"]].rename(
        columns={"icd_code": "primary_icd", "long_title": "primary_diagnosis"}
    )

    # Count total diagnoses per admission (proxy for complexity)
    diag_counts = diag.groupby("hadm_id").size().reset_index(name="n_diagnoses")

    result = primary.merge(diag_counts, on="hadm_id", how="left")
    print(f"  Loaded diagnoses for {len(result):,} admissions")
    return result


def load_icu_stays() -> pd.DataFrame:
    """Load ICU stays — presence indicates higher complexity."""
    print("Loading ICU stays...")
    df = load_csv_gz(ICUSTAYS, usecols=["subject_id", "hadm_id", "stay_id", "los"])
    df["hadm_id"] = df["hadm_id"].astype("Int64")
    icu_flag = df.groupby("hadm_id").agg(
        icu_stay=("stay_id", "count"),
        icu_los=("los", "sum"),
    ).reset_index()
    icu_flag["has_icu"] = True
    print(f"  Found {len(icu_flag):,} admissions with ICU stays")
    return icu_flag


def build_episode_table() -> pd.DataFrame:
    """
    Build a unified table of hospitalization episodes by joining:
    discharge notes + admissions + services + diagnoses + ICU flags.

    Returns one row per hospitalization episode (hadm_id).
    """
    notes = load_discharge_notes()
    admissions = load_admissions()
    services = load_services()
    diagnoses = load_diagnoses()
    icu = load_icu_stays()

    # Start with discharge notes (these are the source texts)
    # Some hadm_ids have multiple discharge note segments — concatenate
    episode_notes = (
        notes.sort_values(["hadm_id", "charttime"])
        .groupby("hadm_id")
        .agg(
            subject_id=("subject_id", "first"),
            discharge_text=("text", lambda x: "\n\n".join(x)),
            n_note_segments=("note_id", "count"),
        )
        .reset_index()
    )

    # Join everything
    episodes = episode_notes.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    episodes = episodes.merge(services, on="hadm_id", how="left")
    episodes = episodes.merge(diagnoses, on="hadm_id", how="left")
    episodes = episodes.merge(icu, on="hadm_id", how="left")
    episodes["has_icu"] = episodes["has_icu"].fillna(False)
    episodes["icu_stay"] = episodes["icu_stay"].fillna(0).astype(int)

    # Compute note length
    episodes["note_length"] = episodes["discharge_text"].str.len()

    print(f"\nBuilt episode table: {len(episodes):,} episodes")
    print(f"  Specialties: {episodes['specialty'].value_counts().to_dict()}")
    print(f"  Has ICU: {episodes['has_icu'].sum():,}")
    print(f"  Mean diagnoses: {episodes['n_diagnoses'].mean():.1f}")
    print(f"  Mean LOS: {episodes['los_days'].mean():.1f} days")

    return episodes


if __name__ == "__main__":
    episodes = build_episode_table()
    print(f"\nEpisode table shape: {episodes.shape}")
    print(episodes.head())
