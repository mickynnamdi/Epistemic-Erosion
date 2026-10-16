"""
Stratified sampling of MIMIC-IV hospitalization episodes.

Sampling strata:
  - Specialty: cardiology, general_medicine, surgery, neurology
  - Complexity: simple (short stay, few dx) vs complex (ICU, many dx)
  - Uncertainty density: low vs medium vs high (from hedge screener)

Defaults: a 30-episode pilot sample and a 200-episode full sample.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import build_episode_table, TARGET_SPECIALTIES
from src.hedge_screener import classify_uncertainty_density, hedge_density
from configs.paths import DATA_DIR


def compute_complexity(episodes: pd.DataFrame) -> pd.Series:
    """
    Classify each episode as 'simple' or 'complex'.

    Complex if any of:
      - Has ICU stay
      - >= 10 diagnoses
      - LOS >= 7 days
    """
    is_complex = (
        episodes["has_icu"] |
        (episodes["n_diagnoses"] >= 10) |
        (episodes["los_days"] >= 7)
    )
    return is_complex.map({True: "complex", False: "simple"})


def compute_hedge_stats(episodes: pd.DataFrame) -> pd.DataFrame:
    """Add hedge density columns to episodes DataFrame."""
    print("Computing hedge density for all episodes...")
    densities = []
    classifications = []
    diag_hedge_counts = []

    for text in tqdm(episodes["discharge_text"], desc="Screening hedges"):
        stats = hedge_density(text)
        densities.append(stats["hedges_per_1k_chars"])
        diag_hedge_counts.append(stats["diagnostic_hedges"])
        classifications.append(classify_uncertainty_density(text))

    episodes = episodes.copy()
    episodes["hedge_density"] = densities
    episodes["diagnostic_hedge_count"] = diag_hedge_counts
    episodes["uncertainty_level"] = classifications
    return episodes


def stratified_sample(
    episodes: pd.DataFrame,
    n_total: int = 200,
    specialties: list = TARGET_SPECIALTIES,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Draw a stratified sample across specialty × complexity × uncertainty.

    Strategy:
      - Equal allocation across 4 specialties (50 each for n=200)
      - Within each specialty, balance simple/complex and low/med/high uncertainty
      - Oversample high-uncertainty notes
    """
    rng = np.random.RandomState(seed)

    # Filter to target specialties
    pool = episodes[episodes["specialty"].isin(specialties)].copy()
    print(f"\nSampling pool: {len(pool):,} episodes across {specialties}")

    per_specialty = n_total // len(specialties)
    sampled = []

    for spec in specialties:
        spec_pool = pool[pool["specialty"] == spec]
        print(f"\n  {spec}: {len(spec_pool):,} available")

        # Target distribution within specialty:
        #   complexity: 50/50 simple/complex
        #   uncertainty: 20% low, 30% medium, 50% high (oversample high)
        targets = {
            ("simple", "low"): int(per_specialty * 0.10),
            ("simple", "medium"): int(per_specialty * 0.15),
            ("simple", "high"): int(per_specialty * 0.25),
            ("complex", "low"): int(per_specialty * 0.10),
            ("complex", "medium"): int(per_specialty * 0.15),
            ("complex", "high"): int(per_specialty * 0.25),
        }

        for (comp, unc), target_n in targets.items():
            stratum = spec_pool[
                (spec_pool["complexity"] == comp) &
                (spec_pool["uncertainty_level"] == unc)
            ]
            n_draw = min(target_n, len(stratum))
            if n_draw > 0:
                drawn = stratum.sample(n=n_draw, random_state=rng)
                sampled.append(drawn)
                print(f"    {comp}/{unc}: drew {n_draw} from {len(stratum)}")
            else:
                print(f"    {comp}/{unc}: EMPTY stratum (target {target_n})")

    result = pd.concat(sampled, ignore_index=True)
    print(f"\nTotal sampled: {len(result)} episodes")
    return result


def create_pilot_sample(episodes: pd.DataFrame, n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Draw a smaller pilot sample (30 episodes) for annotation calibration."""
    return stratified_sample(episodes, n_total=n, seed=seed)


def run_sampling_pipeline(
    n_pilot: int = 30,
    n_full: int = 200,
    save: bool = True,
    screen_per_specialty: int = 2000,
) -> dict:
    """
    Full pipeline: load data → compute features → sample → save.

    Args:
        screen_per_specialty: How many episodes per specialty to screen for
            hedge density (2000 per specialty = 8000 in total, about 5 min).

    Returns dict with 'episodes', 'pilot', 'full' DataFrames.
    """
    # Step 1: Build episode table
    episodes = build_episode_table()

    # Step 2: Add complexity labels
    episodes["complexity"] = compute_complexity(episodes)

    # Step 3: Pre-sample a manageable pool per specialty, then screen hedges
    target_specs = TARGET_SPECIALTIES
    pool_parts = []
    for spec in target_specs:
        spec_df = episodes[
            (episodes["specialty"] == spec) &
            (episodes["note_length"] >= 3000) &
            (episodes["note_length"] <= 20000)
        ]
        n_draw = min(screen_per_specialty, len(spec_df))
        pool_parts.append(spec_df.sample(n=n_draw, random_state=42))
        print(f"  Pre-sampled {n_draw} from {len(spec_df)} {spec} episodes")

    pool = pd.concat(pool_parts, ignore_index=True)
    print(f"\nScreening pool: {len(pool):,} episodes")

    # Step 3b: Add hedge density on the screening pool
    pool = compute_hedge_stats(pool)

    # Step 4: Draw samples from the screened pool
    pilot = create_pilot_sample(pool, n=n_pilot)
    full = stratified_sample(pool, n_total=n_full, seed=123)

    # Step 5: Save
    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save screened pool metadata (no text — too large)
        meta_cols = [c for c in pool.columns if c != "discharge_text"]
        pool[meta_cols].to_csv(DATA_DIR / "screened_pool_metadata.csv", index=False)
        print(f"\nSaved screened pool metadata to {DATA_DIR / 'screened_pool_metadata.csv'}")

        # Save pilot and full samples (with text)
        pilot.to_parquet(DATA_DIR / "pilot_sample.parquet", index=False)
        full.to_parquet(DATA_DIR / "full_sample.parquet", index=False)
        print(f"Saved pilot sample ({len(pilot)} episodes) to {DATA_DIR / 'pilot_sample.parquet'}")
        print(f"Saved full sample ({len(full)} episodes) to {DATA_DIR / 'full_sample.parquet'}")

    # Print summary stats
    print("\n" + "=" * 60)
    print("SAMPLING SUMMARY")
    print("=" * 60)
    for name, df in [("Pilot", pilot), ("Full", full)]:
        print(f"\n{name} sample ({len(df)} episodes):")
        print(f"  Specialty distribution:\n{df['specialty'].value_counts().to_string()}")
        print(f"  Complexity distribution:\n{df['complexity'].value_counts().to_string()}")
        print(f"  Uncertainty distribution:\n{df['uncertainty_level'].value_counts().to_string()}")
        print(f"  Mean note length: {df['note_length'].mean():,.0f} chars")
        print(f"  Mean hedge density: {df['hedge_density'].mean():.2f} per 1k chars")

    return {"episodes": episodes, "pool": pool, "pilot": pilot, "full": full}


if __name__ == "__main__":
    results = run_sampling_pipeline()
