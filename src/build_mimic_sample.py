"""Rebuild the clinical (MIMIC-IV) episode samples under your own PhysioNet DUA.

It reproduces the
three stratified episode draws used in the paper's depth study from a local
MIMIC-IV download; the resulting parquet files are the inputs that
scripts/generate/generate.py, scripts/score_generations.py, scripts/probe/certainty_prior_probe.py
expect. Nothing here downloads data:
you must hold PhysioNet credentials and have accepted the MIMIC-IV and
MIMIC-IV-Note data use agreements (https://physionet.org/content/mimiciv/,
https://physionet.org/content/mimic-iv-note/).

Layout expected under $EE_MIMIC_ROOT (see configs/paths.py):
    mimic-iv-note/2.2/note/discharge.csv.gz
    mimiciv/3.1/hosp/{admissions,services,diagnoses_icd,d_icd_diagnoses}.csv.gz
    mimiciv/3.1/icu/icustays.csv.gz

Two sampling protocols:

  original (seed 123 -> data/full_sample.parquet, 192 episodes)
      pool: per target specialty (cardiology, general_medicine, surgery,
      neurology), episodes with 3,000 <= note length <= 20,000 chars, 2,000
      drawn per specialty with random_state=42; hedge statistics on the full
      note; uncertainty level from src.hedge_screener.classify_uncertainty_density;
      complexity = ICU stay or >= 10 diagnoses or LOS >= 7 days; stratified
      draw of 200 (-> 192 after integer stratum targets: 4 specialties x
      {simple, complex} x {low 10%, medium 15%, high 25%}) with seed 123.
  seeded (seeds 0 and 1 -> data/full_sample_seed{N}.parquet, 184 episodes each)
      same pool construction but random_state=seed; hedge statistics on the
      first 12,000 chars; uncertainty level = high if diagnostic hedges >= 5,
      medium if >= 2, else low; stratified draw of 192 (-> 184) with the same
      seed; episodes already in data/full_sample.parquet are excluded.

Note: uncertainty density enters through the stratification only; no minimum
number of diagnostic hedges is required (the seed-123 draw contains episodes
with 0 and 1 diagnostic hedges).

Usage:
    export EE_MIMIC_ROOT=/path/to/physionet/files
    python -m src.build_mimic_sample --seed 123 --out data/full_sample.parquet
    python -m src.build_mimic_sample --seed 0   --out data/full_sample_seed0.parquet
    python -m src.build_mimic_sample --seed 1   --out data/full_sample_seed1.parquet

Loading the full discharge-note table needs ~16 GB RAM and a few minutes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import DATA_DIR, DISCHARGE_NOTES, MIMIC_ROOT  # noqa: E402
from src.data_loader import TARGET_SPECIALTIES, build_episode_table  # noqa: E402
from src.hedge_screener import hedge_density  # noqa: E402
from src.sampler import compute_complexity, run_sampling_pipeline, stratified_sample  # noqa: E402

TRUNC = 12000


def _check_layout() -> None:
    if not DISCHARGE_NOTES.exists():
        raise FileNotFoundError(
            f"{DISCHARGE_NOTES} not found. Set EE_MIMIC_ROOT to your PhysioNet download "
            f"(currently {MIMIC_ROOT}); see the module docstring for the expected layout.")


def _hedge_stats_seeded(pool: pd.DataFrame) -> pd.DataFrame:
    """Per-episode hedge statistics used by the seeded draws (first 12,000 characters)."""
    pool = pool.copy()
    dens, diag = [], []
    for text in pool["discharge_text"]:
        stats = hedge_density(str(text)[:TRUNC])
        dens.append(stats["hedges_per_1k_chars"])
        diag.append(stats["diagnostic_hedges"])
    pool["hedge_density"] = dens
    pool["diagnostic_hedge_count"] = diag
    pool["uncertainty_level"] = pool["diagnostic_hedge_count"].apply(
        lambda x: "high" if x >= 5 else ("medium" if x >= 2 else "low"))
    return pool


def build_sample(out_path: Path, seed: int = 123, protocol: Optional[str] = None,
                 n_total: int = 192, screen_per_specialty: int = 2000,
                 exclude_from: Optional[Path] = None) -> pd.DataFrame:
    """Draw one stratified MIMIC-IV episode sample and write it to `out_path`.

    protocol: "original" (default when seed == 123) or "seeded" (default otherwise).
    Returns the sampled DataFrame (one row per hadm_id, with `discharge_text`).
    """
    _check_layout()
    out_path = Path(out_path)
    protocol = protocol or ("original" if seed == 123 else "seeded")

    if protocol == "original":
        if seed != 123:
            raise ValueError("the 'original' protocol is defined for seed 123 only")
        full = run_sampling_pipeline(n_pilot=30, n_full=200, save=False,
                                     screen_per_specialty=screen_per_specialty)["full"]
    elif protocol == "seeded":
        episodes = build_episode_table()
        episodes["complexity"] = compute_complexity(episodes)
        parts = []
        for spec in TARGET_SPECIALTIES:
            spec_df = episodes[(episodes["specialty"] == spec)
                               & (episodes["note_length"] >= 3000)
                               & (episodes["note_length"] <= 20000)]
            n_draw = min(screen_per_specialty, len(spec_df))
            parts.append(spec_df.sample(n=n_draw, random_state=seed))
            print(f"  pre-sampled {n_draw} of {len(spec_df)} {spec} episodes")
        pool = _hedge_stats_seeded(pd.concat(parts, ignore_index=True))
        full = stratified_sample(pool, n_total=n_total, seed=seed)
        excl = exclude_from if exclude_from is not None else DATA_DIR / "full_sample.parquet"
        if Path(excl).exists():
            orig = pd.read_parquet(excl, columns=["hadm_id"])
            overlap = set(full["hadm_id"]) & set(orig["hadm_id"])
            if overlap:
                print(f"  removing {len(overlap)} episodes that overlap {excl}")
                full = full[~full["hadm_id"].isin(overlap)]
    else:
        raise ValueError(protocol)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    print(f"wrote {len(full)} episodes -> {out_path}")
    return full


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: data/full_sample.parquet for seed 123, data/full_sample_seed<N>.parquet otherwise")
    ap.add_argument("--protocol", choices=["original", "seeded"], default=None)
    ap.add_argument("--n-total", type=int, default=192, help="stratified target for the seeded protocol")
    ap.add_argument("--screen-per-specialty", type=int, default=2000)
    a = ap.parse_args()
    out = a.out or (DATA_DIR / ("full_sample.parquet" if a.seed == 123 else f"full_sample_seed{a.seed}.parquet"))
    build_sample(out, seed=a.seed, protocol=a.protocol, n_total=a.n_total,
                 screen_per_specialty=a.screen_per_specialty)


if __name__ == "__main__":
    main()
