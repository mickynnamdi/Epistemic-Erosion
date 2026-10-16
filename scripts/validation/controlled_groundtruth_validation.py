"""Controlled ground truth: remove a known fraction of each note's hedges and check that EFS tracks it.

Paper: Appendix J.7 / Table A12 (Spearman rho = -0.891, 3 removal seeds, std 0.006; mean EFS
1.000 / 0.830 / 0.644 / 0.443 / 0.249 for 0 / 25 / 50 / 75 / 100% removed) and Appendix J.5
(rho = -0.721, std 0.009, under the greedy baseline aligner).
Inputs: data/full_sample.parquet (MIMIC-IV, PhysioNet credentialed). CPU only.
Outputs: results/controlled_groundtruth/pairs_<aligner>.csv (scores only, no text) and
results/analysis/controlled_groundtruth_<aligner>.json.
Usage:
    python scripts/validation/controlled_groundtruth_validation.py                    # Table A12
    python scripts/validation/controlled_groundtruth_validation.py --aligner greedy   # J.5 comparison
The independent Claude severity judge of Appendix J.7 is scripts/judges/controlled_gt_claude.py.

Protocol: the first 40 seed-123 notes with >= 6 hedges in their first 6,000 characters;
for each note, level f in {0, .25, .5, .75, 1} and removal seed s in {0, 1, 2}, delete
round(f * n) hedge spans chosen with random.Random(note_index + 1000 * s) and score
EFS(source, corrupted source). Spearman(f, EFS) is computed per seed, then averaged.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import DATA_DIR, RESULTS_DIR, require  # noqa: E402
from src.efs import compute_efs, compute_efs_greedy  # noqa: E402
from src.hedge_screener import find_hedges, hedge_density  # noqa: E402

LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def remove_fraction(text, frac, seed):
    """Delete round(frac * n) of the text's hedge spans, chosen at random with `seed`."""
    ms = find_hedges(text)
    if not ms:
        return text
    rng = random.Random(seed)
    k = int(round(frac * len(ms)))
    drop = set(rng.sample(range(len(ms)), k))
    for i in sorted(range(len(ms)), key=lambda i: -ms[i].start):
        if i in drop:
            text = text[:ms[i].start] + text[ms[i].end:]
    return text


def select_notes(n_docs=40, chars=6000, min_hedges=6):
    df = pd.read_parquet(require(DATA_DIR / "full_sample.parquet", "MIMIC seed-123 sample", "EE_DATA_DIR"))
    return [str(t)[:chars] for t in df["discharge_text"] if hedge_density(str(t)[:chars])["total_hedges"] >= min_hedges][:n_docs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligner", default="hungarian", choices=["hungarian", "greedy"])
    ap.add_argument("--n_docs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    score = compute_efs if args.aligner == "hungarian" else compute_efs_greedy

    rows = []
    for s in range(args.seeds):
        for di, src in enumerate(select_notes(args.n_docs)):
            for f in LEVELS:
                out = remove_fraction(src, f, seed=di + 1000 * s)
                rows.append({"seed": s, "doc": di, "frac_removed": f,
                             "efs": score(src, out, domain="clinical")["efs_score"]})
    d = pd.DataFrame(rows)
    per_seed = [float(stats.spearmanr(g["frac_removed"], g["efs"]).correlation) for _, g in d.groupby("seed")]
    levels = d.groupby("frac_removed")["efs"].mean()
    print(f"Spearman rho per seed {[round(r, 3) for r in per_seed]}: mean {np.mean(per_seed):.3f}, "
          f"std {np.std(per_seed, ddof=1):.3f}")
    print("mean EFS by fraction removed:\n" + levels.round(3).to_string())

    tag = "J7" if args.aligner == "hungarian" else "J5.gt_greedy"
    res = {f"{tag}.n_docs": d.doc.nunique(), f"{tag}.rho_mean": float(np.mean(per_seed)),
           f"{tag}.rho_std": float(np.std(per_seed, ddof=1))}
    res.update({f"{tag}.mean_efs.{k:.2f}": float(v) for k, v in levels.items()})
    (RESULTS_DIR / "controlled_groundtruth").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    d.to_csv(RESULTS_DIR / "controlled_groundtruth" / f"pairs_{args.aligner}.csv", index=False)
    json.dump(res, open(RESULTS_DIR / "analysis" / f"controlled_groundtruth_{args.aligner}.json", "w"), indent=1)
    print(f"wrote {RESULTS_DIR / 'analysis' / f'controlled_groundtruth_{args.aligner}.json'}")


if __name__ == "__main__":
    main()
