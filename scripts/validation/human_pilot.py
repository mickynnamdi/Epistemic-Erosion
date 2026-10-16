"""Inter-rater agreement and EFS-vs-severity correlation of the 20-pair human pilot.

Paper: Appendix K. Severity: mean pairwise quadratic-weighted Cohen's kappa 0.430 (rater pairs 0.554,
0.543, 0.194), all three raters within +-1 level on 70% of pairs. Per-type Fleiss' kappa
(fabrication 0.603, positives 13/14/15 of 20; omission positives range 19/20 vs 6/20). Any
distortion present: raw pairwise agreement 93%, Gwet's AC1 0.926. EFS vs 3-rater mean severity:
Pearson r -0.034 (bootstrap 95% CI [-0.523, 0.457]), Spearman rho -0.070 ([-0.554, 0.440]).
Inputs: three rater JSON exports (--raters) and the legal/scientific annotation sheet (--sheet);
access: withheld (rater work product); CPU only.
Outputs: RESULTS_DIR/analysis/human_pilot.json (check ids K.*).
Usage:
    python scripts/validation/human_pilot.py --raters r1.json r2.json r3.json \\
        --sheet path/annotation_sheet_legal_scientific.csv

The app served a stratified random subsample of the sheet renumbered 1..20
(scripts/validation/annotation_sampling.py); annotation id k is joined to that sample's k-th pair,
and each rater's (domain, model, task) is checked against it. EFS uses the released one-to-one
aligner on the source cut to 12,000 characters, with the sheet's domain. Pairwise labels follow
the --raters order (pairs 1-2, 1-3, 2-3), as do the per-rater positive counts.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from configs.paths import RESULTS_DIR  # noqa: E402
from validation.annotation_sampling import DEFAULT_SHEET, read_sheet, sample_pairs  # noqa: E402

SEV = {"none": 0, "low": 1, "moderate": 2, "high": 3}
TYPES = ["escalation", "attenuation", "omission", "fabrication", "scope_shift", "severity_warp"]
N_BOOT, BOOT_SEED = 10000, 42


def fleiss(M):
    """Fleiss' kappa; M is items x categories counts with a constant number of raters per item."""
    N, nr = M.shape[0], M.sum(1)[0]
    P = ((M ** 2).sum(1) - nr) / (nr * (nr - 1))
    pj = M.sum(0) / (N * nr)
    Pe = (pj ** 2).sum()
    return float((P.mean() - Pe) / (1 - Pe))


def ac1(a, b):
    po = np.mean(a == b)
    pi = (a.mean() + b.mean()) / 2
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe)


def boot_ci(x, y, fn, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap over pairs; resamples with a constant variable are skipped."""
    rng = np.random.default_rng(seed)
    n, v = len(x), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xs, ys = x[idx], y[idx]
        if np.std(xs) > 0 and np.std(ys) > 0:
            v.append(fn(xs, ys))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raters", nargs=3, type=Path, required=True, metavar="RATER_JSON")
    ap.add_argument("--rater-labels", default="R1,R2,R3",
                    help="comma-separated labels for the raters, in --raters order (default: %(default)s)")
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET,
                    help="legal/scientific annotation sheet CSV (default: %(default)s)")
    args = ap.parse_args()
    R = args.rater_labels.split(",")
    assert len(R) == 3, "--rater-labels needs three labels"
    if not args.sheet.exists():
        raise SystemExit(f"[missing input] annotation sheet not found at {args.sheet} (pass --sheet)")
    from src.efs import compute_efs

    ann = {r: {int(a["id"]): a for a in json.load(open(p))["annotations"]} for r, p in zip(R, args.raters)}
    served = sample_pairs(read_sheet(args.sheet))
    for r in R:  # every annotation must describe the pair it is joined to
        assert all((ann[r][p["id"]]["domain"], ann[r][p["id"]]["model"], ann[r][p["id"]]["task"])
                   == (p["domain"], p["model"], p["task"]) for p in served), f"rater {r}: join mismatch"
    ids = [p["id"] for p in served]
    n = len(ids)

    sev = pd.DataFrame({r: [SEV.get(ann[r][i].get("overall_distortion_severity", ""), np.nan) for i in ids]
                        for r in R}, index=ids)
    efs = pd.Series({p["id"]: compute_efs(p["source_text"][:12000], p["output_text"], domain=p["domain"])["efs_score"]
                     for p in served})
    sm = sev.mean(axis=1)
    x, y = efs[ids].values, sm[ids].values

    v = {"K.n_pairs": n, "K.efs_mean": float(efs.mean())}
    pr, sr = stats.pearsonr(x, y), stats.spearmanr(x, y)
    v["K.pearson_r"] = float(pr[0])
    v["K.pearson_p"] = float(pr[1])
    v["K.pearson_boot_lo"], v["K.pearson_boot_hi"] = boot_ci(x, y, lambda a, b: stats.pearsonr(a, b)[0])
    v["K.spearman_rho"] = float(sr.correlation)
    v["K.spearman_p"] = float(sr.pvalue)
    v["K.spearman_boot_lo"], v["K.spearman_boot_hi"] = boot_ci(x, y, lambda a, b: stats.spearmanr(a, b).correlation)

    qk = {f"{a}-{b}": float(cohen_kappa_score(sev[a], sev[b], weights="quadratic"))
          for a, b in itertools.combinations(R, 2)}
    v.update({f"K.severity_qwk.{k}": x_ for k, x_ in qk.items()})
    v["K.severity_qwk.mean"] = float(np.mean(list(qk.values())))
    v["K.severity_within1_pct"] = float(100 * np.mean([(sev.loc[i].max() - sev.loc[i].min()) <= 1 for i in ids]))

    def lab(r, i, t):
        return int(str(ann[r][i][f"has_{t}"]).strip().lower() == "yes")

    for t in TYPES:
        L = np.array([[lab(r, i, t) for r in R] for i in ids])
        v[f"K.fleiss.{t}"] = fleiss(np.stack([L.sum(1), 3 - L.sum(1)], 1))
        for r, pos in zip(R, L.sum(0)):
            v[f"K.positives.{t}.{r}"] = int(pos)

    anyd = np.array([[int(any(lab(r, i, t) for t in TYPES)) for r in R] for i in ids])
    pairs = list(itertools.combinations(range(3), 2))
    v["K.any_distortion.raw_pairwise"] = float(np.mean([np.mean(anyd[:, a] == anyd[:, b]) for a, b in pairs]))
    v["K.any_distortion.ac1_mean_pairwise"] = float(np.mean([ac1(anyd[:, a], anyd[:, b]) for a, b in pairs]))

    out = RESULTS_DIR / "analysis" / "human_pilot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(v, open(out, "w"), indent=2)
    for k, x_ in v.items():
        print(f"{k:45s} {x_:.4f}" if isinstance(x_, float) else f"{k:45s} {x_}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
