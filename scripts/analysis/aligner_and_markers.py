"""Aligner ablations and marker statistics (Appendix D, J.1, J.4, J.5, J.6, Table A11).

Needs: results/scores/{depth,crossdomain,hosted}.csv and their *_greedy.csv versions
(python scripts/score_generations.py --study depth --aligner greedy, same for crossdomain
and hosted), the rebuilt samples, and the depth-study generations (threshold sweep).
Writes results/analysis/aligner_and_markers.json.

Usage:  python scripts/analysis/aligner_and_markers.py [--skip-sweep]
"""
import argparse
import glob
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from configs.paths import DATA_DIR, OUTPUTS_DIR, RESULTS_DIR  # noqa: E402
from analysis.core_tables import TYPES, load  # noqa: E402

LENGTHS = (3000, 6000, 12000)


def _identity(job):
    from src.efs import compute_efs, compute_efs_greedy
    domain, text = job
    return [(compute_efs_greedy if g else compute_efs)(text, text, domain=domain)["by_type"] for g in (True, False)]


def identity_controls(workers):
    """Source == output at 3k/6k/12k characters on the 192-document samples of each domain."""
    docs = {"clinical": pd.read_parquet(DATA_DIR / "full_sample.parquet").discharge_text,
            "legal": pd.read_parquet(DATA_DIR / "legal_sample.parquet").source_text,
            "scientific": pd.read_parquet(DATA_DIR / "scientific_sample.parquet").source_text}
    jobs = [(d, str(t)[:L]) for L in LENGTHS for d, texts in docs.items() for t in texts]
    with Pool(workers) as p:
        res = p.map(_identity, jobs, chunksize=4)
    v, i = {}, 0
    for L in LENGTHS:
        for d, texts in docs.items():
            greedy = [r[0] for r in res[i:i + len(texts)]]
            hung = [r[1] for r in res[i:i + len(texts)]]
            i += len(texts)
            n_g = [sum(b.values()) for b in greedy]
            v[f"A11.{d}.{L}.greedy_fpr_pct"] = 100 * np.mean([n > 0 for n in n_g])
            v[f"A11.{d}.{L}.greedy_spurious"] = int(sum(n_g))
            v[f"J5.identity.{d}.{L}.hungarian_distortions"] = int(sum(sum(b.values()) for b in hung))
            if d == "clinical" and L == 12000:
                v.update({f"J6.identity_per_doc.{t}": np.mean([b.get(t, 0) for b in greedy]) for t in TYPES})
    v["J5.identity_combinations"] = len(jobs)
    v["J5.identity_all_zero"] = int(all(sum(r[1].values()) == 0 for r in res))
    return v


def _identity_efs(job):
    from src.efs import compute_efs_greedy
    d, t = job
    return compute_efs_greedy(t, t, domain=d)["efs_score"]


def identity_ceiling(workers):
    v = {}
    for d, f, col in (("clinical", "full_sample.parquet", "discharge_text"),
                      ("legal", "legal_sample.parquet", "source_text"),
                      ("scientific", "scientific_sample.parquet", "source_text")):
        texts = [str(t)[:12000] for t in pd.read_parquet(DATA_DIR / f)[col]]
        with Pool(workers) as p:
            e = p.map(_identity_efs, [(d, t) for t in texts])
        v[f"A11.{d}.identity_efs"] = float(np.mean(e))
    return v


def greedy_results():
    v = {}
    g, h = load("depth", "greedy"), load("depth")
    v["J5.greedy_depth_efs"] = g.efs.mean()
    v["J5.greedy_omission_pct"] = 100 * g.omission.sum() / g[TYPES].sum().sum()
    v["J5.hungarian_omission_pct"] = 100 * h.omission.sum() / h[TYPES].sum().sum()
    gm, hm = g.groupby("model").efs.mean(), h.groupby("model").efs.mean()
    v["J5.rank_spearman_open"] = stats.spearmanr(gm, hm[gm.index]).statistic
    v["J5.same_top_bottom"] = int(gm.idxmax() == hm.idxmax() and gm.idxmin() == hm.idxmin())
    g123 = g[g.seed == 123]
    v["A11.clinical.observed_greedy"] = g123.efs.mean()
    xg = load("crossdomain", "greedy")
    for d in ("legal", "scientific"):
        v[f"A11.{d}.observed_greedy"] = xg[xg.domain == d].efs.mean()
    hg, hh = load("hosted", "greedy"), load("hosted")
    a, b = hg.groupby("model").efs.mean(), hh.groupby("model").efs.mean()
    v["J5.rank_spearman_hosted"] = stats.spearmanr(a, b[a.index]).statistic
    # J.6: identity floor as a share of each type's mean count among greedy seed-123 generations with that type
    v["J6.omission_plus_fabrication_pct"] = 100 * (h.omission.sum() + h.fabrication.sum()) / h[TYPES].sum().sum()
    return v, g123


def threshold_sweep(workers):
    """Appendix J.5: 1,000 seed-123 clinical generations drawn with numpy seed 0, tau in [0.20, 0.50]."""
    ep = pd.read_parquet(DATA_DIR / "full_sample.parquet")
    src = {int(h): t[:12000] for h, t in zip(ep.hadm_id, ep.discharge_text)}
    files = sorted(glob.glob(str(OUTPUTS_DIR / "full_generations" / "*" / "*.json")))
    pick = np.random.default_rng(0).choice(len(files), size=1000, replace=False)
    gens = [json.load(open(files[i])) for i in pick]
    jobs = [(tau, src[int(g["hadm_id"])], g["output_text"]) for tau in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50) for g in gens]
    with Pool(workers) as p:
        res = p.map(_sweep_one, jobs, chunksize=8)
    v = {}
    for k, tau in enumerate((0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)):
        chunk = res[k * len(gens):(k + 1) * len(gens)]
        v[f"J5.sweep.{tau:.2f}.efs"] = float(np.mean([c[0] for c in chunk]))
        v[f"J5.sweep.{tau:.2f}.omission_pct"] = 100 * sum(c[1] for c in chunk) / sum(c[2] for c in chunk)
    effs = [v[f"J5.sweep.{t:.2f}.efs"] for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)]
    v["J5.sweep.efs_min"], v["J5.sweep.efs_max"] = min(effs), max(effs)
    return v


def _sweep_one(job):
    from src.efs import compute_efs
    tau, s, o = job
    r = compute_efs(s, o, domain="clinical", threshold=tau)
    return r["efs_score"], r["by_type"].get("omission", 0), sum(r["by_type"].values())


def marker_stats():
    """Appendix D and J.1/J.4 marker composition."""
    from src.distortion_analyzer import get_epistemic_strength
    from src.hedge_screener import find_hedges
    from src.hedge_screener_contextual import find_contextual_hedges
    from src.hedge_screener_scientific import find_scientific_hedges
    v = {}
    notes = {s: [t[:12000] for t in pd.read_parquet(DATA_DIR / f).discharge_text]
             for s, f in ((123, "full_sample.parquet"), (0, "full_sample_seed0.parquet"), (1, "full_sample_seed1.parquet"))}
    counts = {s: [len(find_hedges(t)) for t in ts] for s, ts in notes.items()}
    for s in counts:
        v[f"J4.markers_per_doc.seed{s}"] = float(np.mean(counts[s]))
    v["J4.markers_per_doc.3seeds"] = float(np.mean(sum(counts.values(), [])))
    hed = [m for t in notes[123] for m in find_hedges(t)]
    ctx = [len(find_contextual_hedges(t)) for t in notes[123]]
    v["D.contextual_per_doc"] = float(np.mean(ctx))
    v["D.contextual_reduction_pct"] = 100 * (1 - sum(ctx) / len(hed))
    cat = Counter(m.category for m in hed)
    v["J4.severity_temporal_pct"] = 100 * (cat["severity_modifier"] + cat["temporal_hedge"]) / len(hed)
    v["J1.strength_zero_pct"] = 100 * np.mean([get_epistemic_strength(m.text) == 0 for m in hed])
    sci = [m for t in pd.read_parquet(DATA_DIR / "scientific_sample.parquet").source_text for m in find_scientific_hedges(t[:12000])]
    v["J4.scientific_quantitative_pct"] = 100 * Counter(m.category for m in sci)["quantitative_hedge"] / len(sci)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-sweep", action="store_true", help="skip the threshold sweep (needs the depth generations)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    v = {}
    gv, g123 = greedy_results()
    v.update(gv)
    v.update(identity_controls(a.workers))
    v.update(identity_ceiling(a.workers))
    for t in TYPES:   # J.6 floors, as % of the mean count among greedy seed-123 generations that exhibit the type
        mean_when_present = g123.loc[g123[t] > 0, t].mean()
        v[f"J6.floor_pct.{t}"] = 100 * v[f"J6.identity_per_doc.{t}"] / mean_when_present
    v.update(marker_stats())
    if not a.skip_sweep:
        v.update(threshold_sweep(a.workers))
    out = RESULTS_DIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({k: (float(x) if isinstance(x, (np.floating, np.integer)) else x) for k, x in v.items()},
              open(out / "aligner_and_markers.json", "w"), indent=1)
    print(f"wrote {out / 'aligner_and_markers.json'} ({len(v)} values)")


if __name__ == "__main__":
    main()
