"""EFS recomputed with the marker set restricted to certainty-bearing lexicon categories.

Paper: Appendix J.4. Dropping the clinical severity_modifier and temporal_hedge categories and the
scientific quantitative_hedge category (legal: nothing) lowers the mean clinical marker count from
20.5 to 6.2 per document and clinical EFS from 0.450 to 0.366 (three-seed depth study). Model ranking
Spearman 0.873 (p < 1e-3) against the unrestricted metric, BioMistral-7B highest and MedGemma-27B
lowest; matched three models: clinical 0.456, scientific 0.415, legal 0.359, Kruskal-Wallis
H = 44.8, p = 1.9e-10; omission 60.3% and escalation 21.3% (from 11.7%) of distortions; maximum
per-model cross-seed standard deviation 0.020 -> 0.067; unrestricted spans per document 15.8
(seed 123), 23.5 (seed 0), 22.4 (seed 1).
Inputs: EE_DATA_DIR/{full_sample,full_sample_seed0,full_sample_seed1,legal_sample,scientific_sample}.parquet,
        the depth and cross-domain generations in EE_OUTPUTS_DIR (configs/studies.py), and
        results/scores/{depth,crossdomain}.csv (scripts/score_generations.py; they list the scored
        generations and give the unrestricted EFS); access: PhysioNet credentialed (MIMIC-IV notes). CPU only.
Outputs: results/restricted_markers/{clinical,legal,scientific}.csv (per generation),
         results/analysis/restricted_markers.json.
Usage:
    python scripts/analysis/restricted_markers.py [--workers 16]
    python scripts/analysis/restricted_markers.py --stage analyze    # reuse the per-generation CSVs

The restriction is applied coherently: the source marker count |H(S)| (EFS denominator), the claim
extractor (which admits only sentences containing a marker) and each domain's hedge finder all see
the restricted marker set. It is implemented by `restricted_markers()`, a context manager that
temporarily wraps the released hedge finders; src/ is not modified.
"""
import argparse
import json
import sys
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.models import STUDY_11  # noqa: E402
from configs.paths import OUTPUTS_DIR, RESULTS_DIR, require  # noqa: E402
from configs.studies import STUDIES, sample_path  # noqa: E402

DROP = {"clinical": {"severity_modifier", "temporal_hedge"},
        "scientific": {"quantitative_hedge"},
        "legal": set()}
TYPES = ["omission", "escalation", "attenuation", "fabrication", "scope_shift", "severity_warp"]
MAX_SOURCE_CHARS = 12000
OUT_DIR = RESULTS_DIR / "restricted_markers"


def _without(find, drop):
    """A hedge finder that returns `find`'s matches minus those in the dropped categories."""
    def find_restricted(text):
        return [h for h in find(text) if getattr(h, "category", None) not in drop]
    return find_restricted


def _count(find):
    def density(text):   # the cross-domain scorer reads only total_hedges
        return {"total_hedges": len(find(text))}
    return density


@contextmanager
def restricted_markers(drop=DROP):
    """Within the block, every EFS computation sees only markers outside the `drop` categories.

    Wraps the clinical hedge finder where the clinical analyzers look it up (the greedy module, whose
    claim extractor the one-to-one scorer reuses, and the one-to-one module's |H(S)|), and the
    hedge_fn / density_fn of every domain in the cross-domain configuration. Restores them on exit.
    """
    import src.distortion_analyzer as da
    import src.distortion_analyzer_crossdomain as cx
    import src.distortion_analyzer_hungarian as dh
    saved = (da.find_hedges, dh.find_hedges,
             {d: (c["hedge_fn"], c["density_fn"]) for d, c in cx.DOMAIN_CONFIG.items()})
    try:
        da.find_hedges = dh.find_hedges = _without(saved[0], drop["clinical"])
        for d, cfg in cx.DOMAIN_CONFIG.items():
            find = _without(saved[2][d][0], drop[d])
            cfg["hedge_fn"], cfg["density_fn"] = find, _count(find)
        yield
    finally:
        da.find_hedges, dh.find_hedges = saved[0], saved[1]
        for d, cfg in cx.DOMAIN_CONFIG.items():
            cfg["hedge_fn"], cfg["density_fn"] = saved[2][d]


def _score(job):
    from src.efs import compute_efs
    path, source, domain, meta = job
    output = json.load(open(path))["output_text"]
    with restricted_markers():
        r = compute_efs(source, output, domain=domain)
    return dict(meta, efs=r["efs_score"], source_hedges=r["source_hedges"],
                **{t: r["by_type"].get(t, 0) for t in TYPES})


def _load_scores(study):
    d = pd.read_csv(require(RESULTS_DIR / "scores" / f"{study}.csv", f"{study} scores", "EE_RESULTS_DIR"),
                    low_memory=False)
    return d[d.status == "ok"].copy()


def jobs():
    """One job per generation scored in results/scores/{depth,crossdomain}.csv."""
    out = {"clinical": [], "legal": [], "scientific": []}
    depth = _load_scores("depth")
    for part in STUDIES["depth"]["parts"]:
        notes = pd.read_parquet(require(sample_path(part["sample"]), f"sample {part['sample']}", "EE_DATA_DIR"))
        src = {str(int(h)): t[:MAX_SOURCE_CHARS] for h, t in zip(notes.hadm_id, notes.discharge_text)}
        gdir = require(OUTPUTS_DIR / part["dir"], "depth-study generations", "EE_OUTPUTS_DIR")
        for r in depth[depth.seed == part["seed"]].itertuples():
            out["clinical"].append((str(gdir / r.model / r.file), src[str(r.doc_id)], "clinical",
                                    {"seed": part["seed"], "doc_id": str(r.doc_id), "model": r.model, "task": r.task}))
    xd = _load_scores("crossdomain")
    for part in STUDIES["crossdomain"]["parts"]:
        dom = part["domain"]
        docs = pd.read_parquet(require(sample_path(part["sample"]), f"sample {part['sample']}", "EE_DATA_DIR"))
        src = {str(i): t[:MAX_SOURCE_CHARS] for i, t in zip(docs[part["id_col"]], docs.source_text)}
        gdir = require(OUTPUTS_DIR / part["dir"], f"{dom} generations", "EE_OUTPUTS_DIR")
        for r in xd[xd.domain == dom].itertuples():
            out[dom].append((str(gdir / r.model / r.file), src[str(r.doc_id)], dom,
                             {"doc_id": r.doc_id, "model": r.model, "task": r.task}))
    return out


def r9(x):
    return np.round(np.asarray(x, dtype=float), 9)


def seed_mean_by_model(d):
    """Per-model mean of the per-seed mean EFS."""
    return d.groupby(["model", "seed"]).efs.mean().groupby("model").mean()


def analyze(cl, lg, sc):
    base = _load_scores("depth")
    v = {"J4.n_clinical": len(cl), "J4.n_legal": len(lg), "J4.n_scientific": len(sc),
         "J4.unrestricted.clinical_3seed_efs": base.efs.mean(), "J4.clinical_3seed_efs": cl.efs.mean(),
         "J4.unrestricted.markers_per_doc": base.source_hedges.mean(), "J4.markers_per_doc": cl.source_hedges.mean()}
    for s in (123, 0, 1):
        v[f"J4.unrestricted.markers_per_doc.seed{s}"] = base[base.seed == s].source_hedges.mean()

    a, b = seed_mean_by_model(base), seed_mean_by_model(cl)
    common = sorted(set(a.index) & set(b.index))
    rho, p = stats.spearmanr(a[common], b[common])
    v["J4.rank_spearman_vs_unrestricted"], v["J4.rank_spearman_p"] = rho, p
    rank = b.rank(ascending=False)
    for m in STUDY_11:
        v[f"J4.efs.{m}"], v[f"J4.rank.{m}"] = b[m], int(rank[m])

    for name, d in (("unrestricted.", base), ("", cl)):
        tot = d[TYPES].sum()
        for t in TYPES:
            v[f"J4.{name}{t}_pct"] = 100 * tot[t] / tot.sum()
        v[f"J4.{name}crossseed_std_max"] = d.groupby(["model", "seed"]).efs.mean().unstack().std(axis=1).max()

    # between-domain ordering on the three models run in every domain (clinical: seed 123)
    matched = sorted(set(lg.model) & set(sc.model))
    c = cl[(cl.seed == 123) & cl.model.isin(matched)]
    v["J4.matched.clinical_efs"], v["J4.matched.scientific_efs"], v["J4.matched.legal_efs"] = \
        c.efs.mean(), sc.efs.mean(), lg.efs.mean()
    H, pk = stats.kruskal(r9(c.efs), r9(sc.efs), r9(lg.efs))
    v["J4.matched.kruskal_H"], v["J4.matched.kruskal_p"] = H, pk
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["score", "analyze", "all"], default="all")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.stage in ("score", "all"):
        with Pool(a.workers) as pool:
            for dom, js in jobs().items():
                d = pd.DataFrame(pool.map(_score, js, chunksize=16))
                d.to_csv(OUT_DIR / f"{dom}.csv", index=False)
                print(f"{dom:10s} n={len(d)} mean EFS {d.efs.mean():.4f} markers {d.source_hedges.mean():.2f}",
                      flush=True)
    if a.stage in ("analyze", "all"):
        cl, lg, sc = (pd.read_csv(require(OUT_DIR / f"{d}.csv", f"restricted {d} scores", "EE_RESULTS_DIR"))
                      for d in ("clinical", "legal", "scientific"))
        v = analyze(cl, lg, sc)
        for k, x in v.items():
            print(f"{k:45s} {x:.4g}")
        (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
        json.dump({k: (int(x) if isinstance(x, (int, np.integer)) else float(x)) for k, x in v.items()},
                  open(RESULTS_DIR / "analysis" / "restricted_markers.json", "w"), indent=1)
        print(f"wrote {RESULTS_DIR / 'analysis' / 'restricted_markers.json'}")


if __name__ == "__main__":
    main()
