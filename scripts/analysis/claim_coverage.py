"""How many source markers can be charged at all, and how many are charged when EFS reaches 0.

Paper: Appendix J.3 ("Floor, clamp and unclamped score"): "39% of markers occur in sentences that
never form a claim and can never be charged; at EFS = 0 a mean of 68% of markers are implicated,
not all of them."
Inputs: EE_DATA_DIR/full_sample.parquet, EE_OUTPUTS_DIR/full_generations/<model>/*.json,
        results/scores/depth.csv (lists the scored seed-123 generations);
        access: PhysioNet credentialed (MIMIC-IV notes). CPU only.
Outputs: results/claim_coverage/generations.csv, results/analysis/claim_coverage.json.
Usage:
    python scripts/analysis/claim_coverage.py [--workers 16]

Definitions (one-to-one aligner, the extractor's own marker-to-claim assignment):
  * a marker is REACHABLE if the claim extractor places it in a hedged source claim (the `hedges` of
    a claim returned by extract_clinical_claims(source, require_hedge=True)); every distortion
    except fabrication is charged to such a claim, so an unreachable marker can never be charged.
    J3.markers_never_in_claim_pct pools markers over the 192 notes (each note enters every
    generation of it, so pooling over generations gives the same value): 39.3% (paper: 39%).
  * a marker is IMPLICATED in a generation if it belongs to a source claim charged with at least one
    non-fabrication distortion (claims identified by their text, as the distortions record it).
    Over the 568 generations at EFS = 0: per-generation mean 66.1%, pooled 68.3%; every marker is
    implicated in 90 of them (15.8%). The paper's "68%" corresponds to the pooled share.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import OUTPUTS_DIR, RESULTS_DIR, require  # noqa: E402
from configs.studies import sample_path  # noqa: E402

MAX_SOURCE_CHARS = 12000


def _coverage(job):
    from src.distortion_analyzer import extract_clinical_claims
    from src.efs import compute_efs
    from src.hedge_screener import find_hedges
    path, source, meta = job
    r = compute_efs(source, json.load(open(path))["output_text"], domain="clinical")
    claims = extract_clinical_claims(source, require_hedge=True)
    charged = {d.source_span for d in r["distortions"] if d.distortion_type != "fabrication" and d.source_span}
    reachable = {(h.start, h.end) for c in claims for h in c["hedges"]}
    implicated = {(h.start, h.end) for c in claims if c["text"][:200] in charged for h in c["hedges"]}
    return dict(meta, efs=r["efs_score"], markers=len(find_hedges(source)),
                reachable=len(reachable), implicated=len(implicated))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    notes = pd.read_parquet(require(sample_path("full_sample.parquet"), "seed-123 clinical sample", "EE_DATA_DIR"))
    src = {str(int(h)): t[:MAX_SOURCE_CHARS] for h, t in zip(notes.hadm_id, notes.discharge_text)}
    d = pd.read_csv(require(RESULTS_DIR / "scores" / "depth.csv", "depth-study scores", "EE_RESULTS_DIR"),
                    low_memory=False)
    d = d[(d.status == "ok") & (d.seed == 123)]
    gdir = require(OUTPUTS_DIR / "full_generations", "seed-123 clinical generations", "EE_OUTPUTS_DIR")
    jobs = [(str(gdir / r.model / r.file), src[str(r.doc_id)],
             {"doc_id": str(r.doc_id), "model": r.model, "task": r.task}) for r in d.itertuples()]
    with Pool(a.workers) as p:
        g = pd.DataFrame(p.map(_coverage, jobs, chunksize=16))
    out = RESULTS_DIR / "claim_coverage"
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "generations.csv", index=False)

    per_note = g.drop_duplicates("doc_id")
    z = g[g.efs == 0]
    v = {"J3.n_generations": len(g), "J3.coverage_n_notes": len(per_note),
         "J3.markers_never_in_claim_pct": 100 * (1 - per_note.reachable.sum() / per_note.markers.sum()),
         "J3.markers_never_in_claim_pct_mean_per_note": 100 * (1 - per_note.reachable / per_note.markers).mean(),
         "J3.efs0.n": len(z), "J3.efs0.pct_of_generations": 100 * len(z) / len(g),
         "J3.efs0.implicated_pct_mean": 100 * (z.implicated / z.markers).mean(),
         "J3.efs0.implicated_pct_pooled": 100 * z.implicated.sum() / z.markers.sum(),
         "J3.efs0.all_implicated": int((z.implicated == z.markers).sum())}
    for k, x in v.items():
        print(f"{k:45s} {x:.4g}")
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump({k: (int(x) if isinstance(x, (int, np.integer)) else float(x)) for k, x in v.items()},
              open(RESULTS_DIR / "analysis" / "claim_coverage.json", "w"), indent=1)
    print(f"wrote {RESULTS_DIR / 'analysis' / 'claim_coverage.json'}")


if __name__ == "__main__":
    main()
