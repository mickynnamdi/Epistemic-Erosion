"""Measure properties of EFS tested on real clinical notes (identity, monotonicity, scale, fabrication).

Paper: Appendix J.3. 60 notes (16.9 lexicon markers per note); identity EFS(s, s) = 1 on all 60
(greedy baseline aligner 0.823, 80.0% of notes with a spurious distortion); local monotonicity:
0 violations in 919 cumulative single-hedge deletions, mean EFS 0.340 after all deletions; scale
invariance max |delta| = 0; a grafted hedged claim from another note lowers EFS in 59/60 notes
(1.000 -> 0.907); an invented certain assertion is charged in only 1/60 notes.
Inputs: EE_DATA_DIR/full_sample.parquet; access: PhysioNet credentialed (MIMIC-IV notes). CPU only.
Outputs: results/validation/efs_properties.csv (per note), results/analysis/efs_properties.json.
Usage:
    python scripts/validation/efs_properties.py [--workers 16]

Sample: the first 60 of the first 400 seed-123 notes (first 12,000 characters) with >= 4 markers.
Deletion: markers are removed one at a time, cumulatively, in a random order seeded by the note's
position (at most 25 per note). Graft: the first extracted hedged claim, among the first two of each
of the first 80 notes, that comes from another note and shares no clinical term with this note's
hedged claims. Scale: the source and an output with the first half of its markers deleted are both
duplicated (joined by a newline).
"""
import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import RESULTS_DIR, require  # noqa: E402
from configs.studies import sample_path  # noqa: E402

N_DOCS, POOL_DOCS, DONOR_DOCS, CHARS = 60, 400, 80, 12000
MIN_MARKERS, MAX_STEPS = 4, 25
CERTAIN = (" The patient has advanced metastatic disease. "
           "Pulmonary embolism is present. The lesion is malignant.")

_DONORS = []


def _init(donors):
    global _DONORS
    _DONORS = donors


def efs(s, g):
    from src.efs import compute_efs
    return compute_efs(s, g, domain="clinical")


def cut(text, spans):
    """Delete the given character spans; the rest of the text is unchanged."""
    for st, en in sorted(spans, key=lambda x: -x[0]):
        text = text[:st] + text[en:]
    return text


def claim_terms(text):
    from src.distortion_analyzer import extract_clinical_claims
    return {t.lower() for c in extract_clinical_claims(text, require_hedge=True) for t in c["clinical_terms"]}


def one_note(job):
    from src.efs import compute_efs_greedy
    from src.hedge_screener import find_hedges
    di, src = job
    spans = [(h.start, h.end) for h in find_hedges(src)]
    H = len(spans)
    ident = efs(src, src)
    greedy = compute_efs_greedy(src, src, domain="clinical")
    out = {"doc": di, "n_hedges": H, "identity_efs": ident["efs_score"], "identity_dist": ident["distortion_count"],
           "greedy_identity_efs": greedy["efs_score"], "greedy_identity_dist": greedy["distortion_count"]}

    # local monotonicity: cumulative one-at-a-time deletion
    order = list(range(H))
    random.Random(di).shuffle(order)
    prev, removed, viol, e = out["identity_efs"], [], 0, None
    for i in order[:MAX_STEPS]:
        removed.append(spans[i])
        e = efs(src, cut(src, removed))["efs_score"]
        viol += e > prev + 1e-6
        prev = e
    out.update(steps=min(H, MAX_STEPS), violations=viol, final_efs=e)

    # hedged fabrication: graft a real hedged claim from another note with disjoint clinical terms
    terms = claim_terms(src)
    graft = next((c["text"] for c in _DONORS if c["doc"] != di and c["terms"] and not (c["terms"] & terms)), None)
    out["graft_efs_before"] = ident["efs_score"]
    if graft:
        g = efs(src, src + " " + graft)
        out.update(graft_efs_after=g["efs_score"], graft_decreased=g["efs_score"] < ident["efs_score"] - 1e-6)

    # certain (unhedged) fabrication
    c = efs(src, src + CERTAIN)
    out.update(certain_efs_after=c["efs_score"], certain_decreased=c["efs_score"] < ident["efs_score"] - 1e-6)

    # scale invariance
    half = cut(src, spans[: H // 2])
    single = efs(src, half)["efs_score"]
    doubled = efs(src + "\n" + src, half + "\n" + half)["efs_score"]
    out["scale_abs_delta"] = abs(doubled - single)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from src.distortion_analyzer import extract_clinical_claims
    from src.hedge_screener import find_hedges

    notes = pd.read_parquet(require(sample_path("full_sample.parquet"), "seed-123 clinical sample", "EE_DATA_DIR"))
    docs = [str(t)[:CHARS] for t in notes["discharge_text"]][:POOL_DOCS]
    donors = [{"doc": i, "text": c["text"], "terms": {t.lower() for t in c["clinical_terms"]}}
              for i, s in enumerate(docs[:DONOR_DOCS]) for c in extract_clinical_claims(s, require_hedge=True)[:2]]
    chosen = [(i, s) for i, s in enumerate(docs) if len(find_hedges(s)) >= MIN_MARKERS][:N_DOCS]
    with Pool(a.workers, initializer=_init, initargs=(donors,)) as p:
        d = pd.DataFrame(p.map(one_note, chosen, chunksize=1))
    (RESULTS_DIR / "validation").mkdir(parents=True, exist_ok=True)
    d.to_csv(RESULTS_DIR / "validation" / "efs_properties.csv", index=False)

    v = {"J3.n_notes": len(d), "J3.markers_per_note": d.n_hedges.mean(),
         "J3.identity_min_efs": d.identity_efs.min(), "J3.identity_notes_with_distortion": int((d.identity_dist > 0).sum()),
         "J3.greedy_identity_mean_efs": d.greedy_identity_efs.mean(),
         "J3.greedy_identity_fpr_pct": 100 * (d.greedy_identity_dist > 0).mean(),
         "J3.deletion_steps": int(d.steps.sum()), "J3.deletion_violations": int(d.violations.sum()),
         "J3.efs_after_all_deletions": d.final_efs.mean(),
         "J3.scale_max_abs_delta": d.scale_abs_delta.max(),
         "J3.graft_notes_tested": int(d.graft_efs_after.notna().sum()),
         "J3.graft_notes_decreased": int(d.graft_decreased.fillna(False).astype(bool).sum()),
         "J3.graft_efs_before": d.graft_efs_before.mean(), "J3.graft_efs_after": d.graft_efs_after.mean(),
         "J3.certain_notes_decreased": int(d.certain_decreased.sum())}
    for k, x in v.items():
        print(f"{k:36s} {x:.4g}")
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump({k: (int(x) if isinstance(x, (int, np.integer)) else float(x)) for k, x in v.items()},
              open(RESULTS_DIR / "analysis" / "efs_properties.json", "w"), indent=1)
    print(f"wrote {RESULTS_DIR / 'analysis' / 'efs_properties.json'}")


if __name__ == "__main__":
    main()
