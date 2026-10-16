"""EFS versus a raw hedge counter and ROUGE-L under controlled corruptions of clinical notes.

Paper: Appendix J.3 and Table A10 (60 notes, >= 5 markers, 19.1 markers per note; 53 notes shared
with the J.3 property sample). Mean EFS / hedge-count ratio / ROUGE-L:
identity 1.000/1.000/1.000, reorder sentences 0.998/1.000/0.334, delete 50% of hedges
0.670/0.498/0.996, relocate a hedge to another claim 0.806/1.000/0.997, add a hedge
0.935/1.218/0.998, fabricate a marked claim 0.931/1.070/0.993. Deletion ladder (fractions 0, .25,
.5, .75, 1) Spearman with the deleted fraction: EFS -0.897, counter -0.988, ROUGE-L -0.864.
Inputs: EE_DATA_DIR/full_sample.parquet; access: PhysioNet credentialed (MIMIC-IV notes). CPU only.
Outputs: results/validation/discriminance.csv (per note and corruption), results/analysis/discriminance.json.
Usage:
    python scripts/validation/discriminance.py [--workers 16]

Sample: the first 60 of the first 400 seed-123 notes (first 12,000 characters) with >= 5 markers.
k = max(1, markers // 4) markers are relocated or added; every corruption uses random.Random(note
position). Hedge-count ratio = |H(G)| / |H(S)|, unclamped. ROUGE-L: F1, Porter stemmer, source as
reference. Fabricate: append a real hedged claim from another note (among the first two hedged claims
of each of the first 80 notes) that shares no clinical term with this note's hedged claims.
"""
import argparse
import json
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import RESULTS_DIR, require  # noqa: E402
from configs.studies import sample_path  # noqa: E402

N_DOCS, POOL_DOCS, DONOR_DOCS, CHARS = 60, 400, 80, 12000
MIN_MARKERS, J3_MIN_MARKERS = 5, 4
FRACS = [0.25, 0.50, 0.75, 1.00]
MEASURES = ["efs", "hedge_count_ratio", "rougeL"]
_SENT = re.compile(r"(?<=[.!?])\s+")
_DONORS = []


def _init(donors):
    global _DONORS
    _DONORS = donors


def sentences(t):
    return [s for s in _SENT.split(t) if s.strip()]


def sent_index(src):
    """(sentence, start, end) for each sentence."""
    out, pos = [], 0
    for s in sentences(src):
        i = src.find(s, pos)
        out.append((s, i, i + len(s)))
        pos = i + len(s)
    return out


def cut_spans(text, spans):
    for st, en in sorted(spans, key=lambda x: -x[0]):
        text = text[:st] + text[en:]
    return text


def corrupt_delete(src, hedges, frac, rng):
    drop = rng.sample(range(len(hedges)), int(round(frac * len(hedges))))
    return cut_spans(src, [(hedges[i].start, hedges[i].end) for i in drop])


def corrupt_reorder(src, rng):
    ss = sentences(src)
    rng.shuffle(ss)
    return " ".join(ss)


def corrupt_relocate(src, hedges, k, rng):
    """Move k markers from their own sentence to the start of a sentence that has none (count unchanged)."""
    idx = sent_index(src)
    hedged = {i for i, (_, a, b) in enumerate(idx) for h in hedges if a <= h.start < b}
    movable = [h for h in hedges if any(a <= h.start < b for _, a, b in idx)]
    targets = [i for i in range(len(idx)) if i not in hedged]
    if not movable or not targets:
        return None
    k = min(k, len(movable), len(targets))
    moved, recv = rng.sample(movable, k), rng.sample(targets, k)
    new = [s for s, _, _ in idx]
    for h in sorted(moved, key=lambda h: -h.start):
        for i, (s, a, b) in enumerate(idx):
            if a <= h.start < b:
                new[i] = new[i][:h.start - a] + new[i][h.end - a:]
                break
    for h, i in zip(moved, recv):
        new[i] = f"{h.text} {new[i][0].lower()}{new[i][1:]}"
    return " ".join(x for x in new if x.strip())


def corrupt_add(src, hedges, k, rng):
    """Prefix k unhedged sentences with a marker drawn from the source's own markers (count rises)."""
    idx = sent_index(src)
    hedged = {i for i, (_, a, b) in enumerate(idx) for h in hedges if a <= h.start < b}
    targets = [i for i in range(len(idx)) if i not in hedged]
    if not targets or not hedges:
        return None
    recv = rng.sample(targets, min(k, len(targets)))
    vocab = [h.text for h in hedges]
    new = [s for s, _, _ in idx]
    for i in recv:
        new[i] = f"{rng.choice(vocab)} {new[i][0].lower()}{new[i][1:]}"
    return " ".join(new)


def one_note(job):
    from rouge_score import rouge_scorer
    from src.distortion_analyzer import extract_clinical_claims
    from src.efs import compute_efs
    from src.hedge_screener import find_hedges
    di, src = job
    hedges = find_hedges(src)
    k = max(1, len(hedges) // 4)
    variants = [("identity", src), ("reorder", corrupt_reorder(src, random.Random(di)))]
    variants += [(f"delete_{f:.2f}", corrupt_delete(src, hedges, f, random.Random(di))) for f in FRACS]
    variants.append(("relocate", corrupt_relocate(src, hedges, k, random.Random(di))))
    variants.append(("add_hedge", corrupt_add(src, hedges, k, random.Random(di))))
    terms = {t.lower() for c in extract_clinical_claims(src, require_hedge=True) for t in c["clinical_terms"]}
    graft = next((c["text"] for c in _DONORS if c["doc"] != di and c["terms"] and not (c["terms"] & terms)), None)
    if graft:
        variants.append(("fabricate", src + " " + graft))
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows = []
    for name, g in variants:
        if not g:
            continue
        r = compute_efs(src, g, domain="clinical")
        rows.append({"doc": di, "corruption": name, "n_hedges_src": len(hedges), "k": k,
                     "efs": r["efs_score"], "D": r["distortion_count"], "H": r["source_hedges"],
                     "hedge_count_ratio": len(find_hedges(g)) / len(hedges),
                     "rougeL": scorer.score(src, g)["rougeL"].fmeasure})
    return rows


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
    n_markers = [len(find_hedges(s)) for s in docs]
    chosen = [i for i, n in enumerate(n_markers) if n >= MIN_MARKERS][:N_DOCS]
    j3_sample = [i for i, n in enumerate(n_markers) if n >= J3_MIN_MARKERS][:N_DOCS]   # Appendix J.3 properties
    with Pool(a.workers, initializer=_init, initargs=(donors,)) as p:
        d = pd.DataFrame([r for rows in p.map(one_note, [(i, docs[i]) for i in chosen], chunksize=1) for r in rows])
    (RESULTS_DIR / "validation").mkdir(parents=True, exist_ok=True)
    d.to_csv(RESULTS_DIR / "validation" / "discriminance.csv", index=False)

    tab = d.groupby("corruption")[MEASURES].mean()
    print(tab.round(4).to_string())
    ladder = d[d.corruption.str.startswith("delete") | (d.corruption == "identity")].copy()
    ladder["f"] = ladder.corruption.map(lambda c: 0.0 if c == "identity" else float(c.split("_")[1]))
    v = {"A10.n_notes": d.doc.nunique(),
         "A10.markers_per_note": d[d.corruption == "identity"].n_hedges_src.mean(),
         "A10.notes_shared_with_J3": len(set(chosen) & set(j3_sample))}
    for c, row in tab.iterrows():
        for m in MEASURES:
            v[f"A10.{c}.{m}"] = row[m]
    for m in MEASURES:
        v[f"A10.deletion_spearman.{m}"] = stats.spearmanr(ladder["f"], ladder[m]).correlation
    for k, x in v.items():
        print(f"{k:40s} {x:.4g}")
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump({k: (int(x) if isinstance(x, (int, np.integer)) else float(x)) for k, x in v.items()},
              open(RESULTS_DIR / "analysis" / "discriminance.json", "w"), indent=1)
    print(f"wrote {RESULTS_DIR / 'analysis' / 'discriminance.json'}")


if __name__ == "__main__":
    main()
