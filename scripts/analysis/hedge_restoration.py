"""Diagnostic post-hoc hedge restoration: re-insert dropped or hardened source hedges, re-score EFS.

Paper: Appendix F.3. EFS 0.493 -> 0.832 and -66% detected distortions (n = 200) with the
one-to-one aligner; 0.458 -> 0.732 and -50% under the greedy baseline aligner.
Inputs: EE_DATA_DIR/full_sample.parquet, EE_OUTPUTS_DIR/full_generations/<model>/*.json,
        results/scores/depth.csv (defines the sampling frame);
        access: PhysioNet credentialed (MIMIC-IV notes). CPU only.
Outputs: results/hedge_restoration/pairs.csv, results/analysis/hedge_restoration.json.
Usage:
    python scripts/analysis/hedge_restoration.py

Restoration: the greedy baseline detector (as in the paper's run) lists the source sentences
of every omission and escalation; they are appended verbatim, de-duplicated, as a bullet block
"Source-stated uncertainties (restored from input)". The restored output is re-scored.

Expected output:
    one-to-one: EFS 0.4935 -> 0.8317, distortions 8.19 -> 2.765 (-66%)
    greedy:     EFS 0.4584 -> 0.7319, distortions 8.855 -> 4.405 (-50%)
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import OUTPUTS_DIR, RESULTS_DIR, require  # noqa: E402
from configs.studies import sample_path  # noqa: E402

N_PAIRS, SEED = 200, 42
MODELS = ["llama-3.1-8b", "med42-8b", "mistral-7b", "gemma-4-4b", "llama-3.3-70b", "biomistral-7b"]
# NOTE: the sample is drawn by position (DataFrame.sample), so the frame must be in the row order of
# the paper's results file: models in this order, then hadm_id, then task.
FRAME_ORDER = ["llama-3.3-70b", "med42-70b", "llama-3.1-8b", "med42-8b", "mistral-7b", "biomistral-7b",
               "openbio-70b", "medgemma-27b", "gemma-4-4b", "gemma-4-31b", "gemma-4-moe"]
KEYS = ["hadm_id", "model", "task"]


def sample_pairs():
    """The 200 seed-123 generations of the paper's run (six models, random_state 42)."""
    d = pd.read_csv(require(RESULTS_DIR / "scores" / "depth.csv", "depth-study scores", "EE_RESULTS_DIR"),
                    low_memory=False)
    d = d[(d.status == "ok") & (d.seed == 123)].copy()
    d["hadm_id"] = d.doc_id.astype(int)
    d["_order"] = d.model.map(FRAME_ORDER.index)
    d = d.sort_values(["_order", "hadm_id", "task"]).reset_index(drop=True)
    # the paper's run also required >= 1 source hedge, which every seed-123 generation satisfies
    return d[d.model.isin(MODELS) & (d.source_hedges >= 1)].sample(n=N_PAIRS, random_state=SEED)[KEYS]


def restore(source, output):
    """Append the source sentences of every greedy-detected omission and escalation."""
    from src.distortion_analyzer import detect_distortions
    insts = detect_distortions(source, output)
    spans = [i.source_span for i in insts if i.distortion_type == "omission" and i.source_span]
    spans += [i.source_span for i in insts if i.distortion_type == "escalation" and i.source_span]
    spans = list(dict.fromkeys(spans))
    if not spans:
        return output
    block = "\n\n**Source-stated uncertainties (restored from input):**\n"
    for s in spans:
        block += f"- {s.strip().rstrip('.')}.\n"
    return output + block


def score_pair(job):
    from src.efs import compute_efs, compute_efs_greedy
    source, output = job
    restored = restore(source, output)
    row = {}
    for name, fn in (("hungarian", compute_efs), ("greedy", compute_efs_greedy)):
        b, a = fn(source, output, domain="clinical"), fn(source, restored, domain="clinical")
        row.update({f"{name}_efs_before": b["efs_score"], f"{name}_dist_before": b["distortion_count"],
                    f"{name}_efs_after": a["efs_score"], f"{name}_dist_after": a["distortion_count"]})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    pairs = sample_pairs()
    notes = pd.read_parquet(require(sample_path("full_sample.parquet"), "seed-123 clinical sample", "EE_DATA_DIR"))
    sources = {int(h): t[:12000] for h, t in zip(notes.hadm_id, notes.discharge_text)}
    gen_dir = require(OUTPUTS_DIR / "full_generations", "seed-123 clinical generations", "EE_OUTPUTS_DIR")
    jobs = []
    for h, m, t in pairs.itertuples(index=False):
        g = json.load(open(gen_dir / m / f"{h}_{m}_{t}_none.json"))
        jobs.append((sources[h], g["output_text"]))
    with Pool(a.workers) as p:
        res = pd.DataFrame(p.map(score_pair, jobs, chunksize=4))
    d = pd.concat([pairs.reset_index(drop=True), res], axis=1)
    out = RESULTS_DIR / "hedge_restoration"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "pairs.csv", index=False)

    v = {"F3.n": len(d)}
    for name, pre in (("hungarian", "F3."), ("greedy", "F3.greedy_")):
        v[pre + "efs_before"] = d[f"{name}_efs_before"].mean()
        v[pre + "efs_after"] = d[f"{name}_efs_after"].mean()
        v[pre + "dist_before"] = d[f"{name}_dist_before"].mean()
        v[pre + "dist_after"] = d[f"{name}_dist_after"].mean()
        v[pre + "dist_reduction_pct"] = 100 * (1 - v[pre + "dist_after"] / v[pre + "dist_before"])
    for k, x in v.items():
        print(f"{k:28s} {x:.4f}")
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump({k: (int(x) if k == "F3.n" else float(x)) for k, x in v.items()},
              open(RESULTS_DIR / "analysis" / "hedge_restoration.json", "w"), indent=1)
    print(f"wrote {RESULTS_DIR / 'analysis' / 'hedge_restoration.json'}")


if __name__ == "__main__":
    main()
