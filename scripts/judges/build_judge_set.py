"""Build the balanced judge-validation reference set of Appendix I (242 positives + 242 true identity pairs).

Paper: Appendix I / Table A8. Positives: 242 depth-study generations (seed-123 sample) with
    >= 3 distortions and EFS <= 0.4 under the greedy baseline aligner, 22 per model over the
    11 open-weight models. Identity stratum: 242 notes, source == output byte-identical
    (first 3,000 characters), with the pipeline's labels computed on each pair: the greedy
    aligner flags 67/242 = 27.7% of them (Table A8, "FPR 27.7% @ 3k").
    The truncation stress test (source = full note, "output" = its first 3,000
    characters) is written only with --truncation-stress-test; it is not a specificity test.
Inputs: EE_DATA_DIR/full_sample.parquet and full_sample_1000.parquet (MIMIC-IV),
    EE_OUTPUTS_DIR/full_generations/<model>/*.json, and a greedy-aligner score table
    (default EE_RESULTS_DIR/scores/depth_greedy.csv from
    `python scripts/score_generations.py --study depth --aligner greedy`);
    access: PhysioNet credentialed.
Outputs: RESULTS_DIR/judges/reference_set.json (484 records, MIMIC text: never share),
    RESULTS_DIR/judges/truncation_stress_test.json (only with --truncation-stress-test),
    RESULTS_DIR/analysis/build_judge_set.json.
Usage:
    python scripts/judges/build_judge_set.py
    python scripts/judges/build_judge_set.py --compare path/to/reference_set.json
    python scripts/judges/build_judge_set.py --truncation-stress-test
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from configs.paths import DATA_DIR, OUTPUTS_DIR, RESULTS_DIR, require  # noqa: E402

N_PER_STRATUM = 242
IDENTITY_CHARS = 3000
SOURCE_CHARS = 12000
# Row order of the paper's greedy score table: model blocks in this order, then hadm_id, then task.
# The stratified draw below consumes one random stream across models in this order, so it must be kept.
SCORE_TABLE_MODEL_ORDER = ["llama-3.3-70b", "med42-70b", "llama-3.1-8b", "med42-8b", "mistral-7b",
                           "biomistral-7b", "openbio-70b", "medgemma-27b", "gemma-4-4b", "gemma-4-31b",
                           "gemma-4-moe"]


def load_scores(path):
    """Seed-123 depth-study rows as (hadm_id, model, task, efs_score, distortion_count) in the paper's order."""
    df = pd.read_csv(require(path, "greedy-aligner score table", "EE_RESULTS_DIR"))
    if "n_distortions" in df:   # table written by scripts/score_generations.py --aligner greedy
        df = df[(df["seed"] == 123) & (df["status"] == "ok")]
        df = df.rename(columns={"doc_id": "hadm_id", "efs": "efs_score", "n_distortions": "distortion_count"})
        df = df.assign(_m=df["model"].map(SCORE_TABLE_MODEL_ORDER.index))
        df = df.sort_values(["_m", "hadm_id", "task"], kind="stable").drop(columns="_m")
    elif "seed" in df:          # the paper's multi-seed score table (already in this order)
        df = df[df["seed"] == 123]
    return df.reset_index(drop=True)


def stratified(df, n, seed):
    """Shuffle each model's pool with one shared RNG, take n // n_models per model, shuffle the union."""
    rng = random.Random(seed)
    by_model = {m: df[df["model"] == m] for m in df["model"].unique()}
    per_model = max(1, n // len(by_model))
    out = []
    for m, sub in by_model.items():
        idx = list(sub.index)
        rng.shuffle(idx)
        out.extend([sub.loc[i] for i in idx[:per_model]])
    rng.shuffle(out)
    return out[:n]


def load_output(model, hadm, task):
    p = OUTPUTS_DIR / "full_generations" / model / f"{hadm}_{model}_{task}_none.json"
    if not p.exists():
        return None
    return json.load(open(p)).get("output_text", "")


def build(scores_path):
    from src.efs import compute_efs_greedy
    from src.hedge_screener import hedge_density

    depth = pd.read_parquet(require(DATA_DIR / "full_sample.parquet", "depth-study sample", "EE_DATA_DIR"))
    breadth = pd.read_parquet(require(DATA_DIR / "full_sample_1000.parquet", "breadth-study sample", "EE_DATA_DIR"))
    src_lookup = {int(r["hadm_id"]): r["discharge_text"][:SOURCE_CHARS]
                  for _, r in pd.concat([depth, breadth]).drop_duplicates("hadm_id").iterrows()}
    require(OUTPUTS_DIR / "full_generations", "depth-study generations", "EE_OUTPUTS_DIR")

    mu = load_scores(scores_path)
    pos_pool = mu[(mu["distortion_count"] >= 3) & (mu["efs_score"] <= 0.4)]
    print(f"positive pool (>=3 distortions, EFS<=0.4, greedy aligner): {len(pos_pool)}")
    positives = []
    for r in stratified(pos_pool, 250, seed=43):   # 250 // 11 = 22 per model -> 242
        if len(positives) >= N_PER_STRATUM:
            break
        src = src_lookup.get(int(r["hadm_id"]))
        out = load_output(r["model"], int(r["hadm_id"]), r["task"])
        if not src or not out:
            continue
        positives.append({"hadm_id": int(r["hadm_id"]), "model": r["model"], "task": r["task"],
                          "label": "positive", "efs_score": float(r["efs_score"]),
                          "distortion_count": int(r["distortion_count"]),
                          "source_text": src, "output_text": out})

    # Identity notes: shuffled pool of every sampled episode (depth + breadth), notes >= 200 characters.
    rng = random.Random(45)
    all_hadms = sorted(src_lookup)
    rng.shuffle(all_hadms)
    notes = [h for h in all_hadms if len(src_lookup[h]) >= 200][:N_PER_STRATUM]
    identity, truncation = [], []
    for h in notes:
        text = src_lookup[h][:IDENTITY_CHARS]
        res = compute_efs_greedy(text, text, domain="clinical")
        identity.append({"hadm_id": h, "model": "source-self", "task": "identity_true", "label": "negative",
                         "source_text": text, "output_text": text,
                         "pipeline_efs": res["efs_score"], "pipeline_distortion_count": len(res["distortions"]),
                         "source_hedges": hedge_density(text)["total_hedges"]})
        truncation.append({"hadm_id": h, "model": "source-self", "task": "truncation",
                           "source_text": src_lookup[h], "output_text": src_lookup[h][:IDENTITY_CHARS]})
    assert all(r["source_text"] == r["output_text"] for r in identity), "identity pair is not byte-identical"

    # Final shuffle with the same RNG stream; it fixes the record order used by every judge run
    # (batch composition of the batched judge, and the first-60 subsets of the positive control).
    order = [("p", i) for i in range(len(positives))] + [("i", i) for i in range(len(identity))]
    rng.shuffle(order)
    records = [positives[i] if s == "p" else identity[i] for s, i in order]
    trunc_ordered = [truncation[i] for s, i in order if s == "i"]
    return records, trunc_ordered, positives, identity


def compare(records, trunc, ref_path):
    """Match against a reference set JSON by (hadm_id, model, task) and by text."""
    ref = json.load(open(ref_path))
    ref_pos = [r for r in ref if r.get("label") == "positive"]
    ref_id = [r for r in ref if r.get("model") == "source-self"]
    pos = [r for r in records if r["label"] == "positive"]
    ids = [r for r in records if r["model"] == "source-self"]
    key = lambda r: (int(r["hadm_id"]), r["model"], r["task"])  # noqa: E731
    ref_src = {key(r): r for r in ref_pos}
    ref_trunc = {int(r["hadm_id"]): r for r in ref_id}
    v = {"I.build.positives_matching_reference": len({key(r) for r in pos} & set(ref_src)),
         "I.build.positive_order_matches_reference": int([key(r) for r in pos] == [key(r) for r in ref_pos]),
         "I.build.identity_notes_matching_reference": len({int(r["hadm_id"]) for r in ids} & set(ref_trunc)),
         "I.build.identity_order_matches_reference":
             int([int(r["hadm_id"]) for r in ids] == [int(r["hadm_id"]) for r in ref_id]),
         "I.build.positive_texts_identical": sum(
             1 for r in pos if key(r) in ref_src and r["source_text"] == ref_src[key(r)]["source_text"]
             and r["output_text"] == ref_src[key(r)]["output_text"]),
         "I.build.truncation_texts_identical": sum(
             1 for r in trunc if int(r["hadm_id"]) in ref_trunc
             and r["source_text"] == ref_trunc[int(r["hadm_id"])]["source_text"]
             and r["output_text"] == ref_trunc[int(r["hadm_id"])]["output_text"])}
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, default=RESULTS_DIR / "scores" / "depth_greedy.csv",
                    help="greedy-aligner depth-study score table (format of scripts/score_generations.py, "
                         "or the paper's own score table, to reproduce its exact draw)")
    ap.add_argument("--truncation-stress-test", action="store_true",
                    help="also write the truncation stress-test pairs (not a specificity test)")
    ap.add_argument("--compare", type=Path, help="reference set JSON to check the draw against")
    a = ap.parse_args()

    records, trunc, positives, identity = build(a.scores)
    out_dir = RESULTS_DIR / "judges"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(out_dir / "reference_set.json", "w"))
    print(f"wrote {out_dir / 'reference_set.json'}  ({len(positives)} positives + {len(identity)} identity)")
    if a.truncation_stress_test:
        json.dump(trunc, open(out_dir / "truncation_stress_test.json", "w"))
        print(f"wrote {out_dir / 'truncation_stress_test.json'}  (truncation stress test; not a specificity test)")

    per_model = pd.Series([r["model"] for r in positives]).value_counts()
    flagged = sum(r["pipeline_distortion_count"] > 0 for r in identity)
    v = {"I.positives.n": len(positives), "I.positives.n_models": int(per_model.size),
         "I.positives.per_model_min": int(per_model.min()), "I.positives.per_model_max": int(per_model.max()),
         "I.identity.n": len(identity),
         "I.identity.byte_identical": int(all(r["source_text"] == r["output_text"] for r in identity)),
         "A8.identity.pipeline_greedy_flagged": int(flagged),
         "A8.identity.pipeline_greedy_fpr_pct": 100 * flagged / len(identity),
         "A8.identity.pipeline_greedy_mean_efs": float(np.mean([r["pipeline_efs"] for r in identity]))}
    if a.compare:
        v.update(compare(records, trunc, a.compare))
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump(v, open(RESULTS_DIR / "analysis" / "build_judge_set.json", "w"), indent=1)
    for k, x in v.items():
        print(f"  {k:50s} {x}")


if __name__ == "__main__":
    main()
