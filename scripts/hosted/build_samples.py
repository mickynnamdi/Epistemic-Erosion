"""Re-draw the per-seed document lists of the hosted-model study and compare them with the shipped ones.

Paper: Section 5 / Table 2 (100 documents per domain and seed).
Inputs: data/{legal,scientific,medical}_sample.parquet (public), data/hosted_samples/seed0.json.
Outputs: prints, per seed and domain, whether the re-drawn list equals the shipped one;
with --write, writes data/hosted_samples/seed{1,2}.json (seed 0 is never overwritten).
Usage:
    python scripts/hosted/build_samples.py

Seed 0 was drawn with random.Random(42) over legal, then scientific, then medical. Its
medical list was drawn from an earlier version of the PubMedQA sample that no longer
exists (19 of its 100 abstracts are not in the current sample), so seed 0 is shipped as
data and only its legal and scientific lists can be re-drawn. Seeds 1 and 2 use
random.Random(seed) and exclude the ids of the earlier seeds when at least 100 documents remain
(biomedical); the 192-document legal and scientific samples are drawn whole, so they overlap.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import DATA_DIR, require  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "hosted_samples"
DOMAINS = {"legal": ("legal_sample.parquet", "case_id"), "scientific": ("scientific_sample.parquet", "paper_id"),
           "medical": ("medical_sample.parquet", "abstract_id")}
N = 100


def ids_of(domain):
    f, col = DOMAINS[domain]
    return pd.read_parquet(require(DATA_DIR / f, f"{domain} sample", "EE_DATA_DIR"))[col].astype(str).tolist()


def seed0_legal_scientific():
    rng = random.Random(42)
    return {d: [str(i) for i in rng.sample(ids_of(d), N)] for d in ("legal", "scientific")}


def later_seed(seed, earlier):
    rng = random.Random(seed)
    out = {}
    for d in DOMAINS:
        excluded = set().union(*[set(e[d]) for e in earlier])
        avail = [i for i in ids_of(d) if i not in excluded]
        if len(avail) < N:   # legal/scientific have only 192 documents: overlap is then allowed
            avail = ids_of(d)
        out[d] = [str(i) for i in rng.sample(avail, N)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write seed1/seed2 lists instead of only comparing")
    a = ap.parse_args()
    shipped = {s: json.load(open(SAMPLES / f"seed{s}.json")) for s in (0, 1, 2)}
    s0 = seed0_legal_scientific()
    for d in s0:
        print(f"seed 0 {d:10s}: re-drawn == shipped: {s0[d] == shipped[0][d]}")
    print("seed 0 medical   : shipped only (drawn from an earlier PubMedQA sample)")
    s1 = later_seed(1, [shipped[0]])
    s2 = later_seed(2, [shipped[0], s1])
    for s, drawn in ((1, s1), (2, s2)):
        for d in DOMAINS:
            print(f"seed {s} {d:10s}: re-drawn == shipped: {drawn[d] == shipped[s][d]}")
        if a.write:
            json.dump(drawn, open(SAMPLES / f"seed{s}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
