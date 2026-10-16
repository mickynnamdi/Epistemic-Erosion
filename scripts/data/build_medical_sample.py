"""Build the PubMedQA sample used as the biomedical domain of the hosted-model study.

Paper: Section 5 / Table 2 ("Biomedical" column). MIMIC-IV text may not be sent to hosted
APIs, so PubMed abstracts stand in for clinical text there; they are scored with the
clinical lexicon.
Inputs: PubMedQA pqa_labeled, downloaded from the Hugging Face Hub (qiaojin/PubMedQA; MIT). Public.
Outputs: data/medical_sample.parquet (abstract_id, question, source_text, source_hedges, hedges_per_1k).
Usage:
    python scripts/data/build_medical_sample.py

Sampling: abstracts of 600-12,000 characters with >= 2 clinical hedges (378 candidates),
stratified into terciles of hedge density and sampled with random_state=42 (100 per tercile).
`hedges_per_1k` is per 1,000 words (the legal/scientific samples use characters).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import DATA_DIR  # noqa: E402
from src.hedge_screener import find_hedges  # noqa: E402

OUT_PATH = DATA_DIR / 'medical_sample.parquet'
N_TARGET = 300

print("Loading PubMedQA (labeled split)…", flush=True)
from datasets import load_dataset
PUBMEDQA_REV = "9001f2853fb87cab8d220904e0de81ac6973b318"   # Hugging Face revision of 2024-03-06
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", revision=PUBMEDQA_REV)
print(f"  loaded {len(ds)} examples", flush=True)

rows = []
for ex in ds:
    pmid = ex.get("pubid")
    q    = ex.get("question") or ""
    ctxs = ex.get("context", {}) or {}
    abst_chunks = ctxs.get("contexts") if isinstance(ctxs, dict) else []
    if not abst_chunks:
        continue
    abstract = "\n\n".join(abst_chunks)
    if len(abstract) < 600 or len(abstract) > 12000:
        continue
    hedges = find_hedges(abstract)
    n_hedge = len(hedges)
    if n_hedge < 2:  # need enough hedges to compute meaningful EFS
        continue
    rows.append({
        "abstract_id": str(pmid),
        "question": q,
        "source_text": abstract,
        "source_hedges": n_hedge,
        "hedges_per_1k": round(1000 * n_hedge / max(len(abstract.split()), 1), 2),
    })

print(f"  candidates with >= 2 hedges: {len(rows)}", flush=True)

df = pd.DataFrame(rows)
# Stratify by hedge density (low / mid / high)
df["bin"] = pd.qcut(df["hedges_per_1k"], q=3, labels=["low","mid","high"])
per_bin = N_TARGET // 3
extra   = N_TARGET % 3
sampled = []
for i, b in enumerate(["low","mid","high"]):
    n = per_bin + (1 if i < extra else 0)
    pool = df[df["bin"] == b]
    sampled.append(pool.sample(n=min(n, len(pool)), random_state=42))
out = pd.concat(sampled, ignore_index=True).drop(columns=["bin"])

# Stable order
out = out.reset_index(drop=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT_PATH, index=False)
print(f"\nWrote {len(out)} samples to {OUT_PATH}", flush=True)
print(f"  mean hedges per abstract: {out['source_hedges'].mean():.1f}", flush=True)
print(f"  mean abstract length:     {out['source_text'].str.len().mean():.0f} chars", flush=True)
