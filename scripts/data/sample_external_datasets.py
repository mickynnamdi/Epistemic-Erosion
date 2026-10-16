"""Draw the 192-document legal and scientific samples (and the screened pools they come from).

Paper: Section 5 (Multi-LexSum: 192 cases from a screened pool of 1,881; SciTLDR: 192 papers
from a screened pool of 2,615), used by Section 6.5, Table 2 and the appendices.
Inputs: the public releases downloaded by scripts/data/download_public_data.sh into
EE_EXTERNAL_DIR (Multi-LexSum v20230518, SciTLDR-AIC).
Outputs: data/legal_sample.parquet, data/scientific_sample.parquet; with --pools also
data/legal_sample_full.parquet and data/scientific_sample_full.parquet (the screened pools,
needed only for the detector transfer test of Table A14).
Usage:
    python scripts/data/sample_external_datasets.py [--pools]

Sampling: seed 42, 48 documents per stratum, sources with >= 3 domain hedges. Legal strata
are the four most frequent case types; scientific strata are hedge-count quartiles.
Note: the scientific sample applies the >= 3 filter to the full paper text (2,616 candidates),
while the paper's screened pool of 2,615 applied it to the first 12,000 characters; both give
the same 192-paper sample.
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.hedge_screener_legal import find_legal_hedges, legal_hedge_density
from src.hedge_screener_scientific import find_scientific_hedges, scientific_hedge_density
from configs.paths import DATA_DIR, MULTILEXSUM_DIR, SCITLDR_DIR, require

SEED = 42
N_PER_CATEGORY = 48
MAX_SOURCE_CHARS = 12000


def sample_legal(pools=False):
    """Sample from MultiLexSum."""
    print("=== Legal (MultiLexSum) ===")

    # Load all splits
    data_dir = require(MULTILEXSUM_DIR, "Multi-LexSum v20230518", "EE_EXTERNAL_DIR")
    all_cases = []
    for split in ["train", "dev", "test"]:
        with open(data_dir / f"{split}.json") as f:
            for line in f:
                all_cases.append(json.loads(line))
    print(f"  Total cases: {len(all_cases)}")

    # Load source documents
    with open(data_dir / "sources.json") as f:
        sources = json.loads(f.readline())
    print(f"  Total source documents: {len(sources)}")

    # Build episodes: each case with its summary and source docs
    episodes = []
    for case in all_cases:
        summary_long = case.get("summary/long")
        if not summary_long or len(summary_long) < 200:
            continue

        # Combine source documents for this case
        doc_ids = case.get("case_documents", [])
        source_texts = []
        for did in doc_ids:
            if did in sources:
                source_texts.append(sources[did]["doc_text"][:MAX_SOURCE_CHARS])
        if not source_texts:
            continue

        combined_source = "\n\n---\n\n".join(source_texts)[:MAX_SOURCE_CHARS]

        # Check hedge density
        hd = legal_hedge_density(combined_source)
        if hd["total_hedges"] < 3:
            continue

        episodes.append({
            "case_id": case["case_id"],
            "case_type": case.get("case_type", "Other"),
            "court": case.get("court", ""),
            "state": case.get("state", ""),
            "source_text": combined_source,
            "reference_summary": summary_long,
            "source_hedges": hd["total_hedges"],
            "hedges_per_1k": hd["hedges_per_1k_chars"],
        })

    print(f"  After hedge filter (>=3): {len(episodes)}")

    # Get top case types for stratification
    df = pd.DataFrame(episodes)
    if pools:
        pool = df[["case_id", "case_type", "source_text", "reference_summary", "source_hedges"]]
        pool.to_parquet(DATA_DIR / "legal_sample_full.parquet")
        print(f"  Saved screened pool ({len(pool)}) to {DATA_DIR / 'legal_sample_full.parquet'}")
    type_counts = df["case_type"].value_counts()
    print(f"  Case types: {dict(type_counts.head(8))}")

    # Take top 4 case types
    top_types = type_counts.head(4).index.tolist()
    print(f"  Sampling from: {top_types}")

    sampled = []
    random.seed(SEED)
    for ct in top_types:
        pool = df[df["case_type"] == ct]
        n = min(N_PER_CATEGORY, len(pool))
        sampled.append(pool.sample(n=n, random_state=SEED))
        print(f"    {ct}: sampled {n}/{len(pool)}")

    result = pd.concat(sampled).reset_index(drop=True)
    out_path = DATA_DIR / "legal_sample.parquet"
    result.to_parquet(out_path)
    print(f"  Saved {len(result)} episodes to {out_path}")
    print(f"  Mean hedges/doc: {result['source_hedges'].mean():.1f}")
    return result


def sample_scientific(pools=False):
    """Sample from SciTLDR."""
    print("\n=== Scientific (SciTLDR) ===")

    sci_dir = require(SCITLDR_DIR, "SciTLDR-AIC", "EE_EXTERNAL_DIR")
    all_papers = []
    for split in ["train", "dev", "test"]:
        with open(sci_dir / f"{split}.jsonl") as f:
            for line in f:
                paper = json.loads(line)
                paper["split"] = split
                all_papers.append(paper)
    print(f"  Total papers: {len(all_papers)}")

    # Build episodes
    episodes = []
    for paper in all_papers:
        source_text = " ".join(paper["source"])
        if len(source_text) < 200:
            continue

        target_tldr = paper["target"][0] if paper["target"] else ""
        if not target_tldr:
            continue

        # Check hedge density
        hd = scientific_hedge_density(source_text)
        if hd["total_hedges"] < 3:
            continue

        episodes.append({
            "paper_id": paper["paper_id"],
            "title": paper.get("title", ""),
            "source_text": source_text[:MAX_SOURCE_CHARS],
            "reference_tldr": target_tldr,
            "source_hedges": hd["total_hedges"],
            "hedges_per_1k": hd["hedges_per_1k_chars"],
        })

    print(f"  After hedge filter (>=3): {len(episodes)}")

    df = pd.DataFrame(episodes)
    if pools:
        # the paper's screened pool applied the >= 3 filter to the truncated (12,000-char) text
        h = df["source_text"].map(lambda s: scientific_hedge_density(s)["total_hedges"])
        pool = df.assign(source_hedges=h)[h >= 3][["paper_id", "title", "source_text", "reference_tldr", "source_hedges"]]
        pool.reset_index(drop=True).to_parquet(DATA_DIR / "scientific_sample_full.parquet")
        print(f"  Saved screened pool ({len(pool)}) to {DATA_DIR / 'scientific_sample_full.parquet'}")

    # For scientific, we don't have natural categories, so sample uniformly
    # stratified by hedge density (low/med/high)
    df["hedge_bin"] = pd.qcut(df["source_hedges"], q=4, labels=["low", "med", "high", "very_high"])
    print(f"  Hedge bins: {dict(df['hedge_bin'].value_counts())}")

    sampled = []
    random.seed(SEED)
    for hbin in df["hedge_bin"].unique():
        pool = df[df["hedge_bin"] == hbin]
        n = min(N_PER_CATEGORY, len(pool))
        sampled.append(pool.sample(n=n, random_state=SEED))
        print(f"    {hbin}: sampled {n}/{len(pool)}")

    result = pd.concat(sampled).reset_index(drop=True)
    out_path = DATA_DIR / "scientific_sample.parquet"
    result.to_parquet(out_path)
    print(f"  Saved {len(result)} episodes to {out_path}")
    print(f"  Mean hedges/doc: {result['source_hedges'].mean():.1f}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", action="store_true", help="also write the screened pools (Table A14 only)")
    a = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_legal(a.pools)
    sample_scientific(a.pools)
    print("\nDone.")
