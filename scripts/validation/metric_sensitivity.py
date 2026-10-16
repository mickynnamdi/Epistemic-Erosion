"""EFS and ROUGE-L under a de-hedging edit and an unrelated-content deletion of the same notes.

Paper: Appendix F.2 (and Section 6.3). Stripping every detected hedge drops EFS from 1.000 to 0.268
while ROUGE-L stays at 0.994; removing unrelated content lowers both (EFS 0.757, ROUGE-L 0.676);
the greedy baseline aligner gives 0.270 on the de-hedged notes.
Inputs: EE_DATA_DIR/full_sample.parquet; access: PhysioNet credentialed (MIMIC-IV notes). CPU only.
Outputs: results/analysis/metric_sensitivity.json.
Usage:
    python scripts/validation/metric_sensitivity.py

Notes: of the first 60 seed-123 notes (first 6,000 characters) those with >= 3 markers (57).
De-hedge: delete every span the extractor returns. Content cut: keep every sentence containing one of
a fixed list of hedge words and drop every second remaining sentence. Hedge retention = output/source
marker count (source count floored at 1). ROUGE-L: F1, Porter stemmer, source as reference.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import RESULTS_DIR, require  # noqa: E402
from configs.studies import sample_path  # noqa: E402

N_NOTES, CHARS, MIN_MARKERS = 60, 6000, 3
HEDGE_WORDS = re.compile(
    r"\b(possible|possibly|may|might|could|likely|probable|suspected|"
    r"suggestive of|consistent with|cannot rule out|concerning for|appears to|"
    r"questionable|mild|moderate)\b", re.IGNORECASE)


def strip_all_hedges(text):
    """Delete every detected hedge span; the rest of the text is unchanged."""
    from src.hedge_screener import find_hedges
    for m in sorted(find_hedges(text), key=lambda m: -m.start):
        text = text[:m.start] + text[m.end:]
    return text


def content_cut(text):
    """Keep every sentence with a hedge word; drop every second sentence without one."""
    kept, plain = [], 0
    for s in re.split(r"(?<=[.!?])\s+", text):
        if HEDGE_WORDS.search(s):
            kept.append(s)
        else:
            plain += 1
            if plain % 2 == 0:
                continue
            kept.append(s)
    return " ".join(kept)


def main():
    from rouge_score import rouge_scorer
    from src.efs import compute_efs, compute_efs_greedy
    from src.hedge_screener import hedge_density
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def score(src, out):
        n_src = max(hedge_density(src)["total_hedges"], 1)
        return {"efs": compute_efs(src, out, domain="clinical")["efs_score"],
                "hedge_retention": hedge_density(out)["total_hedges"] / n_src,
                "rouge_l": scorer.score(src, out)["rougeL"].fmeasure}

    notes = pd.read_parquet(require(sample_path("full_sample.parquet"), "seed-123 clinical sample", "EE_DATA_DIR"))
    rows = {"identity": [], "dehedge": [], "content_cut": []}
    greedy = []
    for text in notes["discharge_text"].head(N_NOTES):
        src = str(text)[:CHARS]
        if hedge_density(src)["total_hedges"] < MIN_MARKERS:
            continue
        stripped = strip_all_hedges(src)
        rows["identity"].append(score(src, src))
        rows["dehedge"].append(score(src, stripped))
        rows["content_cut"].append(score(src, content_cut(src)))
        greedy.append(compute_efs_greedy(src, stripped, domain="clinical")["efs_score"])

    v = {"F2.n_notes": len(greedy)}
    for cond, lst in rows.items():
        means = pd.DataFrame(lst).mean()
        for k in ("efs", "hedge_retention", "rouge_l"):
            v[f"F2.{cond}.{k}"] = means[k]
    v["F2.dehedge.greedy_efs"] = sum(greedy) / len(greedy)
    for k, x in v.items():
        print(f"{k:32s} {x:.4f}")
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump({k: (int(x) if k == "F2.n_notes" else float(x)) for k, x in v.items()},
              open(RESULTS_DIR / "analysis" / "metric_sensitivity.json", "w"), indent=1)
    print(f"wrote {RESULTS_DIR / 'analysis' / 'metric_sensitivity.json'}")


if __name__ == "__main__":
    main()
