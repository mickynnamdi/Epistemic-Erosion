"""Surface metrics versus EFS on the seed-123 clinical generations of the 11 open-weight models.

Paper: Section 6.3 and Figure 2. ROUGE-L against EFS r = 0.107 (p = 1.2e-17, R^2 = 0.011),
BERTScore against EFS r = -0.101 (p = 6.5e-16), N = 6,336 (11 models x 192 notes x 3 tasks).
Both metrics use the SOURCE note (first 12,000 characters, as scored by EFS) as the reference.
Inputs: EE_DATA_DIR/full_sample.parquet, EE_OUTPUTS_DIR/full_generations/<model>/*.json,
        results/scores/depth.csv (scripts/score_generations.py --study depth);
        access: PhysioNet credentialed (MIMIC-IV notes); the metrics stage needs a GPU for
        BERTScore (roberta-large, about 1.5 GB) or runs slowly on CPU.
Outputs: results/metric_blindness/metrics.csv (one row per generation: hadm_id, model, task,
         rouge_l, bert_score), results/analysis/metric_blindness.json,
         results/figures/fig2_metric_blindness.{pdf,png} with --figure.
Usage:
    python scripts/analysis/metric_blindness.py --stage metrics
    python scripts/analysis/metric_blindness.py --stage analyze [--metrics path.csv] [--figure]
    python scripts/analysis/metric_blindness.py --stage all --figure

Reproduction: ROUGE-L is deterministic; recomputed BERTScore can differ in the fourth decimal
(library and GPU numerics), which does not change the printed r. Pass --metrics path.csv to analyze a stored metrics file. Pairs are joined to EFS by (hadm_id, model, task), never by row order.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import OUTPUTS_DIR, RESULTS_DIR, require  # noqa: E402
from configs.studies import sample_path  # noqa: E402

MODELS = ["llama-3.1-8b", "med42-8b", "llama-3.3-70b", "med42-70b", "openbio-70b", "mistral-7b",
          "biomistral-7b", "medgemma-27b", "gemma-4-4b", "gemma-4-31b", "gemma-4-moe"]
MAX_SOURCE_CHARS = 12000
KEYS = ["hadm_id", "model", "task"]
METRICS_CSV = RESULTS_DIR / "metric_blindness" / "metrics.csv"


def load_pairs():
    """(hadm_id, model, task, source, output) for every seed-123 no-mitigation generation of the 11 models."""
    df = pd.read_parquet(require(sample_path("full_sample.parquet"), "seed-123 clinical sample", "EE_DATA_DIR"))
    sources = {int(h): t[:MAX_SOURCE_CHARS] for h, t in zip(df["hadm_id"], df["discharge_text"])}
    gen_dir = require(OUTPUTS_DIR / "full_generations", "seed-123 clinical generations", "EE_OUTPUTS_DIR")
    rows = []
    for model in MODELS:
        for jf in sorted((gen_dir / model).glob("*.json")):
            d = json.load(open(jf))
            if d.get("mitigation", "none") not in ("none", None):
                continue
            src, out = sources.get(int(d["hadm_id"])), d.get("output_text")
            if src and out:   # as in the paper's run: pairs with an empty side are skipped
                rows.append({"hadm_id": int(d["hadm_id"]), "model": d["model"], "task": d["task"],
                             "source": src, "output": out})
    return pd.DataFrame(rows)


def compute_metrics(pairs, batch_size):
    """ROUGE-L F1 (rougeL, Porter stemmer) and BERTScore F1 (lang='en' -> roberta-large, layer 17,
    no baseline rescaling, inputs truncated at the tokenizer limit of 512 tokens), source as reference."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    t0 = time.time()
    rouge = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(pairs.source, pairs.output)]
    print(f"ROUGE-L: {len(rouge)} pairs in {time.time() - t0:.0f}s", flush=True)

    import torch
    from bert_score import score as bscore
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    _, _, f1 = bscore(list(pairs.output), list(pairs.source), lang="en", verbose=False,
                      batch_size=batch_size, device=device)
    print(f"BERTScore ({device}): {len(f1)} pairs in {time.time() - t0:.0f}s", flush=True)
    return pairs[KEYS].assign(rouge_l=rouge, bert_score=f1.tolist())


def load_efs():
    """Seed-123 EFS of the 11 models, keyed by (hadm_id, model, task)."""
    d = pd.read_csv(require(RESULTS_DIR / "scores" / "depth.csv", "depth-study scores", "EE_RESULTS_DIR"),
                    low_memory=False)
    d = d[(d.status == "ok") & (d.seed == 123) & d.model.isin(MODELS)]
    return d.assign(hadm_id=d.doc_id.astype(int))[KEYS + ["efs"]].rename(columns={"efs": "efs_score"})


def analyze(metrics):
    m = metrics[KEYS + ["rouge_l", "bert_score"]].copy()
    m["hadm_id"] = m["hadm_id"].astype(int)
    j = m.merge(load_efs(), on=KEYS, how="left", validate="one_to_one")
    if j.efs_score.isna().any() or len(j) != 6336:
        raise SystemExit(f"expected 6,336 joined pairs with EFS, got {len(j)} ({int(j.efs_score.isna().sum())} missing)")
    v = {"S6.3.n": len(j)}
    r, p = stats.pearsonr(j.rouge_l, j.efs_score)
    v.update({"S6.3.rouge_r": r, "S6.3.rouge_p": p, "S6.3.rouge_r2": r * r})
    r, p = stats.pearsonr(j.bert_score, j.efs_score)
    v.update({"S6.3.bertscore_r": r, "S6.3.bertscore_p": p})
    return j, v


def figure(j):
    """Figure 2: EFS against ROUGE-L (left) and BERTScore (right), with least-squares lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9.5, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "axes.linewidth": 0.7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "legend.frameon": False})
    blue, red, grey = "#4477AA", "#EE6677", "#BBBBBB"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, col, label in zip(axes, ["rouge_l", "bert_score"], ["ROUGE-L", "BERTScore"]):
        ax.scatter(j[col], j.efs_score, s=3, alpha=0.12, color=blue, rasterized=True)
        r, _ = stats.pearsonr(j[col], j.efs_score)
        z = np.polyfit(j[col], j.efs_score, 1)
        x = np.linspace(j[col].min(), j[col].max(), 100)
        ax.plot(x, np.polyval(z, x), color=red, linewidth=1.2, linestyle="--")
        ax.set_xlabel(label)
        ax.set_ylabel("EFS")
        ax.text(0.95, 0.05, f"r = {r:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=grey, linewidth=0.5))
    fig.tight_layout(w_pad=2)
    out = RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig2_metric_blindness.{ext}")
    plt.close(fig)
    print(f"wrote {out / 'fig2_metric_blindness.pdf'} (+ .png)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["metrics", "analyze", "all"], default="all")
    ap.add_argument("--metrics", type=Path, default=METRICS_CSV,
                    help="per-generation ROUGE-L/BERTScore CSV with hadm_id, model, task columns "
                         "(default: the metrics stage output)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--figure", action="store_true", help="also write Figure 2")
    a = ap.parse_args()
    if a.stage in ("metrics", "all"):
        pairs = load_pairs()
        print(f"{len(pairs)} source-output pairs", flush=True)
        METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
        compute_metrics(pairs, a.batch_size).to_csv(METRICS_CSV, index=False)
        print(f"wrote {METRICS_CSV}")
    if a.stage in ("analyze", "all"):
        j, v = analyze(pd.read_csv(require(a.metrics, "per-generation metrics", "--metrics")))
        for k, x in v.items():
            print(f"{k:22s} {x:.4g}")
        out = RESULTS_DIR / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        json.dump({k: float(x) if k != "S6.3.n" else int(x) for k, x in v.items()},
                  open(out / "metric_blindness.json", "w"), indent=1)
        print(f"wrote {out / 'metric_blindness.json'}")
        if a.figure:
            figure(j)


if __name__ == "__main__":
    main()
