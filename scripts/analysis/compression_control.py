"""Compression-controlled comparison of hedge hardening in human reference summaries and model outputs.

Paper: Appendix F.4. Legal: OLS coefficient of is_model on hardening 0.025 (p = 0.057,
95% CI -0.001..0.050) controlling for compression; overlap band of 180 human and 1,152 model
summaries, hardening 0.082 (human) vs 0.107 (model), one-sided Mann-Whitney p = 0.024.
Scientific: coefficient 0.117 (p = 1.6e-11); compression 0.021 (human) vs 0.317 (model);
only 4 human summaries in the overlap band.
Inputs: EE_DATA_DIR/{legal,scientific}_sample.parquet (reference_summary / reference_tldr),
        results/scores/crossdomain.csv (scripts/score_generations.py --study crossdomain),
        EE_OUTPUTS_DIR/cross_domain/{legal,scientific}/<model>/*.json (for model compression, see NOTE);
        access: public. CPU only.
Outputs: results/analysis/compression_control.json.
Usage:
    python scripts/analysis/compression_control.py

Definitions: hardening = (escalation + severity_warp) / max(|H(S)|, 1); compression = output words /
source words. Human reference summaries are scored with EFS against the full (untruncated) source.
Model: OLS hardening ~ const + is_model + compression. Overlap band: the compression range shared by
the human and model summaries; one-sided Mann-Whitney (model > human) when >= 10 human summaries fall
in the band.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import OUTPUTS_DIR, RESULTS_DIR, require  # noqa: E402
from configs.studies import sample_path  # noqa: E402

DOMAINS = {"legal": ("legal_sample.parquet", "case_id", "reference_summary"),
           "scientific": ("scientific_sample.parquet", "paper_id", "reference_tldr")}
MODELS = ["llama-3.1-8b", "med42-8b", "mistral-7b"]


def words(t):
    return len(t.split())


def human_rows(df, domain, ref_col):
    from src.efs import compute_efs
    rows = []
    for src, summ in zip(df["source_text"], df[ref_col]):
        src, summ = str(src or ""), str(summ or "")
        if not src.strip() or not summ.strip():
            continue
        r = compute_efs(src, summ, domain=domain)
        bt = r["by_type"]
        rows.append({"harden": (bt.get("escalation", 0) + bt.get("severity_warp", 0)) / max(r["source_hedges"], 1),
                     "compression": words(summ) / max(words(src), 1), "is_model": 0})
    return pd.DataFrame(rows)


def model_compression(df, domain, id_col):
    """Output/source word ratio per (doc_id, model, task).

    NOTE: as in the paper, compression is read from the generations in
    outputs/cross_domain/<domain>/, an earlier run of the same three models on the same documents,
    whereas hardening comes from the scored generations in outputs/fullscale/<domain>/ (the texts
    differ for most pairs).
    """
    sources = {str(i): str(t) for i, t in zip(df[id_col], df["source_text"])}
    # The earlier run is not produced by scripts/generate/generate.py; without it, compression is
    # measured on the same generations as hardening, which gives
    # close but not identical numbers.
    run = "cross_domain" if (OUTPUTS_DIR / "cross_domain" / domain).exists() else "fullscale"
    if run == "fullscale":
        print(f"  note: {OUTPUTS_DIR / 'cross_domain' / domain} not found; compression measured on fullscale/")
    comp = {}
    for m in MODELS:
        gdir = require(OUTPUTS_DIR / run / domain / m, f"{domain} generations of {m}", "EE_OUTPUTS_DIR")
        for jf in glob.glob(str(gdir / "*.json")):
            d = json.load(open(jf))
            if d.get("error") and str(d["error"]) != "None":
                continue
            src = sources.get(str(d.get("doc_id")))
            if not src or not d.get("output_text"):
                continue
            comp[(str(d["doc_id"]), m, d.get("task"))] = words(d["output_text"]) / max(words(src), 1)
    return comp


def analyse(domain, scores):
    sample, id_col, ref_col = DOMAINS[domain]
    df = pd.read_parquet(require(sample_path(sample), f"{domain} sample", "EE_DATA_DIR"))
    h = human_rows(df, domain, ref_col)
    d = scores[scores.domain == domain].copy()
    d["harden"] = (d.escalation + d.severity_warp) / d.source_hedges.clip(lower=1)
    comp = model_compression(df, domain, id_col)
    d["compression"] = [comp.get((str(r.doc_id), r.model, r.task), np.nan) for r in d.itertuples()]
    d = d.dropna(subset=["compression"])
    m = pd.DataFrame({"harden": d.harden, "compression": d.compression, "is_model": 1})
    al = pd.concat([h, m], ignore_index=True)
    ols = sm.OLS(al.harden, sm.add_constant(al[["is_model", "compression"]])).fit()
    ci = ols.conf_int().loc["is_model"]
    lo, hi = max(h.compression.min(), m.compression.min()), min(h.compression.max(), m.compression.max())
    hb = h[(h.compression >= lo) & (h.compression <= hi)]
    mb = m[(m.compression >= lo) & (m.compression <= hi)]
    p = f"F4.{domain}."
    v = {p + "n_human": len(h), p + "n_model": len(m), p + "coef_is_model": ols.params["is_model"],
         p + "coef_p": ols.pvalues["is_model"], p + "coef_ci_lo": ci.iloc[0], p + "coef_ci_hi": ci.iloc[1],
         p + "human_compression": h.compression.mean(), p + "model_compression": m.compression.mean(),
         p + "band_n_human": len(hb), p + "band_n_model": len(mb), p + "band_model_harden": mb.harden.mean()}
    if len(hb):
        v[p + "band_human_harden"] = hb.harden.mean()
    if len(hb) >= 10:
        v[p + "band_mannwhitney_p"] = stats.mannwhitneyu(mb.harden, hb.harden, alternative="greater").pvalue
    return v


def main():
    scores = pd.read_csv(require(RESULTS_DIR / "scores" / "crossdomain.csv", "cross-domain scores", "EE_RESULTS_DIR"),
                         low_memory=False)
    scores = scores[scores.status == "ok"]
    v = {}
    for domain in DOMAINS:
        v.update(analyse(domain, scores))
    for k, x in v.items():
        print(f"{k:40s} {x:.4g}")
    out = RESULTS_DIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({k: (int(x) if isinstance(x, (int, np.integer)) else float(x)) for k, x in v.items()},
              open(out / "compression_control.json", "w"), indent=1)
    print(f"wrote {out / 'compression_control.json'}")


if __name__ == "__main__":
    main()
