"""Main-text and appendix tables computed from results/scores/*.csv.

Every function returns a dict {check_id: value}; scripts/make_tables.py writes the
tables and compares each value with the paper (paper_values.py).

Conventions that the paper's numbers depend on (keep them):
  * rank-based tests (Kruskal-Wallis, Mann-Whitney, Friedman, Wilcoxon) use EFS rounded
    to 9 decimals, so that values parsed from CSV and computed in memory tie identically;
  * Table 1 EFS is the pooled mean over the 3 seeds; its +- is the std (ddof=1) of the 3
    per-seed means; the per-model scale correlation uses the mean of the 3 per-seed means;
  * Table 2 EFS is the mean of the 3 per-seed means +- their std (ddof=1);
  * Cohen's d uses the pooled standard deviation (ddof=1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.models import MODELS, STUDY_11  # noqa: E402
from configs.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

TYPES = ["omission", "escalation", "fabrication", "scope_shift", "severity_warp", "attenuation"]
MED = {m for m in STUDY_11 if MODELS[m]["medical"]}
M3 = ["llama-3.1-8b", "med42-8b", "mistral-7b"]
LLAMA_MED42 = ["llama-3.1-8b", "med42-8b"]


def r9(x):
    return np.round(np.asarray(x, dtype=float), 9)


def load(study, aligner="hungarian"):
    suffix = "_greedy" if aligner == "greedy" else ""
    df = pd.read_csv(RESULTS_DIR / "scores" / f"{study}{suffix}.csv", low_memory=False)
    return df[df["status"] == "ok"].copy()


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def mwu_p(a, b):
    return stats.mannwhitneyu(r9(a), r9(b), alternative="two-sided").pvalue


# --------------------------------------------------------------------------- Table 1, A18, Section 6.1-6.2
def table1(depth):
    rows, v = [], {}
    for m in STUDY_11:
        g = depth[depth.model == m]
        seed_means = g.groupby("seed").efs.mean()
        tot = g[TYPES].sum()
        row = {"model": m, "n": len(g), "efs": g.efs.mean(), "efs_std": seed_means.std(ddof=1),
               "dist_per_gen": g.n_distortions.mean(), **{f"{t}_pct": 100 * tot[t] / tot.sum() for t in TYPES}}
        rows.append(row)
        for k in ("efs", "efs_std", "dist_per_gen"):
            v[f"T1.{m}.{k}"] = row[k]
        for t in TYPES:
            v[f"T1.{m}.{t}_pct"] = row[f"{t}_pct"]
    tab = pd.DataFrame(rows).sort_values("efs", ascending=False)
    v["S6.1.mean_efs_3seeds"] = depth.efs.mean()
    v["S6.1.efs_range_min"], v["S6.1.efs_range_max"] = tab.efs.min(), tab.efs.max()
    sd = depth.groupby(["model", "seed"]).efs.mean().groupby("model").std(ddof=1)
    v["S6.1.crossseed_std_min"], v["S6.1.crossseed_std_max"], v["S6.1.crossseed_std_mean"] = sd.min(), sd.max(), sd.mean()
    tot = depth[TYPES].sum()
    a18 = pd.DataFrame({"type": TYPES, "count": [int(tot[t]) for t in TYPES],
                        "pct": [100 * tot[t] / tot.sum() for t in TYPES],
                        "mean_per_gen": [tot[t] / len(depth) for t in TYPES]})
    for _, r in a18.iterrows():
        v[f"A18.{r['type']}.count"], v[f"A18.{r['type']}.pct"], v[f"A18.{r['type']}.per_gen"] = r["count"], r["pct"], r["mean_per_gen"]
    v["A18.total"], v["A18.total_per_gen"] = int(tot.sum()), tot.sum() / len(depth)
    return tab, a18, v


def table_a1(depth, breadth, xd):
    v = {}
    h = stats.kruskal(*[r9(g.efs) for _, g in depth.groupby("model")])
    v["A1.model_H"], v["A1.model_p"] = h.statistic, h.pvalue
    h = stats.kruskal(*[r9(g.efs) for _, g in depth.groupby("task")])
    v["A1.task_H"], v["A1.task_p"] = h.statistic, h.pvalue
    if breadth is not None:   # the specialty test is the only one that needs the breadth study
        spec = pd.read_parquet(DATA_DIR / "full_sample_1000.parquet")[["hadm_id", "specialty"]]
        b = breadth.assign(hadm_id=breadth.doc_id.astype(int)).merge(spec, on="hadm_id")
        h = stats.kruskal(*[r9(g.efs) for _, g in b.groupby("specialty")])
        v["A1.specialty_H"], v["A1.specialty_p"] = h.statistic, h.pvalue
    params = depth.model.map({m: MODELS[m]["params_b"] for m in STUDY_11})
    r = stats.pearsonr(params, depth.efs)
    v["A1.scale_gen_r"], v["A1.scale_gen_p"] = r.statistic, r.pvalue
    seed_mean = depth.groupby(["model", "seed"]).efs.mean().groupby("model").mean()
    r = stats.pearsonr([MODELS[m]["params_b"] for m in seed_mean.index], seed_mean.values)
    v["A1.scale_model_r"], v["A1.scale_model_p"] = r.statistic, r.pvalue
    med, gen = depth[depth.model.isin(MED)].efs, depth[~depth.model.isin(MED)].efs
    u = stats.mannwhitneyu(r9(med), r9(gen), alternative="two-sided")
    v["A1.med_U"], v["A1.med_p"], v["S6.1.med_mean"], v["S6.1.gen_mean"] = u.statistic, u.pvalue, med.mean(), gen.mean()
    cl = depth[(depth.seed == 123) & depth.model.isin(M3)]
    h = stats.kruskal(r9(cl.efs), r9(xd[xd.domain == "legal"].efs), r9(xd[xd.domain == "scientific"].efs))
    v["A1.domain_H"], v["A1.domain_p"] = h.statistic, h.pvalue
    return v


def appendix_b_clusters(depth, xd):
    """Cluster-level re-analysis (Appendix B)."""
    v = {}
    d = depth.assign(e=r9(depth.efs))
    w = d.pivot_table(index=["doc_id", "task", "seed"], columns="model", values="e").dropna()
    f = stats.friedmanchisquare(*[w[c].values for c in w.columns])
    v["B.friedman_blocks_n"], v["B.friedman_blocks_chi2"] = len(w), f.statistic
    # episode-level means of the 9-dp-rounded scores (the paper's procedure; the statistic is sensitive to
    # floating-point ties, e.g. rounding the means again gives a slightly different chi2)
    we = d.groupby(["doc_id", "model"]).e.mean().unstack().dropna()
    f = stats.friedmanchisquare(*[we[c].values for c in we.columns])
    v["B.friedman_episodes_n"], v["B.friedman_episodes_chi2"] = len(we), f.statistic
    d["grp"] = np.where(d.model.isin(MED), "med", "gen")
    ep = d.groupby(["doc_id", "grp"]).e.mean().unstack().dropna()
    diff = (ep["med"] - ep["gen"]).values
    rng = np.random.default_rng(0)
    boot = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)]
    v["B.med_minus_gen"], v["B.med_minus_gen_ci_lo"], v["B.med_minus_gen_ci_hi"] = diff.mean(), *np.percentile(boot, [2.5, 97.5])
    v["B.med_minus_gen_wilcoxon_p"], v["B.n_episodes"] = stats.wilcoxon(diff).pvalue, len(diff)
    cl = d[(d.seed == 123) & d.model.isin(M3)].groupby("doc_id").e.mean()
    x = xd.assign(e=r9(xd.efs))
    lg = x[x.domain == "legal"].groupby("doc_id").e.mean()
    sc = x[x.domain == "scientific"].groupby("doc_id").e.mean()
    h = stats.kruskal(cl.values, lg.values, sc.values)
    v["B.domain_doc_H"], v["B.domain_doc_p"] = h.statistic, h.pvalue
    v["B.clin_vs_sci_p"] = stats.mannwhitneyu(cl, sc, alternative="two-sided").pvalue
    v["B.clin_vs_legal_p"] = stats.mannwhitneyu(cl, lg, alternative="two-sided").pvalue
    v["B.sci_vs_legal_p"] = stats.mannwhitneyu(sc, lg, alternative="two-sided").pvalue
    return v


# --------------------------------------------------------------------------- Mitigation (Tables A5-A7, App. B)
def mitigation(depth, mit, temp):
    base = depth[depth.seed == 123]
    rows, v = [], {}

    def add(label, a, b, key):
        row = {"comparison": label, "n": len(a), "efs": a.mean(), "efs_sd": a.std(ddof=1),
               "delta": a.mean() - b.mean(), "cohen_d": cohen_d(a, b), "mwu_p": mwu_p(a, b)}
        rows.append(row)
        for k in ("n", "efs", "efs_sd", "delta", "cohen_d", "mwu_p"):
            v[f"{key}.{k}"] = row[k]

    b2 = base[base.model.isin(LLAMA_MED42)].efs
    v["A5.baseline.efs"], v["A5.baseline.efs_sd"] = b2.mean(), b2.std(ddof=1)
    for s in ("prompt", "fewshot", "posthoc"):
        add(f"Llama-3.1-8B + Med42-v2-8B, {s}", mit[(mit.condition == s) & mit.model.isin(LLAMA_MED42)].efs, b2, f"A5.{s}")
    for t in ("0.3", "0.7"):
        add(f"Llama-3.1-8B, T={t} (vs the 1,152-row baseline)", temp[temp.condition == f"T={t}"].efs, b2, f"A5.T{t}")
    for m in ("mistral-7b", "gemma-4-4b"):
        bm = base[base.model == m].efs
        v[f"A7.{m}.baseline.efs"] = bm.mean()
        for s in ("prompt", "fewshot", "posthoc"):
            add(f"{m}, {s}", mit[(mit.condition == s) & (mit.model == m)].efs, bm, f"A7.{m}.{s}")
    b3 = base[base.model.isin(M3)].efs
    v["A6.baseline.efs"], v["A6.baseline.n"] = b3.mean(), len(b3)
    for s in ("strict", "cot", "combo", "constrained"):
        add(f"3 models pooled, {s}", mit[(mit.condition == s) & mit.model.isin(M3)].efs, b3, f"A6.{s}")
    # episode-paired Wilcoxon tests against the seed-123 baseline (Appendix B)
    bi = base.set_index(["doc_id", "model", "task"]).efs.pipe(r9)
    bi = pd.Series(bi, index=base.set_index(["doc_id", "model", "task"]).index)
    groups = {"llama_med42": LLAMA_MED42, "mistral": ["mistral-7b"], "gemma": ["gemma-4-4b"], "three": M3}
    for s in ("prompt", "fewshot", "posthoc", "strict", "cot", "combo", "constrained"):
        for g, ms in groups.items():
            sub = mit[(mit.condition == s) & mit.model.isin(ms)]
            if sub.empty:
                continue
            j = pd.Series(r9(sub.efs), index=sub.set_index(["doc_id", "model", "task"]).index).to_frame("x").join(bi.rename("b"), how="inner")
            if len(j):
                v[f"B.paired.{s}.{g}.delta"] = (j.x - j.b).mean()
                v[f"B.paired.{s}.{g}.p"] = stats.wilcoxon(j.x - j.b).pvalue
    return pd.DataFrame(rows), v


# --------------------------------------------------------------------------- Cross-domain (6.5, App. E, J.3, Q, F.1)
W = {"fabrication": 3.0, "escalation": 2.5, "severity_warp": 2.0, "attenuation": 1.5, "scope_shift": 1.0, "omission": 1.0}
W = {k: x / np.mean(list(W.values())) for k, x in W.items()}   # normalised to mean 1 (Appendix F.1)
CORE4 = ["escalation", "attenuation", "omission", "fabrication"]
COMMISSION = ["escalation", "attenuation", "fabrication", "scope_shift", "severity_warp"]


def _variants(g):
    H = g.source_hedges.clip(lower=1)
    return {"efs": g.efs,
            "core4": (1 - g[CORE4].sum(axis=1) / H).clip(lower=0),
            "no_omission": (1 - g[COMMISSION].sum(axis=1) / H).clip(lower=0),
            "unclamped": 1 - g[TYPES].sum(axis=1) / H,
            "efs_w": (1 - sum(W[t] * g[t] for t in TYPES) / H).clip(lower=0)}


def crossdomain(depth, xd):
    v = {}
    sets = {"clinical": depth[depth.seed == 123], "legal": xd[xd.domain == "legal"],
            "scientific": xd[xd.domain == "scientific"]}
    sets3 = dict(sets, clinical=sets["clinical"][sets["clinical"].model.isin(M3)])
    pooled_e, pooled_w = [], []
    for dom, g in sets.items():
        var = _variants(g)
        v[f"X.{dom}.efs"], v[f"X.{dom}.efs_sd"] = g.efs.mean(), g.efs.std(ddof=1)
        v[f"X.{dom}.pct_floor"] = 100 * (g.efs == 0).mean()
        v[f"X.{dom}.pct_clamp_binds"] = 100 * (var["unclamped"] < 0).mean()
        v[f"X.{dom}.unclamped"] = var["unclamped"].mean()
        v[f"X.{dom}.no_omission"] = var["no_omission"].mean()
        v[f"X.{dom}.any_escalation_pct"] = 100 * (g.escalation > 0).mean()
        v[f"X.{dom}.core4"] = var["core4"].mean()
        v[f"X.{dom}.efs_w"] = var["efs_w"].mean()
        v[f"X.{dom}.r_efs_efsw"] = stats.pearsonr(g.efs, var["efs_w"]).statistic
        mm = pd.DataFrame({"m": g.model, "e": g.efs, "w": var["efs_w"], "u": var["unclamped"]}).groupby("m").mean()
        v[f"X.{dom}.rank_rho_efsw"] = stats.spearmanr(mm.e, mm.w).statistic
        v[f"X.{dom}.rank_rho_unclamped"] = stats.spearmanr(mm.e, mm.u).statistic
        pooled_e.append(g.efs), pooled_w.append(var["efs_w"])
        if dom == "clinical":
            v["F1.per_model_r"] = stats.pearsonr(mm.e, mm.w).statistic
    v["F1.pooled_n"] = sum(len(x) for x in pooled_e)
    v["F1.pooled_r"] = stats.pearsonr(pd.concat(pooled_e), pd.concat(pooled_w)).statistic
    v["X.clinical3.efs"] = sets3["clinical"].efs.mean()
    v["X.clinical3.core4"] = _variants(sets3["clinical"])["core4"].mean()
    return v


# --------------------------------------------------------------------------- Hosted models (Table 2, A2, App. E)
HOSTED = ["Phi-4", "gpt-4o", "Mistral-Large-3", "gpt-5.4"]


def hosted(h):
    v, rows = {}, []
    for m in HOSTED:
        for dom in ("legal", "medical", "scientific"):
            s = h[(h.model == m) & (h.domain == dom)].groupby("seed").efs.mean()
            v[f"T2.{m}.{dom}"], v[f"T2.{m}.{dom}.sd"] = s.mean(), s.std(ddof=1)
            rows.append({"model": m, "domain": dom, "efs": s.mean(), "sd": s.std(ddof=1),
                         "n": int((h.model == m).sum() and ((h.model == m) & (h.domain == dom)).sum())})
    v["T2.n_scored"] = len(h)
    for dom in ("legal", "medical", "scientific"):
        g = h[h.domain == dom]
        for i, a in enumerate(HOSTED):
            for b in HOSTED[i + 1:]:
                v[f"A2.{dom}.{a}|{b}"] = mwu_p(g[g.model == a].efs, g[g.model == b].efs)
    L = h[h.domain == "legal"]
    for m in HOSTED:
        cap = L[(L.model == m)].output_tokens >= 1024
        v[f"E.legal_capped.{m}"] = int(cap.sum())
        v[f"E.legal_n.{m}"] = int((L.model == m).sum())
        if cap.any():
            g = L[L.model == m]
            v[f"E.legal_capped_efs.{m}"] = g[g.output_tokens >= 1024].efs.mean()
            v[f"E.legal_uncapped_efs.{m}"] = g[g.output_tokens < 1024].efs.mean()
    for m in HOSTED:
        v[f"E.mean_completion_tokens.{m}"] = h[h.model == m].output_tokens.mean()
    band = h[(h.output_tokens >= 200) & (h.output_tokens <= 600)]
    phi, oth = band[band.model == "Phi-4"].efs, band[band.model != "Phi-4"].efs
    u = stats.mannwhitneyu(phi, oth, alternative="two-sided")
    v.update({"E.band_n": len(band), "E.band_phi": phi.mean(), "E.band_others": oth.mean(),
              "E.band_U": u.statistic, "E.band_p": u.pvalue})
    x = h[h.output_tokens > 0]
    X = pd.concat([pd.get_dummies(x.model, drop_first=True), pd.get_dummies(x.domain, drop_first=True)], axis=1).astype(float)
    X.insert(0, "log_tokens", np.log(x.output_tokens.astype(float)))
    b, se, r2 = _ols(X, x.efs.values)
    v.update({"E.hosted_ols_n": len(x), "E.hosted_ols_coef": b, "E.hosted_ols_ci_lo": b - 1.96 * se,
              "E.hosted_ols_ci_hi": b + 1.96 * se, "E.hosted_ols_r2": r2,
              "E.hosted_loglen_r2": stats.pearsonr(np.log(x.output_tokens.astype(float)), x.efs).statistic ** 2})
    return pd.DataFrame(rows), v


def _ols(X, y):
    """OLS with intercept; returns (coef, se) of the first column of X, and R^2."""
    A = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    n, k = A.shape
    cov = resid @ resid / (n - k) * np.linalg.inv(A.T @ A)
    return beta[1], np.sqrt(cov[1, 1]), 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))


# --------------------------------------------------------------------------- Length (Table A4, App. C, E)
def length(depth, xd):
    v = {}
    v["C.depth_over_1024"] = int((depth.output_tokens > 1024).sum())
    v["C.depth_at_2048_cap"] = int((depth.output_tokens >= 2048).sum())
    v["C.legal_over_1024"] = int((xd[xd.domain == "legal"].output_tokens > 1024).sum())
    v["C.scientific_over_1024"] = int((xd[xd.domain == "scientific"].output_tokens > 1024).sum())
    g = depth[(depth.seed == 123) & (depth.source_hedges > 0)].copy()
    g["om"] = g.omission / g.source_hedges
    g["q"] = pd.qcut(g.out_words, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    tab = g.groupby("q", observed=True).agg(n=("efs", "size"), mean_words=("out_words", "mean"),
                                            omission_per_hedge=("om", "mean"), efs=("efs", "mean")).reset_index()
    for _, r in tab.iterrows():
        for k in ("n", "mean_words", "omission_per_hedge", "efs"):
            v[f"A4.{r['q']}.{k}"] = r[k]
    q5 = g[g.q == "Q5"].groupby("model").om.mean()
    v["E.Q5_om_min"], v["E.Q5_om_max"] = q5.min(), q5.max()
    X = pd.get_dummies(g.model, drop_first=True).astype(float)
    X.insert(0, "log_words", np.log(g.out_words.clip(lower=1)))
    b, se, r2 = _ols(X, g.om.values)
    t = b / se
    v.update({"E.len_ols_coef": b, "E.len_ols_ci_lo": b - 1.96 * se, "E.len_ols_ci_hi": b + 1.96 * se,
              "E.len_ols_p": 2 * stats.t.sf(abs(t), len(g) - X.shape[1] - 1), "E.len_ols_r2": r2})
    return tab, v
