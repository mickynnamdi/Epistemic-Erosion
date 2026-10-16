"""Human-grounded erosion: do model summaries keep the uncertainty radiologists marked in RadGraph?

Paper: Sec. 6.7, Appendix J.2 and Table A9. 200 radiologist-annotated chest-radiograph reports with
>= 1 observation labelled uncertain (OBS-U; 680 per model), summarized by BioMistral-7B,
Llama-3.1-8B, Med42-v2-8B and Mistral-7B; a lexicon-blind Llama-3.3-70B judge labels each
observation PRESERVED / ESCALATED / OMITTED (2,720 judgements). Every seventh item is judged again
with the source report in place of the summary (389 observations, 161 reports): preserved 83.8%
(source) vs 54.5% (summary), -29.3 points, exact McNemar p 1.2e-25, report-clustered bootstrap
95% CI [23.3, 35.3] on the drop (10,000 resamples); hardened among mentions 12.8% of 374 vs
36.1% of 332 (ratio 2.816, CI [2.130, 3.954]; difference CI [17.5, 29.2]); 329 mentioned in
both arms: 14.3% vs 36.5%, p 2.5e-16; the secondary rows of Table A9 (observations the report
also states definitely elsewhere removed); survival over all 2,720 judgements 0.510
[0.465, 0.555], per model 0.512 / 0.540 / 0.513 / 0.476; judge error on the source itself
(omitted) 3.9%; every model's paired drop McNemar p <= 2e-4.
Inputs: RadGraph 1.0.0 train.json and dev.json under EE_RADGRAPH_ROOT; access: PhysioNet
credentialed. summarize: GPU (four 7-8B models, bf16). judge: GPU (Llama-3.3-70B, bf16, about
140 GB). analyze: CPU.
Outputs: RESULTS_DIR/radgraph/radgraph_pairs.json (summarize), RESULTS_DIR/radgraph/judgements.csv
(judge; both contain RadGraph text, keep them local), RESULTS_DIR/analysis/radgraph_erosion.json
(analyze; check ids J2.* and A9.*).
Usage:
    python scripts/validation/radgraph_erosion.py --stage summarize
    python scripts/validation/radgraph_erosion.py --stage judge
    python scripts/validation/radgraph_erosion.py --stage analyze [--judgements path.csv --pairs path.json]

Unit of analysis: one radiologist OBS-U entity occurrence judged against one (model, report)
summary; the same span text can occur twice in a report and gives two rows. The control item j is
output item 7*j with the summary swapped for the source, so pairing is positional (CSV row
n_output + j with row 7*j); it is verified against the pairs file and on (model, report, obs,
sentence) before use. The secondary subset drops observations whose report also contains an
OBS-DP or OBS-DA entity with the same span text (case-insensitive), re-derived from RadGraph.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.models import resolve  # noqa: E402
from configs.paths import MODELS_ROOT, RADGRAPH_ROOT, RESULTS_DIR, require  # noqa: E402

OUT_DIR = RESULTS_DIR / "radgraph"
DEFAULT_PAIRS = OUT_DIR / "radgraph_pairs.json"
DEFAULT_JUDGEMENTS = OUT_DIR / "judgements.csv"

# Summarizers, in the paper's run order. The order fixes the item order, and therefore
# which items fall on the every-seventh control stride.
SUMMARIZERS = ["llama-3.1-8b", "mistral-7b", "biomistral-7b", "med42-8b"]
SUM_SYS = "You are a radiologist writing the IMPRESSION section of a chest radiograph report."
SUM_PROMPT = ("Summarize the following radiology report into a concise IMPRESSION section. "
              "Preserve the clinical content faithfully.\n\nREPORT:\n{TEXT}\n\nIMPRESSION:")
SUM_BATCH, SUM_MAX_NEW, SUM_MAX_LEN = 8, 220, 1536

JUDGE = "llama-3.3-70b"
JUDGE_SYS = "You are a precise clinical evaluator. Answer with exactly one word."
JUDGE_PROMPT = """A radiologist wrote this sentence in a chest radiograph report:

"{SENT}"

In that sentence the finding "{OBS}" is stated as UNCERTAIN (not confirmed).

Here is an AI-generated summary of the same report:

\"\"\"
{OUT}
\"\"\"

How does the summary treat the finding "{OBS}"?
Answer with exactly one word:
  PRESERVED  - the summary mentions it and still presents it as uncertain
  ESCALATED  - the summary mentions it but presents it as certain or definite
  OMITTED    - the summary does not mention it

Answer:"""
JUDGE_BATCH, JUDGE_MAX_NEW, JUDGE_PAD_LEN, JUDGE_OUT_CHARS = 8, 24, 1280, 3000
LABELS = ("PRESERVED", "ESCALATED", "OMITTED")
MENTIONED = ("PRESERVED", "ESCALATED")

CONTROL_STRIDE = 7
N_BOOT, BOOT_SEED = 10000, 20260823
CSV_COLUMNS = ["model", "report_id", "obs", "sent", "out", "arm", "label", "def_dup"]


# ------------------------------------------------------------------------------ RadGraph
def radgraph_records():
    """{report_id: record} for the radiologist-annotated train and dev splits, in file order."""
    root = require(RADGRAPH_ROOT, "RadGraph 1.0.0", "EE_RADGRAPH_ROOT")
    recs = {}
    for split in ("train", "dev"):
        recs.update(json.load(open(require(root / f"{split}.json", f"RadGraph {split}.json", "EE_RADGRAPH_ROOT"))))
    return recs


def load_reports(recs):
    """Reports with >= 1 OBS-U entity; each observation carries the sentence that contains it."""
    out = []
    for k, r in recs.items():
        toks = r["text"].split()
        obs = []
        for e in r["entities"].values():
            if e["label"] != "OBS-U":
                continue
            span = " ".join(toks[e["start_ix"]:e["end_ix"] + 1])
            s = max([i for i, t in enumerate(toks[:e["start_ix"]]) if t.endswith(".")] + [-1]) + 1
            en = next((i for i, t in enumerate(toks) if i >= e["end_ix"] and t.endswith(".")), len(toks) - 1)
            obs.append({"observation": span, "source_sentence": " ".join(toks[s:en + 1])})
        if obs:
            out.append({"report_id": k, "text": r["text"], "uncertain_observations": obs})
    return out


def definite_spans(recs):
    """{report_id: lower-cased span texts of OBS-DP / OBS-DA entities}."""
    out = {}
    for rid, r in recs.items():
        toks = r["text"].split()
        out[rid] = {" ".join(toks[e["start_ix"]:e["end_ix"] + 1]).lower()
                    for e in r["entities"].values() if e["label"] in ("OBS-DP", "OBS-DA")}
    return out


def build_items(pairs):
    """Judgement items: every (pair, observation), then every seventh one against the source."""
    items = []
    for p in pairs:
        for o in p["uncertain_observations"]:
            items.append({"model": p["model"], "report_id": p["report_id"], "obs": o["observation"],
                          "sent": o["source_sentence"], "out": p["output"], "arm": "output"})
    src = {}
    for q in pairs:
        src.setdefault(q["report_id"], q["source"])
    items += [dict(it, out=src[it["report_id"]], arm="control_source") for it in items[::CONTROL_STRIDE]]
    return items


# ----------------------------------------------------------------------------- summarize
def stage_summarize(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reports = load_reports(radgraph_records())
    n_obs = sum(len(r["uncertain_observations"]) for r in reports)
    print(f"{len(reports)} reports with >= 1 OBS-U; {n_obs} radiologist-marked uncertain observations")
    assert len(reports) > 150 and n_obs > 500, "RadGraph did not load as expected"

    pairs = []
    for name in args.models:
        path, rev = resolve(name, MODELS_ROOT)
        kw = {"revision": rev} if rev else {}
        tok = AutoTokenizer.from_pretrained(path, **kw)
        tok.pad_token = tok.pad_token or tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto", **kw).eval()
        print(f"  loaded {name}", flush=True)

        prompts = []
        for r in reports:
            text = SUM_SYS + "\n\n" + SUM_PROMPT.format(TEXT=r["text"])
            try:
                prompts.append(tok.apply_chat_template([{"role": "user", "content": text}],
                                                       tokenize=False, add_generation_prompt=True))
            except Exception:  # models without a chat template
                prompts.append(text)
        outs = []
        for i in range(0, len(prompts), SUM_BATCH):
            enc = tok(prompts[i:i + SUM_BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=SUM_MAX_LEN).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=SUM_MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in g]
        for r, o in zip(reports, outs):
            pairs.append({"model": name, "report_id": r["report_id"], "source": r["text"], "output": o,
                          "uncertain_observations": r["uncertain_observations"]})
        del model
        torch.cuda.empty_cache()

    args.pairs.parent.mkdir(parents=True, exist_ok=True)
    json.dump(pairs, open(args.pairs, "w"))
    print(f"wrote {len(pairs)} (model, report) pairs to {args.pairs}")


# --------------------------------------------------------------------------------- judge
def parse_label(raw):
    r = raw.strip().upper()
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", r):
            return lab
    return None


def stage_judge(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pairs = json.load(open(require(args.pairs, "summaries (run --stage summarize)", "--pairs")))
    items = build_items(pairs)
    print(f"{len(items)} judgements ({len(items) - sum(len(p['uncertain_observations']) for p in pairs)} controls)")

    path, rev = resolve(JUDGE, MODELS_ROOT)
    kw = {"revision": rev} if rev else {}
    tok = AutoTokenizer.from_pretrained(path, **kw)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto", **kw).eval()
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": JUDGE_SYS},
         {"role": "user", "content": JUDGE_PROMPT.format(SENT=it["sent"], OBS=it["obs"], OUT=it["out"][:JUDGE_OUT_CHARS])}],
        tokenize=False, add_generation_prompt=True) for it in items]
    labels = []
    for i in range(0, len(prompts), JUDGE_BATCH):
        enc = tok(prompts[i:i + JUDGE_BATCH], return_tensors="pt", padding="max_length",
                  max_length=JUDGE_PAD_LEN, truncation=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=JUDGE_MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        labels += [parse_label(tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True)) for x in g]
        if (i // JUDGE_BATCH) % 50 == 0:
            print(f"  {min(i + JUDGE_BATCH, len(prompts))}/{len(prompts)}", flush=True)

    definite = definite_spans(radgraph_records())
    args.judgements.parent.mkdir(parents=True, exist_ok=True)
    with open(args.judgements, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for it, lab in zip(items, labels):
            w.writerow(dict(it, label=lab or "", def_dup=it["obs"].lower() in definite[it["report_id"]]))
    print(f"wrote {args.judgements}; {sum(lab is None for lab in labels)} replies could not be parsed")


# ------------------------------------------------------------------------------- analyze
def mcnemar_exact(b, c):
    """Two-sided exact McNemar: binomial test on the discordant pairs."""
    return 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)


def cluster_bootstrap(cluster_ids, stat_fn, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile CI from resampling report ids with replacement; stat_fn(idx) may return None."""
    by_cluster = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        by_cluster[cid].append(i)
    idx_lists = [np.asarray(by_cluster[k], dtype=np.int64) for k in sorted(by_cluster)]
    rng = np.random.default_rng(seed)
    K, vals = len(idx_lists), []
    for _ in range(n_boot):
        pick = rng.integers(0, K, size=K)
        v = stat_fn(np.concatenate([idx_lists[j] for j in pick]))
        if v is not None:
            vals.append(v)
    lo, hi = np.percentile(np.asarray(vals, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def load_rows(path):
    csv.field_size_limit(10 ** 7)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "empty judgements file"
    missing = {"model", "report_id", "obs", "sent", "out", "arm", "label"} - set(rows[0])
    assert not missing, f"judgements file lacks columns {missing}"
    bad = sum(r["label"] not in LABELS for r in rows)
    assert not bad, f"{bad} judgements have no parsed label; every item must be labelled for the paired design"
    assert {r["arm"] for r in rows} == {"output", "control_source"}, "arm vocabulary"
    return rows


def check_pairing(rows, pairs_path):
    """Verify that control row j is output row 7*j with only the judged text swapped."""
    n_out = sum(r["arm"] == "output" for r in rows)
    assert [r["arm"] for r in rows[:n_out]] == ["output"] * n_out, "output rows are not one contiguous block"
    assert len(rows) - n_out == -(-n_out // CONTROL_STRIDE), "control count is not ceil(n_output / 7)"
    out_rows, ctl_rows = rows[:n_out], rows[n_out:]
    key = lambda r: (r["model"], r["report_id"], r["obs"], r["sent"])  # noqa: E731
    assert all(key(out_rows[CONTROL_STRIDE * j]) == key(c) for j, c in enumerate(ctl_rows)), "positional pairing failed"
    if pairs_path is not None and Path(pairs_path).exists():
        items = build_items(json.load(open(pairs_path)))
        fields = ("model", "report_id", "obs", "sent", "out", "arm")
        assert len(items) == len(rows) and all(str(it[k]) == r[k] for it, r in zip(items, rows) for k in fields), \
            "judgement rows do not follow the item order rebuilt from the pairs file"
        print(f"pairing verified against {pairs_path} ({len(items)} items)")
    else:
        print("pairs file not given or absent: pairing verified on keys only")
    return out_rows, ctl_rows


def attach_def_dup(rows):
    """Re-derive def_dup from RadGraph; if the file carries the column, it must agree exactly."""
    if not (RADGRAPH_ROOT / "train.json").exists():
        assert "def_dup" in rows[0], "no def_dup column and RadGraph not available (set EE_RADGRAPH_ROOT)"
        print("RadGraph not available: using the def_dup column of the judgements file as is")
        return
    definite = definite_spans(radgraph_records())
    derived = [r["obs"].lower() in definite[r["report_id"]] for r in rows]
    if "def_dup" in rows[0]:
        mism = sum(d != (r["def_dup"] == "True") for d, r in zip(derived, rows))
        assert mism == 0, f"re-derived def_dup disagrees with the file on {mism} rows"
    for d, r in zip(derived, rows):
        r["def_dup"] = "True" if d else "False"


def paired_block(pairs):
    """Source arm vs model arm on paired items. pairs: [(output_row, control_row)]."""
    lab_o = np.array([o["label"] for o, _ in pairs])
    lab_s = np.array([c["label"] for _, c in pairs])
    src_p, out_p = lab_s == "PRESERVED", lab_o == "PRESERVED"
    src_e, out_e = lab_s == "ESCALATED", lab_o == "ESCALATED"
    src_m, out_m = np.isin(lab_s, MENTIONED), np.isin(lab_o, MENTIONED)
    rid = [o["report_id"] for o, _ in pairs]
    n = len(pairs)
    b, c = int((src_p & ~out_p).sum()), int((~src_p & out_p).sum())
    eb, ec = int((src_e & ~out_e).sum()), int((~src_e & out_e).sum())

    def cond_diff(idx):
        sm, om = src_m[idx], out_m[idx]
        if sm.sum() == 0 or om.sum() == 0:
            return None
        return float(out_e[idx][om].mean() - src_e[idx][sm].mean())

    def cond_ratio(idx):
        sm, om = src_m[idx], out_m[idx]
        if sm.sum() == 0 or om.sum() == 0:
            return None
        s = src_e[idx][sm].mean()
        return None if s == 0 else float(out_e[idx][om].mean() / s)

    src_mn, out_mn = int(src_m.sum()), int(out_m.sum())
    src_h, out_h = int((src_e & src_m).sum()), int((out_e & out_m).sum())
    bm = src_m & out_m
    return {
        "n": n, "n_reports": len(set(rid)),
        "source_preserved": src_p.mean(), "model_preserved": out_p.mean(), "b": b, "c": c,
        "preserved_p": mcnemar_exact(b, c),
        "drop_ci": cluster_bootstrap(rid, lambda idx: float(src_p[idx].mean() - out_p[idx].mean())),
        "source_escalated": src_e.mean(), "model_escalated": out_e.mean(),
        "escalated_p": mcnemar_exact(eb, ec),
        "rise_ci": cluster_bootstrap(rid, lambda idx: float(out_e[idx].mean() - src_e[idx].mean())),
        "source_omitted": 1 - src_m.mean(),
        "source_mentioned_n": src_mn, "model_mentioned_n": out_mn,
        "source_hardening": src_h / src_mn, "model_hardening": out_h / out_mn,
        "hardening_ratio": (out_h / out_mn) / (src_h / src_mn) if src_h else float("nan"),
        "hardening_diff_ci": cluster_bootstrap(rid, cond_diff),
        "hardening_ratio_ci": cluster_bootstrap(rid, cond_ratio),
        "both_mentioned_n": int(bm.sum()),
        "both_source_hardening": float((src_e & bm).sum() / bm.sum()),
        "both_model_hardening": float((out_e & bm).sum() / bm.sum()),
        "both_p": mcnemar_exact(int((src_e & ~out_e & bm).sum()), int((~src_e & out_e & bm).sum())),
        "source_chars": float(np.mean([len(c["out"]) for _, c in pairs])),
        "model_chars": float(np.mean([len(o["out"]) for o, _ in pairs])),
    }


def table_a9_rows(v, name, s):
    pct = lambda x: 100 * float(x)  # noqa: E731
    v[f"A9.preserved.{name}.n"] = s["n"]
    v[f"A9.preserved.{name}.source_pct"] = pct(s["source_preserved"])
    v[f"A9.preserved.{name}.model_pct"] = pct(s["model_preserved"])
    v[f"A9.preserved.{name}.delta_pts"] = pct(s["model_preserved"] - s["source_preserved"])
    v[f"A9.preserved.{name}.p"] = s["preserved_p"]
    v[f"A9.preserved.{name}.ci_lo"] = -pct(s["drop_ci"][1])
    v[f"A9.preserved.{name}.ci_hi"] = -pct(s["drop_ci"][0])
    v[f"A9.escalated.{name}.n"] = s["n"]
    v[f"A9.escalated.{name}.source_pct"] = pct(s["source_escalated"])
    v[f"A9.escalated.{name}.model_pct"] = pct(s["model_escalated"])
    v[f"A9.escalated.{name}.delta_pts"] = pct(s["model_escalated"] - s["source_escalated"])
    v[f"A9.escalated.{name}.p"] = s["escalated_p"]
    v[f"A9.escalated.{name}.ci_lo"] = pct(s["rise_ci"][0])
    v[f"A9.escalated.{name}.ci_hi"] = pct(s["rise_ci"][1])
    v[f"A9.hardened_mentioned.{name}.n_source"] = s["source_mentioned_n"]
    v[f"A9.hardened_mentioned.{name}.n_model"] = s["model_mentioned_n"]
    v[f"A9.hardened_mentioned.{name}.source_pct"] = pct(s["source_hardening"])
    v[f"A9.hardened_mentioned.{name}.model_pct"] = pct(s["model_hardening"])
    v[f"A9.hardened_mentioned.{name}.delta_pts"] = pct(s["model_hardening"] - s["source_hardening"])
    v[f"A9.hardened_mentioned.{name}.ratio"] = float(s["hardening_ratio"])
    v[f"A9.hardened_mentioned.{name}.ratio_ci_lo"], v[f"A9.hardened_mentioned.{name}.ratio_ci_hi"] = \
        s["hardening_ratio_ci"]
    v[f"A9.hardened_mentioned.{name}.delta_ci_lo"] = pct(s["hardening_diff_ci"][0])
    v[f"A9.hardened_mentioned.{name}.delta_ci_hi"] = pct(s["hardening_diff_ci"][1])


def stage_analyze(args):
    rows = load_rows(require(args.judgements, "judgements (run --stage judge)", "--judgements"))
    out_rows, ctl_rows = check_pairing(rows, args.pairs)
    attach_def_dup(rows)
    pairs_all = [(out_rows[CONTROL_STRIDE * j], c) for j, c in enumerate(ctl_rows)]
    pairs_clean = [(o, c) for o, c in pairs_all if o["def_dup"] == "False"]

    v = {"J2.n_reports": len({r["report_id"] for r in out_rows}), "J2.n_judgements": len(out_rows)}
    prim, sec = paired_block(pairs_all), paired_block(pairs_clean)
    table_a9_rows(v, "primary", prim)
    table_a9_rows(v, "secondary", sec)

    v["J2.control.n"], v["J2.control.n_reports"] = prim["n"], prim["n_reports"]
    v["J2.control.source_preserved_pct"] = 100 * float(prim["source_preserved"])
    v["J2.control.model_preserved_pct"] = 100 * float(prim["model_preserved"])
    v["J2.control.drop_pts"] = 100 * float(prim["source_preserved"] - prim["model_preserved"])
    v["J2.control.drop_p"] = prim["preserved_p"]
    v["J2.control.drop_ci_lo"], v["J2.control.drop_ci_hi"] = (100 * x for x in prim["drop_ci"])
    v["J2.control.omitted_pct"] = 100 * float(prim["source_omitted"])
    v["J2.both_mentioned.n"] = prim["both_mentioned_n"]
    v["J2.both_mentioned.source_pct"] = 100 * prim["both_source_hardening"]
    v["J2.both_mentioned.model_pct"] = 100 * prim["both_model_hardening"]
    v["J2.both_mentioned.p"] = prim["both_p"]
    v["J2.mean_chars.source"], v["J2.mean_chars.model"] = prim["source_chars"], prim["model_chars"]

    pres = np.array([r["label"] == "PRESERVED" for r in out_rows])
    v["J2.survival"] = float(pres.mean())
    v["J2.survival_ci_lo"], v["J2.survival_ci_hi"] = cluster_bootstrap(
        [r["report_id"] for r in out_rows], lambda idx: float(pres[idx].mean()))
    worst_p = 0.0
    for m in sorted({r["model"] for r in out_rows}):
        sel = np.array([r["model"] == m for r in out_rows])
        v[f"J2.survival.{m}"] = float(pres[sel].mean())
        v[f"J2.n.{m}"] = int(sel.sum())
        sub = [(o, c) for o, c in pairs_all if o["model"] == m]
        b = sum(c["label"] == "PRESERVED" and o["label"] != "PRESERVED" for o, c in sub)
        cc = sum(c["label"] != "PRESERVED" and o["label"] == "PRESERVED" for o, c in sub)
        v[f"J2.paired_drop_p.{m}"] = mcnemar_exact(b, cc)
        worst_p = max(worst_p, v[f"J2.paired_drop_p.{m}"])
    v["J2.paired_drop_p_max"] = worst_p

    out = RESULTS_DIR / "analysis" / "radgraph_erosion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: (float(x) if isinstance(x, np.floating) else x) for k, x in v.items()}, open(out, "w"), indent=2)
    for k, x in v.items():
        print(f"{k:50s} {x:.6g}" if isinstance(x, float) else f"{k:50s} {x}")
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["summarize", "judge", "analyze"], required=True)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                    help="summaries JSON: written by summarize, read by judge and analyze (default: %(default)s)")
    ap.add_argument("--judgements", type=Path, default=DEFAULT_JUDGEMENTS,
                    help="judge labels CSV: written by judge, read by analyze (default: %(default)s)")
    ap.add_argument("--models", nargs="+", default=SUMMARIZERS,
                    help="summarize only: configs.models keys; the paper uses all four in the default order")
    args = ap.parse_args()
    {"summarize": stage_summarize, "judge": stage_judge, "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
