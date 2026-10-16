"""Clinician-proxy decision study: does reading the model output instead of the source change a triage decision?

Paper: Appendix F.5. A Claude Sonnet 4.6 proxy (API-default sampling) changes at least one of
    diagnosis confidence / urgency / action-required in 82.7% of 150 eroded pairs (urgency 65%,
    action 46%), against a 27.3% floor from querying it twice on the same source; paired exact
    McNemar b = 88, c = 5, p = 1.1e-20. A Llama-3.3-70B proxy (greedy, left-padded batches) on a
    sample stratified over EFS deciles: point-biserial r = 0.010 (p = 0.908, n = 150), change
    rate over five equal-width EFS bins 70.8 / 75.0 / 82.1 / 73.1 / 75.0%. The same text scored
    twice under left padding disagrees in 12.5% of 40 notes (quoted in Appendix F.6).
Inputs: EE_DATA_DIR/full_sample.parquet, EE_OUTPUTS_DIR/full_generations, score tables
    RESULTS_DIR/scores/depth_greedy.csv (pairs) and depth.csv (harm); access: PhysioNet
    credentialed, paid API (ANTHROPIC_API_KEY) for proxy/identity, GPU (2 x 80 GB,
    Llama-3.3-70B bf16) for harm.
Outputs: RESULTS_DIR/clinician_proxy/{pairs,decisions,identity_decisions,harm_pairs}.csv,
    harm_summary.json; RESULTS_DIR/analysis/clinician_proxy.json.
Usage:
    python scripts/decision/clinician_proxy.py --stage pairs      # 150 eroded pairs
    python scripts/decision/clinician_proxy.py --stage proxy      # Claude on source and output
    python scripts/decision/clinician_proxy.py --stage identity   # Claude twice on the source
    python scripts/decision/clinician_proxy.py --stage harm       # Llama-3.3-70B, stratified over EFS
    python scripts/decision/clinician_proxy.py --stage analyze [--decisions CSV --identity-decisions CSV
        --harm-pairs CSV --harm-summary JSON]
"""
import argparse
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from configs.paths import DATA_DIR, OUTPUTS_DIR, RESULTS_DIR, require  # noqa: E402
from judges.build_judge_set import load_scores  # noqa: E402
from judges.reference_set_judge import anthropic_client, load_causal_lm, wilson  # noqa: E402

OUT = RESULTS_DIR / "clinician_proxy"
PROMPT = """You are an ED-staff physician triaging a patient. Read the clinical text and return ONLY a JSON object with the following fields, exactly:

{
  "primary_diagnosis": "<short diagnosis>",
  "diagnosis_confidence": "low|moderate|high|definitive",
  "treatment_decision": "<short treatment plan>",
  "action_required_now": "yes|no",
  "urgency": "watch|same_day|ED|admit_now",
  "reasoning_one_line": "<one sentence>"
}

Do not include any text outside the JSON object.

Clinical text:
\"\"\"
{NOTE}
\"\"\"
"""
SYS = "You are a precise JSON-outputting evaluator. Output only valid JSON."   # local 70B proxy only
FIELDS = ["diagnosis_confidence", "urgency", "action_required_now"]
SHORT = {"diagnosis_confidence": "confidence", "urgency": "urgency", "action_required_now": "action"}
NOTE_CHARS = 8000
N_PAIRS, EFS_ERODED = 150, 0.5
N_PER_BIN, N_BINS, BATCH, MAX_NEW, N_DETERMINISM = 15, 10, 8, 200, 40


def parse_json_response(text):
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                pass
    return None


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def claude_caller(model):
    client = anthropic_client()

    def call(text):
        # NOTE: temperature is not set, so the API default applies (as in the paper's run);
        # this is why the same text queried twice can disagree (the 27.3% floor).
        msg = client.messages.create(model=model, max_tokens=400,
                                     messages=[{"role": "user", "content": PROMPT.replace("{NOTE}", text[:NOTE_CHARS])}])
        return msg.content[0].text
    return call


def generation(model, hadm, task):
    p = OUTPUTS_DIR / "full_generations" / model / f"{hadm}_{model}_{task}_none.json"
    return json.load(open(p)) if p.exists() else None


def stage_pairs(a):
    df = load_scores(a.scores)
    if "mitigation" in df:
        df = df[df["mitigation"].fillna("none").astype(str) == "none"]
    df = df[(df["efs_score"] < EFS_ERODED) & (df["source_hedges"] >= 1)]
    src_df = pd.read_parquet(require(DATA_DIR / "full_sample.parquet", "depth-study sample", "EE_DATA_DIR"))
    src_map = {int(h): t[:6000] for h, t in zip(src_df["hadm_id"], src_df["discharge_text"])}
    rows = []
    for _, r in df.sample(n=N_PAIRS, random_state=42).iterrows():
        hadm = int(r["hadm_id"])
        gen = generation(r["model"], hadm, r["task"])
        if gen is None or src_map.get(hadm) is None or gen.get("error"):
            continue
        rows.append({"hadm_id": hadm, "model": r["model"], "task": r["task"], "efs_score": r["efs_score"],
                     "source_text": src_map[hadm], "output_text": gen["output_text"]})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "pairs.csv", index=False)
    print(f"wrote {OUT / 'pairs.csv'} ({len(rows)} eroded pairs; MIMIC text, never share)")


def stage_proxy(a):
    pairs = pd.read_csv(require(OUT / "pairs.csv", "proxy pairs (--stage pairs)", "EE_RESULTS_DIR"))
    call = claude_caller(a.model)
    rows = []
    for i, p in pairs.iterrows():
        try:
            s, o = parse_json_response(call(p["source_text"])), parse_json_response(call(p["output_text"]))
        except Exception as e:  # noqa: BLE001
            print(f"  [{i + 1}] {type(e).__name__}", flush=True)
            continue
        if not (s and o):
            print(f"  [{i + 1}] parse failure", flush=True)
            continue
        rows.append({"hadm_id": p["hadm_id"], "model": p["model"], "task": p["task"], "efs": p["efs_score"],
                     **{f"src_{k}": v for k, v in s.items()}, **{f"out_{k}": v for k, v in o.items()}})
    pd.DataFrame(rows).to_csv(OUT / "decisions.csv", index=False)
    print(f"wrote {OUT / 'decisions.csv'} ({len(rows)} pairs)")


def stage_identity(a):
    pairs = pd.read_csv(require(OUT / "pairs.csv", "proxy pairs (--stage pairs)", "EE_RESULTS_DIR"))
    call = claude_caller(a.model)

    def one(p):
        try:
            x, y = parse_json_response(call(p["source_text"])), parse_json_response(call(p["source_text"]))
        except Exception as e:  # noqa: BLE001
            print(f"  fail: {type(e).__name__}", flush=True)
            return None
        if not isinstance(x, dict) or not isinstance(y, dict):
            return None
        return {"hadm_id": p["hadm_id"], "model": p["model"], "task": p["task"], "efs_score": p["efs_score"],
                **{f"a_{k}": x.get(k) for k in FIELDS}, **{f"b_{k}": y.get(k) for k in FIELDS}}

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [r for r in ex.map(one, [p for _, p in pairs.iterrows()]) if r]
    pd.DataFrame(rows).to_csv(OUT / "identity_decisions.csv", index=False)
    print(f"wrote {OUT / 'identity_decisions.csv'} ({len(rows)} pairs)")


def stage_harm(a):
    import torch
    sc = pd.read_csv(require(a.hungarian_scores, "one-to-one aligner score table", "EE_RESULTS_DIR"))
    if "n_distortions" in sc:   # format written by scripts/score_generations.py
        sc = sc[(sc["seed"] == 123) & (sc["status"] == "ok")].rename(columns={"doc_id": "hadm_id", "efs": "efs_score"})
    if "mitigation" in sc:
        sc = sc[sc["mitigation"].fillna("none") == "none"]
    sc = sc[sc["source_hedges"] >= 1].copy()
    sc["bin"] = pd.qcut(sc["efs_score"], N_BINS, labels=False, duplicates="drop")
    samp = pd.concat([g.sample(min(N_PER_BIN, len(g)), random_state=42) for _, g in sc.groupby("bin")])
    src_df = pd.read_parquet(require(DATA_DIR / "full_sample.parquet", "depth-study sample", "EE_DATA_DIR"))
    src = {int(h): t[:12000] for h, t in zip(src_df["hadm_id"], src_df["discharge_text"])}
    rows = []
    for r in samp.itertuples():
        gen = generation(r.model, int(r.hadm_id), r.task)
        if gen is not None:
            rows.append({"hadm_id": r.hadm_id, "model": r.model, "task": r.task, "efs": r.efs_score,
                         "src": src[int(r.hadm_id)], "out": gen["output_text"]})
    print(f"{len(rows)} pairs stratified over {samp['bin'].nunique()} EFS bins", flush=True)

    tok, model = load_causal_lm(a.proxy_model)

    def build(text):
        return tok.apply_chat_template([{"role": "system", "content": SYS},
                                        {"role": "user", "content": PROMPT.replace("{NOTE}", text[:NOTE_CHARS])}],
                                       tokenize=False, add_generation_prompt=True)

    # source, output, and a second copy of the first 40 sources (determinism check)
    prompts = [build(r["src"]) for r in rows] + [build(r["out"]) for r in rows] \
        + [build(r["src"]) for r in rows[:N_DETERMINISM]]
    outs = []
    for i in range(0, len(prompts), BATCH):
        enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True, truncation=True,
                  max_length=4096).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True) for x in g]
    n = len(rows)
    S = [parse_json_response(x) for x in outs[:n]]
    O = [parse_json_response(x) for x in outs[n:2 * n]]
    S2 = [parse_json_response(x) for x in outs[2 * n:]]
    det = [i for i in range(len(S2)) if isinstance(S[i], dict) and isinstance(S2[i], dict)]
    floor = sum(any(S[i].get(f) != S2[i].get(f) for f in FIELDS) for i in det)
    recs = [{"hadm_id": r["hadm_id"], "model": r["model"], "task": r["task"], "efs": r["efs"],
             "any_change": any(S[i].get(f) != O[i].get(f) for f in FIELDS)}
            for i, r in enumerate(rows) if isinstance(S[i], dict) and isinstance(O[i], dict)]
    pd.DataFrame(recs).to_csv(OUT / "harm_pairs.csv", index=False)
    json.dump({"n": len(recs), "parse_fail": n - len(recs), "identity_floor_n": len(det),
               "identity_floor_pct_greedy": round(100 * floor / max(len(det), 1), 1)},
              open(OUT / "harm_summary.json", "w"), indent=2)
    print(f"wrote {OUT / 'harm_pairs.csv'} and harm_summary.json")


def changed(df, a, b):
    return pd.DataFrame({f: df[f"{a}_{f}"] != df[f"{b}_{f}"] for f in FIELDS}).assign(
        any=lambda x: x[FIELDS].any(axis=1))


def stage_analyze(a):
    keys = ["hadm_id", "model", "task"]
    v = {}
    if a.decisions.exists() and a.identity_decisions.exists():   # Claude proxy (API stages proxy, identity)
        ero = pd.read_csv(a.decisions).drop_duplicates(subset=keys)
        ctl = pd.read_csv(a.identity_decisions)
        ce, cc = changed(ero, "src", "out"), changed(ctl, "a", "b")
        v.update({"F5.proxy.n": len(ero), "F5.proxy.any_change_pct": 100 * float(ce["any"].mean()),
                  "F5.identity_floor.n": len(ctl), "F5.identity_floor.any_change_pct": 100 * float(cc["any"].mean())})
        for f in FIELDS:
            v[f"F5.proxy.{SHORT[f]}_changed_pct"] = 100 * float(ce[f].mean())
            v[f"F5.identity_floor.{SHORT[f]}_changed_pct"] = 100 * float(cc[f].mean())

        # Paired comparison on the same pairs (eroded reading vs identity control).
        pe = pd.concat([ero[keys], ce], axis=1).merge(pd.concat([ctl[keys], cc], axis=1), on=keys, suffixes=("_e", "_c"))
        v["F5.paired.n"] = len(pe)
        for f in FIELDS + ["any"]:
            tag = SHORT.get(f, "any")
            e, c = pe[f"{f}_e"].astype(bool), pe[f"{f}_c"].astype(bool)
            b, cn = int((e & ~c).sum()), int((~e & c).sum())
            lo, hi = wilson(int(c.sum()), len(pe))
            v.update({f"F5.paired.{tag}.b_eroded_only": b, f"F5.paired.{tag}.c_control_only": cn,
                      f"F5.paired.{tag}.mcnemar_exact_p": mcnemar_exact(b, cn),
                      f"F5.paired.{tag}.floor_wilson_lo_pct": 100 * lo, f"F5.paired.{tag}.floor_wilson_hi_pct": 100 * hi,
                      f"F5.paired.{tag}.excess_pts": 100 * float(e.mean() - c.mean())})
    else:
        print(f"  note: {a.decisions} or {a.identity_decisions} not found; Claude-proxy statistics skipped")

    if a.harm_pairs.exists():
        h = pd.read_csv(a.harm_pairs)
        r, p = stats.pointbiserialr(h["any_change"].astype(int), h["efs"])
        v.update({"F5.harm.n": len(h), "F5.harm.any_change_pct": 100 * float(h["any_change"].mean()),
                  "F5.harm.pointbiserial_r": float(r), "F5.harm.pointbiserial_p": float(p),
                  "F5.harm.efs_min": float(h["efs"].min()), "F5.harm.efs_max": float(h["efs"].max())})
        # five equal-width, right-closed EFS bins (pandas default)
        rate = h.groupby(pd.cut(h["efs"], 5), observed=False)["any_change"].mean()
        v.update({f"F5.harm.bin{i + 1}.change_pct": 100 * float(x) for i, x in enumerate(rate)})
    if a.harm_summary.exists():
        s = json.load(open(a.harm_summary))
        v.update({"F6.left_padded_identity_floor_pct": float(s["identity_floor_pct_greedy"]),
                  "F6.left_padded_identity_floor_n": int(s["identity_floor_n"])})
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump(v, open(RESULTS_DIR / "analysis" / "clinician_proxy.json", "w"), indent=1)
    for k, x in v.items():
        print(f"  {k:45s} {x}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["pairs", "proxy", "identity", "harm", "analyze"])
    ap.add_argument("--scores", type=Path, default=RESULTS_DIR / "scores" / "depth_greedy.csv",
                    help="greedy-aligner score table used to draw the eroded pairs")
    ap.add_argument("--hungarian-scores", type=Path, default=RESULTS_DIR / "scores" / "depth.csv",
                    help="one-to-one aligner score table used for the EFS-stratified harm sample")
    ap.add_argument("--model", default="claude-sonnet-4-6", help="Claude proxy model")
    ap.add_argument("--proxy-model", default="llama-3.3-70b", help="local proxy for --stage harm")
    ap.add_argument("--decisions", type=Path, default=OUT / "decisions.csv")
    ap.add_argument("--identity-decisions", type=Path, default=OUT / "identity_decisions.csv")
    ap.add_argument("--harm-pairs", type=Path, default=OUT / "harm_pairs.csv")
    ap.add_argument("--harm-summary", type=Path, default=OUT / "harm_summary.json")
    a = ap.parse_args()
    {"pairs": stage_pairs, "proxy": stage_proxy, "identity": stage_identity, "harm": stage_harm,
     "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    main()
