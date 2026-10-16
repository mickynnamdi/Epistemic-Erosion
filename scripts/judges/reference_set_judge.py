"""LLM judge on the Appendix I reference set (Llama-3.3-70B or Claude) and the Table A8 statistics.

Paper: Appendix I / Table A8 and Section 6.6. Identity stratum (byte-identical pairs): the
    Llama-3.3-70B judge flags 1/242 = 0.4% (Wilson 95% CI [0.1, 2.3]; zero parse failures, all
    242 rated "same"; the flagged pair names no distortion), Claude Sonnet 4.6 flags 0/242
    (CI [0.0, 1.6]); the greedy pipeline flags 67/242 = 27.7% at 3,000 characters. Positives:
    the 70B judge flags all 242; pipeline distortion count vs judge 0-5 severity r = 0.186
    (p = 0.004). Truncation stress test (not a specificity test): 70B judge flags 139/242 (57.4%);
    notes average 10,210 characters, the 3,000-character "output" keeps 24.7% of the note's
    hedges, 236/242 pairs (97.5%) lose a hedge relative to the 6,000-character excerpt the judge
    sees, and the greedy pipeline flags 89.7% on that excerpt.
Inputs: RESULTS_DIR/judges/reference_set.json and truncation_stress_test.json
    (scripts/judges/build_judge_set.py); access: PhysioNet credentialed, plus GPU (2 x 80 GB,
    Llama-3.3-70B bf16) for --backend hf or paid API (ANTHROPIC_API_KEY) for --backend anthropic.
Outputs: RESULTS_DIR/judges/reference_judge_<model>_<stratum>.csv,
    RESULTS_DIR/analysis/reference_set_judge.json.
Usage:
    python scripts/judges/reference_set_judge.py --stage judge --stratum identity --backend hf
    python scripts/judges/reference_set_judge.py --stage judge --stratum positives --backend hf
    python scripts/judges/reference_set_judge.py --stage judge --stratum truncation --backend hf
        (needs build_judge_set.py --truncation-stress-test)
    python scripts/judges/reference_set_judge.py --stage judge --stratum identity --backend anthropic
    python scripts/judges/reference_set_judge.py --stage analyze \
        [--positives CSV] [--truncation CSV] [--identity CSV] [--identity-claude CSV] [--truncation-set JSON]
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from configs.paths import MODELS_ROOT, RESULTS_DIR, require  # noqa: E402

JUDGE_DIR = RESULTS_DIR / "judges"
DEFAULT_HF_JUDGE = "llama-3.3-70b"
DEFAULT_CLAUDE_JUDGE = "claude-sonnet-4-6"

# Verbatim prompt that produced every stored judgement (Appendix I shows it condensed).
JUDGE_PROMPT = """You are an expert evaluator assessing whether an LLM output faithfully preserves the uncertainty expressed in a source clinical note.

**Source text (excerpt):**
{source_excerpt}

**LLM output:**
{output_text}

Rate the following on the scales provided. Respond ONLY with a JSON object, no other text.

{{
  "uncertainty_distortion_score": <0-5 integer, where 0 = perfect preservation, 5 = severe distortion>,
  "hedges_lost_count": <estimated number of uncertainty markers (e.g., "possible", "likely", "cannot rule out") that were dropped or changed>,
  "certainty_shift": "<one of: 'more_certain', 'same', 'less_certain', 'mixed'>",
  "worst_example": "<one sentence describing the worst distortion you found, or 'none' if faithful>"
}}"""
SYS = "You are a precise JSON-outputting evaluator. Output only valid JSON."
SOURCE_CHARS = 6000        # the judge sees the first 6,000 characters of the source
BATCH, MAX_NEW_BATCHED, MAX_NEW_SINGLE = 8, 128, 400


def parse(raw):
    """First {...} block of the reply as JSON, else {}."""
    try:
        m = re.search(r"\{.*\}", raw or "", re.S)
        if m:
            return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        pass
    return {}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def anthropic_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run the Claude judge (paid API).")
    from anthropic import Anthropic
    return Anthropic()


def load_causal_lm(name):
    """Tokenizer (left padding) and bf16 model, from EE_MODELS_ROOT or the pinned Hub revision."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from configs.models import resolve
    path, rev = resolve(name, MODELS_ROOT)
    kw = {"revision": rev} if rev else {}
    tok = AutoTokenizer.from_pretrained(path, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # required for correct batched generation
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto", **kw).eval()
    return tok, model


def judge_hf_batched(pairs, model_name):
    """Greedy, 128 new tokens, left-padded batches of 8 (Table A3; used for the identity stratum)."""
    import torch
    tok, model = load_causal_lm(model_name)
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYS},
         {"role": "user", "content": JUDGE_PROMPT.format(source_excerpt=s, output_text=o)}],
        tokenize=False, add_generation_prompt=True) for s, o in pairs]
    raws = []
    for i in range(0, len(prompts), BATCH):
        enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_BATCHED, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        raws += [tok.decode(g, skip_special_tokens=True) for g in out[:, enc["input_ids"].shape[1]:]]
        print(f"  [{min(i + BATCH, len(prompts))}/{len(prompts)}]", flush=True)
    return raws


def judge_hf_single(pairs, model_name):
    """One pair at a time through the package generator, greedy, 400 new tokens (positives, truncation)."""
    from src.generator import GenerationRequest, generate
    raws = []
    for i, (s, o) in enumerate(pairs):
        req = GenerationRequest(hadm_id=0, task="judge", model=model_name, source_text=s, system_prompt=SYS,
                                user_prompt=JUDGE_PROMPT.format(source_excerpt=s, output_text=o),
                                max_tokens=MAX_NEW_SINGLE, temperature=0.0)
        raws.append(generate(req, backend="hf").output_text or "")
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(pairs)}]", flush=True)
    return raws


def judge_anthropic(pairs, model_name):
    """One call per pair, temperature 0, 400 output tokens; a failed call yields None (row dropped)."""
    client = anthropic_client()
    raws = []
    for i, (s, o) in enumerate(pairs):
        try:
            m = client.messages.create(model=model_name, max_tokens=MAX_NEW_SINGLE, temperature=0.0, system=SYS,
                                       messages=[{"role": "user", "content":
                                                  JUDGE_PROMPT.format(source_excerpt=s, output_text=o)}])
            raws.append(m.content[0].text)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] {type(e).__name__}", flush=True)
            raws.append(None)
    return raws


def stage_judge(a):
    if a.stratum == "truncation":
        recs = json.load(open(require(JUDGE_DIR / "truncation_stress_test.json", "truncation stratum "
                                      "(build_judge_set.py --truncation-stress-test)", "EE_RESULTS_DIR")))
    else:
        recs = json.load(open(require(JUDGE_DIR / "reference_set.json", "reference set", "EE_RESULTS_DIR")))
        recs = [r for r in recs if (r["label"] == "positive") == (a.stratum == "positives")]
    if a.stratum == "identity":
        assert all(r["source_text"] == r["output_text"] for r in recs), "identity pair is not byte-identical"
    model = a.model or (DEFAULT_CLAUDE_JUDGE if a.backend == "anthropic" else DEFAULT_HF_JUDGE)
    pairs = [(r["source_text"][:SOURCE_CHARS], r["output_text"]) for r in recs]
    print(f"{len(pairs)} {a.stratum} pairs | backend={a.backend} judge={model}", flush=True)
    t0 = time.time()
    if a.backend == "anthropic":
        raws = judge_anthropic(pairs, model)
    elif a.stratum == "identity":
        raws = judge_hf_batched(pairs, model)
    else:
        raws = judge_hf_single(pairs, model)
    rows = []
    for r, raw in zip(recs, raws):
        if raw is None:
            continue
        p = parse(raw)
        rows.append({**{k: v for k, v in r.items() if k not in ("source_text", "output_text")},
                     "judge_distortion_score": p.get("uncertainty_distortion_score"),
                     "judge_hedges_lost": p.get("hedges_lost_count"),
                     "judge_certainty_shift": p.get("certainty_shift"),
                     "judge_worst_example": p.get("worst_example"), "judge_raw": raw})
    out = JUDGE_DIR / f"reference_judge_{model.replace('/', '_')}_{a.stratum}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows, {time.time() - t0:.0f}s)")


def judge_block(df, tag):
    """Flag rate (score >= 1) with Wilson CI on one judge's identity-stratum judgements."""
    ok = df[df["judge_distortion_score"].notna()]
    n = len(ok)
    flagged = ok[ok["judge_distortion_score"] >= 1]
    lo, hi = wilson(len(flagged), n)
    none_example = flagged["judge_worst_example"].fillna("").astype(str).str.strip().str.lower().eq("none")
    return {f"I.identity.{tag}.n_scored": n, f"I.identity.{tag}.parse_fail": int(len(df) - n),
            f"I.identity.{tag}.flagged": int(len(flagged)),
            f"I.identity.{tag}.flag_pct": 100 * len(flagged) / n,
            f"I.identity.{tag}.wilson_lo_pct": 100 * lo, f"I.identity.{tag}.wilson_hi_pct": 100 * hi,
            f"I.identity.{tag}.rated_same": int((ok["judge_certainty_shift"] == "same").sum()),
            f"I.identity.{tag}.flagged_with_no_distortion_named": int(none_example.sum())}


def truncation_facts(path):
    """What the truncation pairs contain, measured on their text."""
    from src.efs import compute_efs_greedy
    from src.hedge_screener import hedge_density
    recs = [r for r in json.load(open(path)) if r.get("model") == "source-self" and r["source_text"] != r["output_text"]]
    chars, ratio, lose, flag = [], [], 0, 0
    for r in recs:
        s, o = r["source_text"], r["output_text"]
        h_out = hedge_density(o)["total_hedges"]
        h_full = hedge_density(s)["total_hedges"]
        h_shown = hedge_density(s[:SOURCE_CHARS])["total_hedges"]
        chars.append(len(s))
        ratio.append(h_out / max(h_full, 1))
        lose += h_out < h_shown
        flag += len(compute_efs_greedy(s[:SOURCE_CHARS], o, domain="clinical")["distortions"]) > 0
    n = len(recs)
    return {"I.truncation.n": n, "I.truncation.mean_source_chars": float(np.mean(chars)),
            "I.truncation.hedges_retained_pct": 100 * float(np.mean(ratio)),
            "I.truncation.pairs_losing_hedge": int(lose), "A8.truncation.pairs_losing_hedge_pct": 100 * lose / n,
            "A8.truncation.pipeline_greedy_flag_pct": 100 * flag / n}


def stage_analyze(a):
    v = {}
    if a.positives.exists():
        pos = pd.read_csv(a.positives)
        pos = pos[pos["label"] == "positive"]
        ok = pos[pos["judge_distortion_score"].notna()]
        r, p = stats.pearsonr(ok["distortion_count"], ok["judge_distortion_score"])
        v.update({"I.positives.n": len(pos), "I.positives.judge70b_flagged": int((ok["judge_distortion_score"] >= 1).sum()),
                  "I.positives.pearson_r": float(r), "I.positives.pearson_p": float(p)})
    if a.truncation.exists():
        tr = pd.read_csv(a.truncation)
        tr = tr[tr["model"] == "source-self"]
        k = int((tr["judge_distortion_score"] >= 1).sum())
        v.update({"A8.truncation.judge70b_flagged": k, "I.truncation.judge70b_flag_pct": 100 * k / len(tr)})
    for path, tag in ((a.identity, "judge70b"), (a.identity_claude, "claude")):
        if path.exists():
            df = pd.read_csv(path)
            v.update(judge_block(df, tag))
            if tag == "judge70b":
                k = int((df["pipeline_distortion_count"] > 0).sum())
                v.update({"A8.identity.pipeline_greedy_flagged": k,
                          "A8.identity.pipeline_greedy_fpr_pct": 100 * k / len(df)})
    if a.truncation_set.exists():
        v.update(truncation_facts(a.truncation_set))
    if not v:
        raise SystemExit("no judgement files found; run --stage judge or pass judgement files")
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump(v, open(RESULTS_DIR / "analysis" / "reference_set_judge.json", "w"), indent=1)
    for k, x in v.items():
        print(f"  {k:55s} {x}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["judge", "analyze"])
    ap.add_argument("--stratum", choices=["positives", "identity", "truncation"], default="identity")
    ap.add_argument("--backend", choices=["hf", "anthropic"], default="hf")
    ap.add_argument("--model", help=f"judge model (default {DEFAULT_HF_JUDGE} for hf, {DEFAULT_CLAUDE_JUDGE} for anthropic)")
    ap.add_argument("--positives", type=Path, default=JUDGE_DIR / f"reference_judge_{DEFAULT_HF_JUDGE}_positives.csv",
                    help="70B judgements of the positives (rows with label == positive are used)")
    ap.add_argument("--truncation", type=Path, default=JUDGE_DIR / f"reference_judge_{DEFAULT_HF_JUDGE}_truncation.csv",
                    help="70B judgements of the truncation pairs (rows with model == source-self are used)")
    ap.add_argument("--identity", type=Path, default=JUDGE_DIR / f"reference_judge_{DEFAULT_HF_JUDGE}_identity.csv")
    ap.add_argument("--identity-claude", type=Path,
                    default=JUDGE_DIR / f"reference_judge_{DEFAULT_CLAUDE_JUDGE}_identity.csv")
    ap.add_argument("--truncation-set", type=Path, default=JUDGE_DIR / "truncation_stress_test.json",
                    help="truncation pairs with text (for the stratum's hedge and pipeline facts)")
    a = ap.parse_args()
    stage_judge(a) if a.stage == "judge" else stage_analyze(a)


if __name__ == "__main__":
    main()
