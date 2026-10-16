"""Length-matched token-distribution probe of assertive vs hedge continuations.

Paper: Section 6.8, Appendix M, Table A15 (e.g. Llama-3.1-8B: 24.818x without a source
hedge, 3.449x with one, 89.7% reduction).
Inputs: data/full_sample.parquet (MIMIC-IV, PhysioNet credentialed); one model from
configs/models.py (GPU; Gemma is run in bfloat16, the others in float16).
Outputs: results/probe/probe_<model>.json and results/analysis/probe_<model>.json.
Usage:
    python scripts/probe/certainty_prior_probe.py --model llama-3.1-8b   # also mistral-7b, med42-8b, gemma-3-4b

Design: 500 hedge-bearing and 500 non-hedge MIMIC sentences. Each hedge sentence is
paired with a non-hedge sentence whose length lies within max(8 chars, 15%) of it; the
partner is drawn at random (random.Random(42)) among the unused candidates in that window.
The next-token distribution after "The CT" is summed over single-token spellings of the
assertive and hedge word lists; the resulting token-id sets differ by tokenizer (e.g.
20/22 ids for Llama-3.1-8B, 8/11 for Mistral-7B), and are reported in the output.
Probabilities are computed with the model's float16/bfloat16 weights, so repeated runs on
different hardware or library versions agree to about 1%.
"""
import json, sys, re, statistics, math, random
from pathlib import Path
import torch
import pandas as pd
import scipy.stats as st
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import DATA_DIR, MODELS_ROOT, RESULTS_DIR, require  # noqa: E402
from configs.models import resolve  # noqa: E402

N = 500

HEDGES_RE = re.compile(
    r'\b(?:may|could|might|possibly|possible|likely|unlikely|suggestive of|'
    r'consistent with|concerning for|appears to|seems to|suspected|presumed|'
    r'cannot rule out|cannot exclude)\b', re.IGNORECASE,
)
ASSERTIVE_TARGETS = ['confirms','shows','reveals','demonstrates','is','has','displays','indicates']
HEDGE_TARGETS = ['may','could','might','possibly','appears','seems','suggests','likely']
PROMPT = "Summarize the following clinical sentence in a single sentence.\n\nSentence: {sentence}\n\nOne-sentence summary: The CT"

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="llama-3.1-8b",
                 help="short name from configs/models.py (paper: llama-3.1-8b, mistral-7b, med42-8b, gemma-3-4b)")
_args = _ap.parse_args()
SHORT = _args.model


def get_sentences(text, want_hedge):
    out = []
    for s in re.split(r'(?<=[.!?])\s+', text[:8000]):
        s = s.strip()
        if not (15 < len(s) < 250):
            continue
        if bool(HEDGES_RE.search(s)) != want_hedge:
            continue
        if not re.search(r'\b[A-Za-z]{4,}\b', s):
            continue
        out.append(s)
    return out


print("Collecting sentence pools...")
ep = pd.read_parquet(require(DATA_DIR / 'full_sample.parquet', 'MIMIC seed-123 sample', 'EE_DATA_DIR')).sample(frac=1.0, random_state=42)
hedge_pool, nonhedge_pool = [], []
for _, r in ep.iterrows():
    hedge_pool.extend(get_sentences(r['discharge_text'], True))
    nonhedge_pool.extend(get_sentences(r['discharge_text'], False))
    if len(hedge_pool) >= 3000 and len(nonhedge_pool) >= 5000:
        break
print(f"  hedge pool: {len(hedge_pool)}  non-hedge pool: {len(nonhedge_pool)}")

# Bin non-hedge by length
nh_by_len = {}
for s in nonhedge_pool:
    nh_by_len.setdefault(len(s), []).append(s)

rng = random.Random(42)
hedge_sample, nonhedge_sample = [], []
used = set()
for hs in hedge_pool:
    if len(hedge_sample) >= N:
        break
    hl = len(hs)
    margin = max(8, int(0.15 * hl))
    candidates = []
    for L in range(hl - margin, hl + margin + 1):
        for cs in nh_by_len.get(L, []):
            if id(cs) not in used:
                candidates.append(cs)
    if not candidates:
        continue
    chosen = rng.choice(candidates)
    used.add(id(chosen))
    hedge_sample.append(hs)
    nonhedge_sample.append(chosen)

print(f"\nMatched pairs collected: {len(hedge_sample)}")
hl = [len(s) for s in hedge_sample]
nl = [len(s) for s in nonhedge_sample]
print(f"  hedge length    : mean={statistics.mean(hl):.1f} median={statistics.median(hl):.0f} stdev={statistics.stdev(hl):.1f}")
print(f"  non-hedge length: mean={statistics.mean(nl):.1f} median={statistics.median(nl):.0f} stdev={statistics.stdev(nl):.1f}")
u, pval = st.mannwhitneyu(hl, nl, alternative='two-sided')
print(f"  Mann-Whitney length p = {pval:.3g}  ({'MATCHED' if pval>0.05 else 'still differs'})")

MODEL_ID, _rev = resolve(SHORT, MODELS_ROOT)
_revkw = {"revision": _rev} if _rev else {}
print(f"\nLoading {SHORT} from {MODEL_ID}...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, **_revkw)
_dtype = torch.bfloat16 if 'gemma' in SHORT.lower() else torch.float16   # Gemma underflows in float16
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=_dtype, device_map='auto', **_revkw)
model.eval()


def to_ids(words):
    ids = set()
    for w in words:
        for token in [w, ' '+w, w.capitalize(), ' '+w.capitalize()]:
            tids = tok(token, add_special_tokens=False)['input_ids']
            if len(tids) == 1:
                ids.add(tids[0])
    return ids


a_ids = list(to_ids(ASSERTIVE_TARGETS))
h_ids = list(to_ids(HEDGE_TARGETS))
print(f"Assertive ids: {len(a_ids)}  Hedge ids: {len(h_ids)}")


def run_condition(sentences, label):
    print(f"\nRunning {label} (n={len(sentences)})...")
    rows = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, s in enumerate(sentences):
            inp = tok(PROMPT.format(sentence=s), return_tensors='pt').to(device)
            logits = model(**inp).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1).cpu()
            a = sum(float(probs[t]) for t in a_ids)
            h = sum(float(probs[t]) for t in h_ids)
            rows.append({'assertive_mass': a, 'hedge_mass': h, 'preview': s[:120]})
            if (i+1) % 100 == 0:
                print(f"  [{i+1}/{len(sentences)}] mean_a={statistics.mean(r['assertive_mass'] for r in rows):.4f} "
                      f"mean_h={statistics.mean(r['hedge_mass'] for r in rows):.4f}", flush=True)
    return rows


h_rows = run_condition(hedge_sample, 'HEDGE-CONTEXT (length-matched)')
n_rows = run_condition(nonhedge_sample, 'NON-HEDGE BASELINE (length-matched)')


def summarise(rows, label):
    a = [r['assertive_mass'] for r in rows]
    h = [r['hedge_mass'] for r in rows]
    mean_a, mean_h = statistics.mean(a), statistics.mean(h)
    per_tok_a = mean_a / len(a_ids)
    per_tok_h = mean_h / len(h_ids)
    n_pref = sum(1 for r in rows if r['assertive_mass'] > r['hedge_mass'])
    p = n_pref / len(rows)
    z = 1.96; n = len(rows)
    denom = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    print(f"\n=== {label} (n={n}) ===")
    print(f"  Mean P(assertive)       = {mean_a:.4f}")
    print(f"  Mean P(hedge)           = {mean_h:.4f}")
    print(f"  Sum ratio               = {mean_a/max(mean_h,1e-9):.2f}x")
    print(f"  Per-token ratio         = {per_tok_a/max(per_tok_h,1e-9):.2f}x")
    print(f"  Prompts assertive>hedge = {n_pref}/{n} = {p:.1%}")
    print(f"  95% Wilson CI           = [{max(0,centre-half):.3f}, {min(1,centre+half):.3f}]")
    return {'n': n, 'mean_assertive': mean_a, 'mean_hedge': mean_h,
            'sum_ratio': mean_a/max(mean_h,1e-9),
            'per_token_ratio': per_tok_a/max(per_tok_h,1e-9),
            'fraction': p, 'wilson_lo': max(0,centre-half), 'wilson_hi': min(1,centre+half),
            'n_prefer_assertive': n_pref}


s_h = summarise(h_rows, 'HEDGE-CONTEXT')
s_n = summarise(n_rows, 'NON-HEDGE BASELINE')

print()
print("="*70)
print("CONTRAST (length-matched)")
print("="*70)
print(f"  Hedge-context sum ratio        = {s_h['sum_ratio']:.2f}x")
print(f"  Non-hedge baseline sum ratio   = {s_n['sum_ratio']:.2f}x")
print(f"  Hedge-context per-token ratio  = {s_h['per_token_ratio']:.2f}x")
print(f"  Non-hedge per-token ratio      = {s_n['per_token_ratio']:.2f}x")
print(f"  Fraction: hedge {s_h['fraction']:.1%} vs non-hedge {s_n['fraction']:.1%}  diff={s_h['fraction']-s_n['fraction']:+.1%}")

# Reduction calculation
red = (s_n['sum_ratio'] - s_h['sum_ratio']) / max(s_n['sum_ratio'] - 1, 1e-9) * 100
print(f"\n  Source-hedge reduces assertive-class advantage by ~{red:.1f}%")
print(f"    (formula: (baseline - hedge) / (baseline - 1.0))")

(RESULTS_DIR / 'probe').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'analysis').mkdir(parents=True, exist_ok=True)
json.dump({f"M.{SHORT}.baseline_ratio": s_n['sum_ratio'], f"M.{SHORT}.hedge_ratio": s_h['sum_ratio'],
           f"M.{SHORT}.reduction_pct": red, f"M.{SHORT}.hedge_frac_assertive_pct": 100 * s_h['fraction'],
           f"M.{SHORT}.baseline_frac_assertive_pct": 100 * s_n['fraction'],
           f"M.{SHORT}.n_assertive_ids": len(a_ids), f"M.{SHORT}.n_hedge_ids": len(h_ids),
           "M.length_match_p": float(pval)},
          open(RESULTS_DIR / 'analysis' / f'probe_{SHORT}.json', 'w'), indent=1)
with open(RESULTS_DIR / 'probe' / f'probe_{SHORT}.json', 'w') as f:
    json.dump({
        'n_per_condition': N,
        'length_matched': True,
        'matching_p_value': float(pval),
        'assertive_tokens': ASSERTIVE_TARGETS,
        'hedge_tokens': HEDGE_TARGETS,
        'n_assertive_ids': len(a_ids),
        'n_hedge_ids': len(h_ids),
        'hedge_context': s_h,
        'non_hedge_baseline': s_n,
        'reduction_percent': red,
        'per_prompt_hedge': h_rows,
        'per_prompt_nonhedge': n_rows,
    }, f, indent=2)
print(f"\nWrote {RESULTS_DIR / 'probe' / f'probe_{SHORT}.json'}")