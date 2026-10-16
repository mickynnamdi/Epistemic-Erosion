"""De-hedging with the remaining text held fixed: does deleting only the hedge spans change a clinical decision?

Paper: Appendix F.6. On 185 hedge-rich notes a Llama-3.3-70B clinician proxy (greedy,
    fixed-length padding) changes at least one decision in 15.7% of de-hedged notes against a
    1.1% identity floor (same notes scored twice under reversed batch order): excess 14.6
    points, bootstrap 95% CI [9.2, 20.0] (5,000 resamples), exact McNemar b = 28, c = 1,
    p = 1.1e-7.
Inputs: EE_DATA_DIR/full_sample.parquet; access: PhysioNet credentialed, GPU (2 x 80 GB,
    Llama-3.3-70B bf16) for the run stage.
Outputs: RESULTS_DIR/clinician_proxy/dehedge_pairs.csv, RESULTS_DIR/analysis/dehedge_decisions.json.
Usage:
    python scripts/decision/dehedge_decisions.py --stage run
    python scripts/decision/dehedge_decisions.py --stage analyze [--pairs CSV] [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from configs.paths import DATA_DIR, RESULTS_DIR, require  # noqa: E402
from decision.clinician_proxy import FIELDS, PROMPT, SYS, mcnemar_exact, parse_json_response  # noqa: E402
from judges.reference_set_judge import load_causal_lm  # noqa: E402

OUT = RESULTS_DIR / "clinician_proxy"
BATCH, MAX_NEW, PAD_LEN, NOTE_CHARS, MIN_HEDGES = 8, 200, 3072, 8000, 3
N_BOOT = 5000


def strip_all_hedges(text):
    """Delete every detected hedge span; everything else is left untouched."""
    from src.hedge_screener import find_hedges
    for m in sorted(find_hedges(text), key=lambda m: -m.start):
        text = text[:m.start] + text[m.end:]
    return text


def stage_run(a):
    import torch
    from src.hedge_screener import find_hedges
    ep = pd.read_parquet(require(DATA_DIR / "full_sample.parquet", "depth-study sample", "EE_DATA_DIR"))
    notes = [str(t)[:NOTE_CHARS] for t in ep["discharge_text"]]
    notes = [n for n in notes if len(find_hedges(n)) >= MIN_HEDGES]
    print(f"{len(notes)} hedge-rich notes (>= {MIN_HEDGES} hedges)", flush=True)
    tok, model = load_causal_lm(a.proxy_model)

    def build(t):
        return tok.apply_chat_template([{"role": "system", "content": SYS},
                                        {"role": "user", "content": PROMPT.replace("{NOTE}", t)}],
                                       tokenize=False, add_generation_prompt=True)

    def run(prompts):
        # Fixed-length padding: with plain left padding the logits depend on batch position.
        outs = []
        for i in range(0, len(prompts), BATCH):
            enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding="max_length", max_length=PAD_LEN,
                      truncation=True).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True) for x in g]
        return [parse_json_response(o) for o in outs]

    p_src = [build(n) for n in notes]
    a1 = run(p_src)
    a2 = list(reversed(run(list(reversed(p_src)))))     # same notes, reversed batch order
    b = run([build(strip_all_hedges(n)) for n in notes])

    def changed(x, y):
        return isinstance(x, dict) and isinstance(y, dict) and any(x.get(f) != y.get(f) for f in FIELDS)

    ok = [i for i in range(len(notes)) if all(isinstance(z[i], dict) for z in (a1, a2, b))]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"identity_changed": [changed(a1[i], a2[i]) for i in ok],
                  "dehedge_changed": [changed(a1[i], b[i]) for i in ok]}).to_csv(OUT / "dehedge_pairs.csv", index=False)
    print(f"wrote {OUT / 'dehedge_pairs.csv'} ({len(ok)} notes; {len(notes) - len(ok)} parse failures)")


def stage_analyze(a):
    d = pd.read_csv(require(a.pairs, "de-hedging decisions", "EE_RESULTS_DIR"))
    ident, deh = d["identity_changed"].astype(bool).values, d["dehedge_changed"].astype(bool).values
    n = len(d)
    b, c = int((deh & ~ident).sum()), int((ident & ~deh).sum())
    diff = deh.astype(int) - ident.astype(int)
    rng = np.random.default_rng(a.seed)
    boot = 100 * diff[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    v = {"F6.n": n, "F6.identity_floor_pct": 100 * ident.mean(), "F6.dehedge_change_pct": 100 * deh.mean(),
         "F6.excess_pts": 100 * diff.mean(), "F6.mcnemar_b_dehedge_only": b, "F6.mcnemar_c_identity_only": c,
         "F6.mcnemar_exact_p": mcnemar_exact(b, c), "F6.bootstrap_ci_lo": float(lo), "F6.bootstrap_ci_hi": float(hi),
         "F6.bootstrap_seed": a.seed}
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump(v, open(RESULTS_DIR / "analysis" / "dehedge_decisions.json", "w"), indent=1)
    for k, x in v.items():
        print(f"  {k:30s} {x}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["run", "analyze"])
    ap.add_argument("--proxy-model", default="llama-3.3-70b")
    ap.add_argument("--pairs", type=Path, default=OUT / "dehedge_pairs.csv")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed (the paper's seed is not known)")
    a = ap.parse_args()
    stage_run(a) if a.stage == "run" else stage_analyze(a)


if __name__ == "__main__":
    main()
