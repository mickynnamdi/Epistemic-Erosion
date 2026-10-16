"""Claude severity judge on the controlled ground-truth pairs (known fraction of hedges removed).

Paper: Appendix J.7. An independent Claude Sonnet 4.6 judge's 0-5 severity rating tracks the
    ground-truth fraction of removed hedges, Spearman rho = 0.650 (40 clinical documents x 5
    fractions = 200 pairs, one removal seed; 147 pairs returned a parseable rating).
    The EFS side of J.7 (Table A12, rho = -0.891) is scripts/validation/controlled_groundtruth_validation.py.
Inputs: EE_DATA_DIR/full_sample.parquet; access: PhysioNet credentialed, paid API
    (ANTHROPIC_API_KEY) for the judge stage.
Outputs: RESULTS_DIR/controlled_groundtruth/claude_severity.csv (no text),
    RESULTS_DIR/analysis/controlled_gt_claude.json.
Usage:
    python scripts/judges/controlled_gt_claude.py --stage judge
    python scripts/judges/controlled_gt_claude.py --stage analyze [--judgements CSV]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from configs.paths import DATA_DIR, RESULTS_DIR, require  # noqa: E402
from judges.reference_set_judge import anthropic_client  # noqa: E402

LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEV_PROMPT = """You are rating how severely a summary distorts the CERTAINTY of its source (ignore factual content, fluency, length).

SOURCE:
{source}

OUTPUT:
{output}

Rate epistemic-distortion severity 0-5 (0=certainty perfectly preserved, 5=source uncertainty largely removed/hardened). Respond ONLY with JSON: {{"severity": <0-5 integer>}}"""


def remove_fraction(text, frac, seed):
    from src.hedge_screener import find_hedges
    ms = find_hedges(text)
    if not ms:
        return text
    rng = random.Random(seed)
    k = int(round(frac * len(ms)))
    drop = set(rng.sample(range(len(ms)), k))
    for i in sorted(range(len(ms)), key=lambda i: -ms[i].start):
        if i in drop:
            text = text[:ms[i].start] + text[ms[i].end:]
    return text


def stage_judge(a):
    from src.efs import compute_efs_greedy
    from src.hedge_screener import hedge_density
    df = pd.read_parquet(require(DATA_DIR / "full_sample.parquet", "depth-study sample", "EE_DATA_DIR"))
    docs = [str(t) for t in df["discharge_text"] if hedge_density(str(t)[:6000])["total_hedges"] >= 6][:a.n_docs]
    client = anthropic_client()
    rows = []
    for di, src in enumerate(docs):
        src = src[:6000]
        for f in LEVELS:
            out = remove_fraction(src, f, seed=di)
            try:
                # NOTE: temperature is not set (API default), as in the paper's run.
                m = client.messages.create(model=a.model, max_tokens=60, system="Output ONLY the JSON object.",
                                           messages=[{"role": "user", "content":
                                                      SEV_PROMPT.format(source=src[:3000], output=out[:2500])}])
                mm = re.search(r'"severity"\s*:\s*(\d+)', m.content[0].text)
                sev = int(mm.group(1)) if mm else np.nan
            except Exception:  # noqa: BLE001
                sev = np.nan
            # EFS column: greedy baseline aligner, as stored with the paper's run
            rows.append({"doc": di, "frac_removed": f,
                         "efs": compute_efs_greedy(src, out, domain="clinical")["efs_score"], "judge_severity": sev})
    out_dir = RESULTS_DIR / "controlled_groundtruth"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "claude_severity.csv", index=False)
    print(f"wrote {out_dir / 'claude_severity.csv'}")


def stage_analyze(a):
    d = pd.read_csv(require(a.judgements, "Claude severity judgements", "EE_RESULTS_DIR"))
    dd = d.dropna(subset=["judge_severity"])
    rho_f = stats.spearmanr(dd["frac_removed"], dd["judge_severity"])
    rho_e = stats.spearmanr(dd["efs"], dd["judge_severity"])
    v = {"J7.claude.n_pairs": len(d), "J7.claude.n_rated": len(dd),
         "J7.claude.severity_vs_fraction_rho": float(rho_f.correlation),
         "J7.claude.severity_vs_fraction_p": float(rho_f.pvalue),
         "J7.claude.severity_vs_greedy_efs_rho": float(rho_e.correlation)}
    v.update({f"J7.claude.mean_severity.{k:.2f}": float(x)
              for k, x in dd.groupby("frac_removed")["judge_severity"].mean().items()})
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump(v, open(RESULTS_DIR / "analysis" / "controlled_gt_claude.json", "w"), indent=1)
    for k, x in v.items():
        print(f"  {k:40s} {x}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["judge", "analyze"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--n-docs", type=int, default=40)
    ap.add_argument("--judgements", type=Path, default=RESULTS_DIR / "controlled_groundtruth" / "claude_severity.csv")
    a = ap.parse_args()
    stage_judge(a) if a.stage == "judge" else stage_analyze(a)


if __name__ == "__main__":
    main()
