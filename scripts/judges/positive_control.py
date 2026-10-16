"""Positive control for the Claude reference-set judge: four arms of 60 pairs it should or should not flag.

Paper: Appendix I. With the identical prompt the Claude judge flags 98.3% of real eroded
    generations (mean severity 2.833, 60 distinct written explanations), 60.0% of mechanically
    de-hedged outputs, 1.7% of the truncation stress-test pairs and 0% of byte-identical pairs,
    so its 0/242 on the identity stratum is not a constant response.
Inputs: RESULTS_DIR/judges/reference_set.json and truncation_stress_test.json
    (scripts/judges/build_judge_set.py --truncation-stress-test); access: PhysioNet credentialed
    and paid API (ANTHROPIC_API_KEY) for the judge stage.
Outputs: RESULTS_DIR/judges/positive_control_<model>.csv, RESULTS_DIR/analysis/positive_control.json.
Usage:
    python scripts/judges/positive_control.py --stage judge
    python scripts/judges/positive_control.py --stage analyze [--judgements CSV]
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from configs.paths import RESULTS_DIR, require  # noqa: E402
from judges.reference_set_judge import JUDGE_PROMPT, SYS, anthropic_client, parse  # noqa: E402

JUDGE_DIR = RESULTS_DIR / "judges"
ARMS = {"P_positive_eroded": "P", "D_dehedged": "D", "T_truncation_old": "T", "I_true_identity": "I"}

# Fixed hedge list used only to build the de-hedged arm (independent of the EFS lexicon).
HEDGES = [r"\bpossible\b", r"\bpossibly\b", r"\bprobable\b", r"\bprobably\b", r"\bmay\b",
          r"\bmight\b", r"\bcould\b", r"\bsuggests?\b", r"\bsuggestive of\b", r"\bappears?\b",
          r"\bseems?\b", r"\buncertain\b", r"\bunclear\b", r"\blikely\b", r"\bunlikely\b",
          r"\bpotential(?:ly)?\b", r"\bcannot be (?:ruled out|excluded)\b", r"\bconcerning for\b",
          r"\bconsistent with\b", r"\bpresumed\b", r"\bapparent\b", r"\bperhaps\b"]


def strip_hedges(t):
    for p in HEDGES:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", t)


def stage_judge(a):
    ref = json.load(open(require(JUDGE_DIR / "reference_set.json", "reference set", "EE_RESULTS_DIR")))
    trunc = json.load(open(require(JUDGE_DIR / "truncation_stress_test.json", "truncation stratum "
                                   "(build_judge_set.py --truncation-stress-test)", "EE_RESULTS_DIR")))
    pos = [r for r in ref if r["label"] == "positive"][:a.n]
    ident = [r for r in ref if r["model"] == "source-self"][:a.n]
    arms = [("P_positive_eroded", r["source_text"][:6000], r["output_text"]) for r in pos]
    arms += [("T_truncation_old", r["source_text"][:6000], r["output_text"]) for r in trunc[:a.n]]
    arms += [("D_dehedged", r["source_text"], strip_hedges(r["source_text"])) for r in ident]
    arms += [("I_true_identity", r["source_text"], r["output_text"]) for r in ident]
    client = anthropic_client()
    print(f"{len(arms)} calls across 4 arms ({a.n} each), judge={a.model}", flush=True)

    def one(item):
        arm, s, o = item
        try:
            m = client.messages.create(model=a.model, max_tokens=400, temperature=0.0, system=SYS,
                                       messages=[{"role": "user", "content":
                                                  JUDGE_PROMPT.format(source_excerpt=s, output_text=o)}])
            p = parse(m.content[0].text)
            return {"arm": arm, "score": p.get("uncertainty_distortion_score"), "shift": p.get("certainty_shift"),
                    "hedges_lost": p.get("hedges_lost_count"), "worst": p.get("worst_example")}
        except Exception as e:  # noqa: BLE001
            return {"arm": arm, "score": None, "err": type(e).__name__}

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(one, arms))
    out = JUDGE_DIR / f"positive_control_{a.model}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


def stage_analyze(a):
    df = pd.read_csv(require(a.judgements, "positive-control judgements", "EE_RESULTS_DIR"))
    v = {}
    for arm, tag in ARMS.items():
        s = df[(df["arm"] == arm) & df["score"].notna()]
        v.update({f"I.positive_control.{tag}.n": len(s),
                  f"I.positive_control.{tag}.flag_pct": 100 * float((s["score"] >= 1).mean()),
                  f"I.positive_control.{tag}.mean_severity": float(s["score"].mean()),
                  f"I.positive_control.{tag}.distinct_explanations": int(s["worst"].nunique())})
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump(v, open(RESULTS_DIR / "analysis" / "positive_control.json", "w"), indent=1)
    for k, x in v.items():
        print(f"  {k:45s} {x}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["judge", "analyze"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--n", type=int, default=60, help="pairs per arm")
    ap.add_argument("--judgements", type=Path, default=JUDGE_DIR / "positive_control_claude-sonnet-4-6.csv")
    a = ap.parse_args()
    stage_judge(a) if a.stage == "judge" else stage_analyze(a)


if __name__ == "__main__":
    main()
