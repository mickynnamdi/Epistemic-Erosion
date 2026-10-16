"""Agreement of AI annotators with the human majority, the rule pipeline and each other.

Paper: Appendix J.9, Table A13 (Gwet's AC1 on "any distortion present"): vs human majority
(n=20) Claude Sonnet 4.6 0.947, Llama-3.3-70B 0.942, gpt-oss-20B 0.934 (n=18), Gemma-4-31B 0.559;
vs pipeline (n=164) 0.907, 0.921, 0.792 (n=139), 0.607; Claude vs Llama-3.3-70B 0.948.
Inputs: three rater JSON exports (--raters), the legal/scientific sheet (--sheet-legal-scientific),
the clinical sheet (--sheet-clinical, MIMIC-IV text), and the AI labels written by
scripts/validation/multi_annotator.py (--ai-dir). access: rater files and sheets are withheld
(clinical sheet: PhysioNet credentialed); CPU only.
Outputs: RESULTS_DIR/analysis/multi_annotator_agreement.json (check ids A13.*) and
RESULTS_DIR/multi_annotator/agreement_detail.json (raw %, Cohen's kappa, AC1, per type).
Usage:
    python scripts/validation/multi_annotator_agreement.py \\
        --raters r1.json r2.json r3.json [--rater-labels R1,R2,R3] \\
        --sheet-legal-scientific ls.csv --sheet-clinical clinical.csv [--pipeline hungarian|greedy]

Joins. The annotation app served a stratified random subsample of the legal/scientific sheet and
renumbered it 1..20 (scripts/validation/annotation_sampling.py), so human annotation id k is joined
to sheet row MAP[k]; the script asserts that each rater's (domain, model, task) matches that row.
AI labels and pipeline labels are keyed by (sheet name, sheet row id) on both sides.

Pipeline labels. `--pipeline hungarian` (the default) recomputes them with compute_efs, the
one-to-one aligner, as in Table A13's vs-pipeline column. `--pipeline greedy` uses the greedy
baseline aligner (compute_efs_greedy) and gives 0.900 / 0.914 / 0.782 / 0.613 for the four
annotators. The vs-human and AI-vs-AI columns do not use the pipeline.
"""
import argparse
import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from configs.paths import RESULTS_DIR  # noqa: E402
from validation.annotation_sampling import read_sheet, sample_pairs  # noqa: E402
from validation.multi_annotator import AI_DIR, DEFAULT_SHEETS, SHEET_CLINICAL, SHEET_LS, TYPES  # noqa: E402

BACKEND_SUFFIXES = ("_hf", "_anthropic")


# --- agreement statistics on binary labels ---
def raw(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((a == b).mean())


def cohen(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in (0, 1))
    return float((po - pe) / (1 - pe)) if (1 - pe) > 1e-9 else 1.0


def gwet_ac1(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    q = np.concatenate([a, b]).mean()
    pe = 2 * q * (1 - q)
    return float((po - pe) / (1 - pe)) if (1 - pe) > 1e-9 else 1.0


def agree(a, b):
    return {"n": int(len(a)), "raw": raw(a, b), "cohen_kappa": cohen(a, b), "gwet_ac1": gwet_ac1(a, b)}


def yn(v):
    return 1 if str(v).strip().lower() == "yes" else 0


def any_distortion(labels, keys):
    return np.array([max(labels[k][t] for t in TYPES) for k in keys])


# --- inputs ---
def load_ai(ai_dir):
    """{annotator: {"<sheet>_<id>": {type: 0/1}}}; files with a parse error are left out."""
    ai = {}
    for sub in sorted(Path(ai_dir).iterdir()) if Path(ai_dir).is_dir() else []:
        if not sub.is_dir():
            continue
        labels = {}
        for f in glob.glob(str(sub / "*.json")):
            r = json.load(open(f))
            if r.get("parse_error") or any(t not in r for t in TYPES):
                continue
            labels[f"{r['sheet']}_{r['id']}"] = {t: int(r[t]) for t in TYPES}
        if labels:
            name = sub.name
            for suf in BACKEND_SUFFIXES:
                name = name[:-len(suf)] if name.endswith(suf) else name
            ai[name] = labels
    return ai


def human_majority(rater_paths, sheet_rows):
    """Majority (>= 2 of 3) per type on the 20 served pairs, keyed by sheet row.

    Also returns each rater's any-distortion vector (in annotation-id order) for human-human agreement.
    """
    served = sample_pairs(sheet_rows)
    anns = [{int(a["id"]): a for a in json.load(open(p))["annotations"]} for p in rater_paths]
    for A in anns:  # every annotation must describe the sheet row it is joined to
        for p in served:
            got = (A[p["id"]]["domain"], A[p["id"]]["model"], A[p["id"]]["task"])
            assert got == (p["domain"], p["model"], p["task"]), f"annotation {p['id']} does not match sheet row"
    human = {f"{SHEET_LS}_{p['sheet_row_id']}": {
        t: 1 if sum(yn(A[p["id"]].get(f"has_{t}", "")) for A in anns) >= 2 else 0 for t in TYPES}
        for p in served}
    per_rater = [np.array([int(any(yn(A[p["id"]].get(f"has_{t}", "")) for t in TYPES)) for p in served])
                 for A in anns]
    return human, per_rater


def sheet_texts(sheets):
    """{"<sheet>_<id>": (source, output, domain)}; the clinical sheet has no domain column."""
    text = {}
    for name, path in sheets.items():
        if path is None or not Path(path).exists():
            print(f"  [skip] sheet {name}: {path} not found")
            continue
        df = pd.read_csv(path, dtype=str)
        for _, r in df.iterrows():
            dom = "clinical" if name == SHEET_CLINICAL else r.get("domain")
            text[f"{name}_{r.get('id')}"] = (str(r.get("source_text", "")), str(r.get("output_text", "")), dom)
    return text


def pipeline_labels(keys, text, pipeline):
    from src.efs import compute_efs, compute_efs_greedy
    fn = compute_efs_greedy if pipeline == "greedy" else compute_efs
    pipe = {}
    for k in keys:
        if k not in text:
            continue
        s, o, dom = text[k]
        if not s.strip() or not o.strip() or s == "nan" or o == "nan":
            continue
        bt = fn(s, o, domain=dom).get("by_type", {})
        pipe[k] = {t: 1 if bt.get(t, 0) > 0 else 0 for t in TYPES}
    return pipe


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raters", nargs=3, type=Path, metavar="RATER_JSON",
                    help="the three human raters' JSON exports (omit to skip the vs-human column)")
    ap.add_argument("--rater-labels", default="R1,R2,R3",
                    help="comma-separated labels for the raters, in --raters order (default: %(default)s)")
    ap.add_argument("--sheet-legal-scientific", type=Path, default=DEFAULT_SHEETS[SHEET_LS])
    ap.add_argument("--sheet-clinical", type=Path, default=DEFAULT_SHEETS[SHEET_CLINICAL])
    ap.add_argument("--ai-dir", type=Path, default=AI_DIR,
                    help="folder with one sub-folder of labels per AI annotator (default: %(default)s)")
    ap.add_argument("--pipeline", choices=["greedy", "hungarian"], default="hungarian",
                    help="aligner for the pipeline labels; Table A13 uses hungarian (default)")
    args = ap.parse_args()
    rater_labels = args.rater_labels.split(",")
    assert len(rater_labels) == 3, "--rater-labels needs three labels"
    if not args.sheet_legal_scientific.exists():
        raise SystemExit(f"[missing input] legal/scientific sheet not found at {args.sheet_legal_scientific}")

    ai = load_ai(args.ai_dir)
    if not ai:
        raise SystemExit(f"[missing input] no AI annotator labels under {args.ai_dir}; "
                         "run scripts/validation/multi_annotator.py first")
    print("AI annotators found:", {k: len(v) for k, v in ai.items()})

    v, detail = {}, {"pipeline_aligner": args.pipeline, "ai_annotators": {k: len(x) for k, x in ai.items()},
                     "vs_human_majority": {}, "vs_pipeline": {}, "ai_vs_ai": {}, "human_pairwise": {}}

    # (1) AI vs human majority on the 20 served pairs
    if args.raters:
        human, per_rater = human_majority(args.raters, read_sheet(args.sheet_legal_scientific))
        for name, labels in ai.items():
            ks = [k for k in human if k in labels]
            if not ks:
                continue
            a = agree(any_distortion(labels, ks), any_distortion(human, ks))
            detail["vs_human_majority"][name] = {
                "any_distortion": a,
                "per_type": {t: agree([labels[k][t] for k in ks], [human[k][t] for k in ks]) for t in TYPES}}
            v[f"A13.vs_human.{name}.ac1"] = a["gwet_ac1"]
            v[f"A13.vs_human.{name}.n"] = a["n"]
        for i, j in combinations(range(3), 2):
            a = agree(per_rater[i], per_rater[j])
            detail["human_pairwise"][f"{rater_labels[i]}-{rater_labels[j]}"] = a
            v[f"A13.human_pairwise.{rater_labels[i]}-{rater_labels[j]}.ac1"] = a["gwet_ac1"]

    # (2) AI vs pipeline, on every pair the annotator labelled
    text = sheet_texts({SHEET_LS: args.sheet_legal_scientific, SHEET_CLINICAL: args.sheet_clinical})
    pipe = pipeline_labels(set().union(*[set(x) for x in ai.values()]), text, args.pipeline)
    print(f"pipeline labels ({args.pipeline} aligner) computed for {len(pipe)} pairs")
    for name, labels in ai.items():
        ks = [k for k in pipe if k in labels]
        if ks:
            a = agree(any_distortion(labels, ks), any_distortion(pipe, ks))
            detail["vs_pipeline"][name] = {"any_distortion": a}
            tag = "vs_pipeline" if args.pipeline == "hungarian" else f"vs_pipeline_{args.pipeline}"
            v[f"A13.{tag}.{name}.ac1"] = a["gwet_ac1"]
            v[f"A13.{tag}.{name}.n"] = a["n"]

    # (3) AI vs AI on the pairs both labelled
    for a_name, b_name in combinations(list(ai), 2):
        ks = [k for k in ai[a_name] if k in ai[b_name]]
        if ks:
            a = agree(any_distortion(ai[a_name], ks), any_distortion(ai[b_name], ks))
            detail["ai_vs_ai"][f"{a_name}|{b_name}"] = a
            v[f"A13.ai_vs_ai.{a_name}|{b_name}.ac1"] = a["gwet_ac1"]
            v[f"A13.ai_vs_ai.{a_name}|{b_name}.n"] = a["n"]

    suffix = "" if args.pipeline == "hungarian" else f"_{args.pipeline}"
    (RESULTS_DIR / "multi_annotator").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump(detail, open(RESULTS_DIR / "multi_annotator" / f"agreement_detail{suffix}.json", "w"), indent=2)
    out = RESULTS_DIR / "analysis" / f"multi_annotator_agreement{suffix}.json"
    json.dump(v, open(out, "w"), indent=2)
    for k, x in v.items():
        print(f"{k:70s} {x:.4f}" if isinstance(x, float) else f"{k:70s} {x}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
