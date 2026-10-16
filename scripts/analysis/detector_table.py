"""Summarise the 12 detector runs into Table A16.

Paper: Section 6.9, Appendix N, Table A16 (accuracy mean +- std over seeds 42/123/2024,
macro-F1, per-type F1, per encoder).
Inputs: results/learned_detector/model_<encoder>_seed<S>/metrics.json (scripts/detector/train.py),
or --runs-dir pointing at another folder of such runs.
Outputs: results/tables/tableA16_detector.csv and results/analysis/detector_table.json.
Usage:
    python scripts/analysis/detector_table.py [--runs-dir DIR] [--report best|last]

As in the paper, accuracy is the best epoch's validation accuracy, while macro-F1 and
per-type F1 come from the classification report of the LAST epoch (--report last, the
default); --report best uses the checkpoint's own epoch instead.
Retraining gives a different train/validation split than the paper's (the paper's split
followed an unsorted file listing), so expect agreement to within about a point, not exactly.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import RESULTS_DIR  # noqa: E402

ENCODERS = {"RoBERTa-base": "roberta-base", "PubMedBERT": "BiomedNLP-BiomedBERT-base-uncased-abstract",
            "SciBERT": "scibert_scivocab_uncased", "RoBERTa-large": "roberta-large"}
TYPES = ["omission", "fabrication", "escalation", "attenuation", "scope_shift", "severity_warp"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=RESULTS_DIR / "learned_detector")
    ap.add_argument("--report", default="last", choices=["last", "best"])
    a = ap.parse_args()
    rows, v = [], {}
    for name, enc in ENCODERS.items():
        runs = [json.load(open(p)) for p in sorted(a.runs_dir.glob(f"model_{enc}_seed*/metrics.json"))]
        if not runs:
            print(f"no runs for {name} in {a.runs_dir}")
            continue
        key = "report_best_epoch" if a.report == "best" and "report_best_epoch" in runs[0] else "report"
        acc = [100 * r["best_val_acc"] for r in runs]
        mf1 = [r[key]["macro avg"]["f1-score"] for r in runs]
        row = {"encoder": name, "n_seeds": len(runs), "acc_mean": np.mean(acc), "acc_std": np.std(acc, ddof=1),
               "macro_f1_mean": np.mean(mf1), "macro_f1_std": np.std(mf1, ddof=1),
               **{f"f1_{t}": np.mean([r[key][t]["f1-score"] for r in runs]) for t in TYPES}}
        rows.append(row)
        v.update({f"A16.{name}.{k}": float(x) for k, x in row.items() if k not in ("encoder",)})
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tables" / "tableA16_detector.csv", index=False)
    json.dump(v, open(RESULTS_DIR / "analysis" / "detector_table.json", "w"), indent=1)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
