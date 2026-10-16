"""Cross-domain transfer of the clinical-trained distortion detector (Table A14).

Paper: Appendix L, Table A14 (7-class and binary accuracy on legal n=482, scientific n=279
and a 2,000-example clinical validation subsample).
Inputs:
  * detector checkpoints results/learned_detector/model_<encoder>_seed<S>/ (scripts/detector/train.py;
    MIMIC-derived) and results/learned_detector/val.csv;
  * legal / scientific screened pools data/{legal,scientific}_sample_full.parquet
    (python scripts/data/sample_external_datasets.py --pools; public);
  * Llama-3.1-8B generations for the first pool documents
    (python scripts/generate/generate.py --study crossdomain --pool --domain legal --model llama-3.1-8b --limit 200).
GPU recommended.
Outputs: results/detector/transfer.csv and results/analysis/detector_transfer.json.
Usage:
    python scripts/detector/transfer.py --seed 42

How the test sets are built (kept exactly as in the paper's run): the first 200 generation
files, in sorted file-name order, of the first model folder in sorted order (Llama-3.1-8B),
are labelled by the simplified keyword labeller below (not by the EFS pipeline), and the
resulting sentence pairs form the test set. The paper evaluated an earlier single-seed set of
checkpoints; --seed selects which retrained checkpoints to evaluate.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import DATA_DIR, OUTPUTS_DIR, RESULTS_DIR, require  # noqa: E402

LABELS = ["no_distortion", "escalation", "attenuation", "omission", "fabrication", "scope_shift", "severity_warp"]
LABEL2ID = {x: i for i, x in enumerate(LABELS)}
DET_DIR = RESULTS_DIR / "learned_detector"
ENCODERS = {"PubMedBERT": "BiomedNLP-BiomedBERT-base-uncased-abstract", "RoBERTa-base": "roberta-base",
            "RoBERTa-large": "roberta-large", "SciBERT": "scibert_scivocab_uncased"}

# Simplified keyword labeller used only for this transfer test (copied verbatim from the paper's run).
HEDGES = {"possible": 1, "possibly": 1, "may": 1, "might": 1, "could": 1, "cannot rule out": 1, "cannot exclude": 1,
          "questionable": 1, "equivocal": 1, "indeterminate": 1, "uncertain": 1, "unclear": 1, "suspected": 2,
          "suggestive of": 2, "suspicious": 2, "concerning for": 2, "consistent with": 2, "compatible with": 2,
          "appears to": 2, "seems to": 2, "arguably": 2, "potentially": 2, "likely": 3, "probable": 3,
          "most likely": 3, "presumed": 3, "apparently": 3, "confirmed": 4, "definite": 4, "clearly": 4,
          "proven": 4, "undoubtedly": 4, "certainly": 4}
TERMS = {
    "legal": r"\b(?:court|ruling|statute|law|claim|liability|defendant|plaintiff|judgment|conviction|sentence|violation|"
             r"constitutional|evidence|precedent|motion|appeal|injunction|damages|negligence)\b",
    "scientific": r"\b(?:result|finding|hypothesis|effect|significant|correlation|association|p-value|data|evidence|"
                  r"conclusion|observation|analysis|experiment|study|research|mechanism|model|prediction)\b",
}


def strength(text):
    tl = text.lower()
    for t, s in sorted(HEDGES.items(), key=lambda x: -len(x[0])):
        if t in tl:
            return s
    return 0


def claims(text, domain):
    out = []
    for sent in re.split(r'(?<=[.!?])\s+|\n(?=[A-Z#\d])', text):
        sent = sent.strip()
        if len(sent) < 15:
            continue
        hs = [t for t in sorted(HEDGES, key=lambda x: -len(x)) if t in sent.lower()]
        ct = re.findall(TERMS[domain], sent, re.I)
        if ct or hs:
            out.append({"text": sent, "hedges": hs, "strength": max([strength(sent)], default=4) if hs else 4,
                        "terms": ct})
    return out


def label_pairs(src, out, domain):
    sc, oc = claims(src, domain), claims(out, domain)
    exs, matched = [], set()
    for s in sc:
        if not s["hedges"]:
            continue
        st = {t.lower() for t in s["terms"]}
        if not st:
            continue
        best, bi, bo = None, None, 0
        for i, o in enumerate(oc):
            ov = len(st & {t.lower() for t in o["terms"]})
            if ov > bo:
                bo, best, bi = ov, o, i
        if best is None or bo == 0:
            exs.append((s["text"][:512], "", "omission"))
            continue
        matched.add(bi)
        if s["strength"] < best["strength"] or (s["hedges"] and not best["hedges"]):
            exs.append((s["text"][:512], best["text"][:512], "escalation"))
        elif s["strength"] > best["strength"] and best["hedges"]:
            exs.append((s["text"][:512], best["text"][:512], "attenuation"))
        elif s["hedges"] and best["hedges"]:
            extra = {t.lower() for t in best["terms"]} - {t.lower() for t in s["terms"]}
            exs.append((s["text"][:512], best["text"][:512], "scope_shift" if extra else "no_distortion"))
    src_terms = {t.lower() for c in sc for t in c["terms"]}
    for i, o in enumerate(oc):
        ot = {t.lower() for t in o["terms"]}
        if i not in matched and o["hedges"] and ot and not (ot & src_terms):
            exs.append(("", o["text"][:512], "fabrication"))
    return exs


def test_set(domain, max_files=200):
    pool = {"legal": ("legal_sample_full.parquet", "case_id"), "scientific": ("scientific_sample_full.parquet", "paper_id")}
    f, col = pool[domain]
    ep = pd.read_parquet(require(DATA_DIR / f, f"{domain} screened pool", "EE_DATA_DIR"))
    src = {str(r[col]): str(r["source_text"])[:12000] for _, r in ep.iterrows()}
    gdir = require(OUTPUTS_DIR / "fullscale" / domain, f"{domain} pool generations", "EE_OUTPUTS_DIR")
    rows, n = [], 0
    for mdir in sorted(d for d in gdir.iterdir() if d.is_dir()):
        for jf in sorted(mdir.glob("*.json")):
            if n >= max_files:
                break
            d = json.load(open(jf))
            if d.get("error") not in (None, "None"):
                continue
            out, s = d.get("output_text", ""), src.get(str(d.get("doc_id", "")))
            if not out or len(out) < 50 or not s:
                continue
            rows += [{"source_text": a, "output_text": b, "label": lab} for a, b, lab in label_pairs(s, out, domain)]
            n += 1
        if n >= max_files:
            break
    return pd.DataFrame(rows)


def evaluate(model_dir, df):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(dev).eval()
    preds, gold = [], [LABEL2ID.get(x, 0) for x in df["label"]]
    # Tokenise one pair at a time, as in training: tok(a, "") encodes a single segment, whereas a
    # batched call would add separator tokens for the empty side of omission/fabrication pairs.
    pairs = [(str(s) if pd.notna(s) else "", str(o) if pd.notna(o) else "") for s, o in zip(df.source_text, df.output_text)]
    with torch.no_grad():
        for i in range(0, len(pairs), 32):
            encs = [tok(s, o, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
                    for s, o in pairs[i:i + 32]]
            batch = {k: torch.cat([e[k] for e in encs]).to(dev) for k in ("input_ids", "attention_mask")}
            preds += mdl(**batch).logits.argmax(-1).cpu().tolist()
    bg, bp = [int(x != 0) for x in gold], [int(x != 0) for x in preds]
    return {"accuracy": accuracy_score(gold, preds), "f1_macro": f1_score(gold, preds, average="macro", zero_division=0),
            "binary_accuracy": accuracy_score(bg, bp)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42, help="which retrained checkpoints to evaluate")
    a = ap.parse_args()
    sets = {d: test_set(d) for d in ("legal", "scientific")}
    val = pd.read_csv(require(DET_DIR / "val.csv", "detector validation split", "EE_RESULTS_DIR"), keep_default_na=False)
    sets["clinical_val"] = val.sample(n=min(len(val), 2000), random_state=42)
    rows, v = [], {}
    for name, enc in ENCODERS.items():
        mdir = DET_DIR / f"model_{enc}_seed{a.seed}"
        if not mdir.exists():
            print(f"skip {name}: no checkpoint at {mdir}")
            continue
        for dom, df in sets.items():
            r = evaluate(mdir, df)
            rows.append({"encoder": name, "domain": dom, "n": len(df), **r})
            v.update({f"A14.{name}.{dom}.n": len(df), f"A14.{name}.{dom}.acc": 100 * r["accuracy"],
                      f"A14.{name}.{dom}.binary_acc": 100 * r["binary_accuracy"]})
            print(f"{name:14s} {dom:13s} n={len(df):5d} acc={r['accuracy']:.3f} binary={r['binary_accuracy']:.3f}")
    (RESULTS_DIR / "detector").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "detector" / "transfer.csv", index=False)
    json.dump(v, open(RESULTS_DIR / "analysis" / "detector_transfer.json", "w"), indent=1)


if __name__ == "__main__":
    main()
