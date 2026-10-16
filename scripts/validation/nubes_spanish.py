"""Validate the Spanish hedge lexicon against human uncertainty cues in NUBes / IULA+.

Paper: Appendix J.1 (NUBes paragraph): 615 annotation files, 2,795 uncertainty cues, 321 distinct
surface forms; over five random 50/50 document splits the hand-built Spanish lexicon recovers
0.448 +- 0.019 of held-out human cues (worst split 0.420), sentence-level precision 0.824, recall
0.487, F1 0.612; a 100-form lexicon derived on the training half reaches cue recall
0.909 +- 0.008 but precision 0.250 and F1 0.394.
Inputs: NUBes / IULA+ BRAT files (*.txt + *.ann) under EE_NUBES_DIR; access: public (CC BY-SA 3.0 ES).
Outputs: RESULTS_DIR/analysis/nubes_spanish.json (check ids J1.nubes.*).
Usage:
    python scripts/validation/nubes_spanish.py

Unit of analysis. A "document" is every .ann file with a matching .txt (615); only 355 of them
contain an uncertainty cue. A sentence is a non-empty line; it is gold-uncertain when a human cue overlaps it, and
predicted uncertain when any lexicon pattern matches it. Cue recall asks whether any pattern
matches inside the human cue's own text.
"""
import glob
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import NUBES_DIR, RESULTS_DIR, require  # noqa: E402

N_SPLITS = 5   # seeds 0..4, random.Random(seed).shuffle of document indices

# The hand-built Spanish lexicon of the multilingual probe (Appendix F.7), verbatim: 19 patterns.
# NOTE: r"\bprobablemente\b" is listed twice, so only 18 patterns are distinct. The duplicate
# changes no number here (every score only asks whether ANY pattern matches), and it is kept so
# that the lexicon is the one the paper calls "19-pattern".
ES_LEXICON = [
    r"\bposible(?:s)?\b", r"\bposiblemente\b", r"\bprobable(?:s)?\b", r"\bprobablemente\b",
    r"\bpuede(?:n)?\b", r"\bpodr[ií]a(?:n)?\b", r"\bsugiere(?:n)?\b", r"\bparece(?:n)?\b",
    r"\bincierto(?:s)?\b", r"\binciert[ao](?:s)?\b", r"\bpoco clara?\b", r"\bprobablemente\b",
    r"\bevidencia de baja calidad\b", r"\bno se puede(?:n)? (?:descartar|excluir)\b",
    r"\bquiz[áa]s?\b", r"\btal vez\b", r"\bpotencial(?:mente)?\b", r"\bpresunto\b", r"\bsospecha\b",
]


def load(root):
    """[(text, [(start, end, cue text)])] for every .ann with a .txt, in sorted path order."""
    docs = []
    for ann in sorted(glob.glob(str(root / "**" / "*.ann"), recursive=True)):
        txt = ann[:-4] + ".txt"
        if not os.path.exists(txt):
            continue
        text = open(txt, encoding="utf-8", errors="replace").read()
        cues = []
        for ln in open(ann, encoding="utf-8", errors="replace"):
            if not ln.startswith("T"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            h = p[1].split()
            if not h[0].startswith("Uncert"):
                continue
            try:  # BRAT "LABEL start end", possibly discontinuous: first start, last end
                cues.append((int(h[1]), int(h[-1]), p[2]))
            except ValueError:
                continue
        docs.append((text, cues))
    return docs


def compile_forms(forms):
    return [re.compile(r"(?<!\w)" + re.escape(f) + r"(?!\w)", re.IGNORECASE) for f in forms]


def score(docs, regexes):
    n_cues = hit = 0
    tp = fp = fn = tn = 0
    for text, cues in docs:
        for _, _, phrase in cues:
            n_cues += 1
            if any(r.search(phrase) for r in regexes):
                hit += 1
        pos = 0
        for line in text.split("\n"):
            s, e = pos, pos + len(line)
            pos = e + 1
            if not line.strip():
                continue
            gold = any(not (ce <= s or cs >= e) for cs, ce, _ in cues)
            pred = any(r.search(line) for r in regexes)
            if pred and gold:
                tp += 1
            elif pred and not gold:
                fp += 1
            elif gold:
                fn += 1
            else:
                tn += 1
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"cue_recall": hit / n_cues if n_cues else 0.0, "precision": P, "recall": R, "f1": F}


def main():
    root = require(NUBES_DIR, "NUBes / IULA+ corpus", "EE_NUBES_DIR")
    docs = load(root)
    forms_all = Counter(p.lower() for _, cs in docs for _, _, p in cs)
    v = {"J1.nubes.n_files": len(docs),
         "J1.nubes.n_files_with_cue": sum(1 for _, cs in docs if cs),
         "J1.nubes.n_cues": sum(len(cs) for _, cs in docs),
         "J1.nubes.n_forms": len(forms_all),
         "J1.nubes.lexicon_patterns": len(ES_LEXICON),
         "J1.nubes.lexicon_patterns_distinct": len(set(ES_LEXICON))}
    print(f"{v['J1.nubes.n_files']} files ({v['J1.nubes.n_files_with_cue']} with a cue), "
          f"{v['J1.nubes.n_cues']} cues, {v['J1.nubes.n_forms']} distinct forms")

    paper_re = [re.compile(p, re.IGNORECASE) for p in ES_LEXICON]
    runs = {"handbuilt": [], "derived19": [], "derived100": []}
    for seed in range(N_SPLITS):
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        cut = len(idx) // 2
        train = [docs[i] for i in idx[:cut]]
        test = [docs[i] for i in idx[cut:]]
        forms = Counter(p.lower() for _, cs in train for _, _, p in cs)
        runs["handbuilt"].append(score(test, paper_re))
        runs["derived19"].append(score(test, compile_forms([f for f, _ in forms.most_common(19)])))
        runs["derived100"].append(score(test, compile_forms([f for f, _ in forms.most_common(100)])))
        print(f"  split {seed}: hand-built cue recall {runs['handbuilt'][-1]['cue_recall']:.3f} | "
              f"derived-100 {runs['derived100'][-1]['cue_recall']:.3f}")

    for name, rs in runs.items():
        for m in ("cue_recall", "precision", "recall", "f1"):
            a = np.array([r[m] for r in rs])
            v[f"J1.nubes.{name}.{m}_mean"] = float(a.mean())
            v[f"J1.nubes.{name}.{m}_std"] = float(a.std(ddof=1))
            v[f"J1.nubes.{name}.{m}_min"] = float(a.min())

    out = RESULTS_DIR / "analysis" / "nubes_spanish.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(v, open(out, "w"), indent=2)
    for k, x in v.items():
        print(f"{k:45s} {x:.4f}" if isinstance(x, float) else f"{k:45s} {x}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
