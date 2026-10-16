"""Validate the clinical hedge extractor against radiologist uncertainty labels in RadGraph.

Paper: Appendix J.1 (RadGraph paragraph) and Sec. 6.7: over 3,622 observation-bearing sentences the
clinical extractor, restricted to spans with non-zero epistemic strength, reaches precision
0.570, recall 0.609, F1 0.588; F1 0.606 and 0.691 against each of the two radiologists on the test
split; precision 0.170 when every returned span counts. Inter-radiologist agreement on the test
split: 98.9% of 1,011 sentences, Cohen's kappa 0.879, F1 0.884.
Inputs: RadGraph 1.0.0 train.json, dev.json, test.json under EE_RADGRAPH_ROOT (only the 600
radiologist-annotated reports; the model-labelled MIMIC-CXR/CheXpert graphs are not used);
access: PhysioNet credentialed.
Outputs: RESULTS_DIR/analysis/radgraph_extractor.json (check ids J1.radgraph.*).
Usage:
    python scripts/validation/radgraph_extractor.py

Sentences are cut after every whitespace token that ends in "."; a report's tail without a
full stop is one more sentence. Extractor evaluation keeps sentences that contain at least one
observation entity (OBS-*) and calls them gold-uncertain when one of those is OBS-U. The pooled
set is train + dev + test as annotated by radiologist 1. The inter-radiologist figures use every
sentence of the 100 test reports (gold-uncertain = contains an OBS-U entity).
"""
import json
import sys
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import RADGRAPH_ROOT, RESULTS_DIR, require  # noqa: E402
from src.distortion_analyzer import get_epistemic_strength  # noqa: E402
from src.hedge_screener import find_hedges  # noqa: E402


def sentence_bounds(toks):
    bounds, s = [], 0
    for i, t in enumerate(toks):
        if t == "." or t.endswith("."):
            bounds.append((s, i))
            s = i + 1
    if s < len(toks):
        bounds.append((s, len(toks) - 1))
    return bounds


def token_labels(entities):
    lab = {}
    for e in entities.values():
        for i in range(e["start_ix"], e["end_ix"] + 1):
            lab.setdefault(i, set()).add(e["label"])
    return lab


def observation_sentences(text, entities):
    """(sentence, gold_uncertain) for each sentence that carries an observation entity."""
    toks, lab = text.split(), token_labels(entities)
    out = []
    for a, c in sentence_bounds(toks):
        obs = {x for i in range(a, c + 1) for x in lab.get(i, set()) if x.startswith("OBS")}
        if obs:
            out.append((" ".join(toks[a:c + 1]), "OBS-U" in obs))
    return out


def uncertain_flags(text, entities):
    """1 per sentence of the report if it contains an OBS-U entity, else 0 (all sentences)."""
    toks, lab = text.split(), token_labels(entities)
    return [int(any("OBS-U" in lab.get(i, set()) for i in range(a, b + 1))) for a, b in sentence_bounds(toks)]


def evaluate(pairs, epistemic_only):
    tp = fp = fn = tn = 0
    for text, gold in pairs:
        spans = find_hedges(text)
        if epistemic_only:
            spans = [h for h in spans if get_epistemic_strength(h.text) > 0]
        pred = len(spans) > 0
        tp += pred and gold
        fp += pred and not gold
        fn += (not pred) and gold
        tn += (not pred) and (not gold)
    P, R = tp / (tp + fp), tp / (tp + fn)
    return {"n": tp + fp + fn + tn, "gold": tp + fn, "precision": P, "recall": R, "f1": 2 * P * R / (P + R)}


def main():
    root = require(RADGRAPH_ROOT, "RadGraph 1.0.0", "EE_RADGRAPH_ROOT")
    pools = {}
    for split in ("train", "dev"):
        pools[split] = [x for r in json.load(open(require(root / f"{split}.json", f"RadGraph {split}.json",
                                                          "EE_RADGRAPH_ROOT"))).values()
                        for x in observation_sentences(r["text"], r["entities"])]
    test = list(json.load(open(require(root / "test.json", "RadGraph test.json", "EE_RADGRAPH_ROOT"))).values())
    pools["test_r1"] = [x for r in test for x in observation_sentences(r["text"], r["labeler_1"]["entities"])]
    pools["test_r2"] = [x for r in test for x in observation_sentences(r["text"], r["labeler_2"]["entities"])]
    pools["pooled"] = pools["train"] + pools["dev"] + pools["test_r1"]

    v = {}
    for name, pairs in pools.items():
        for mode, epi in (("epistemic", True), ("all_spans", False)):
            res = evaluate(pairs, epi)
            for k, x in res.items():
                v[f"J1.radgraph.{name}.{mode}.{k}"] = x

    a, b = [], []
    for r in test:
        a += uncertain_flags(r["text"], r["labeler_1"]["entities"])
        b += uncertain_flags(r["text"], r["labeler_2"]["entities"])
    v["J1.radgraph.interrater.n_sentences"] = len(a)
    v["J1.radgraph.interrater.agreement"] = sum(x == y for x, y in zip(a, b)) / len(a)
    v["J1.radgraph.interrater.kappa"] = float(cohen_kappa_score(a, b))
    v["J1.radgraph.interrater.f1"] = float(f1_score(a, b))
    v["J1.radgraph.interrater.positives_r1"] = sum(a)
    v["J1.radgraph.interrater.positives_r2"] = sum(b)
    v["J1.radgraph.interrater.r1_subset_of_r2"] = int(all(y >= x for x, y in zip(a, b)))

    out = RESULTS_DIR / "analysis" / "radgraph_extractor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(v, open(out, "w"), indent=2)
    for k, x in v.items():
        print(f"{k:55s} {x:.4f}" if isinstance(x, float) else f"{k:55s} {x}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
