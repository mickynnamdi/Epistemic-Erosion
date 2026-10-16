"""Validate the hedge extractors against BioScope's human speculation annotations.

Paper: Appendix J.1, "BioScope, all extractors" (abstracts, 11,871 sentences: scientific
extractor F1 0.559, P 0.628; clinical extractor F1 0.395, P 0.533, R 0.313; contextual
variant F1 0.082, P 0.405, R 0.046). Zero-shot: nothing is tuned on BioScope.
Inputs: BioScope abstracts.xml and full_papers.xml in EE_BIOSCOPE_DIR (public, research use;
download manually, see scripts/data/download_public_data.sh). CPU only.
Outputs: results/bioscope/bioscope_extractor_validation.json and results/analysis/bioscope.json.
Usage:
    python scripts/validation/bioscope_extractor_validation.py

Task: sentence-level speculation detection. Gold = 1 if the sentence contains at least one
human <cue type="speculation">; prediction = 1 if the extractor finds at least one hedge.
Also reported: cue-level recall (share of gold cue phrases overlapped by a predicted span).
"""
import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import BIOSCOPE_DIR as BIOSCOPE, RESULTS_DIR, require  # noqa: E402
from src.hedge_screener import find_hedges  # noqa: E402


def parse_bioscope(xml_path: Path):
    """Return list of {text, gold_spec, cue_spans:[(start,end,phrase)]}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sentences = []
    for sent in root.iter("sentence"):
        # Reconstruct plain text and record speculation-cue char spans by
        # walking the element tree in document order.
        parts = []
        cue_spans = []
        pos = 0

        def walk(el):
            nonlocal pos
            if el.text:
                parts.append(el.text)
                pos += len(el.text)
            for child in el:
                if child.tag == "cue" and child.get("type") == "speculation":
                    start = pos
                    ctext = "".join(child.itertext())
                    parts.append(ctext)
                    pos += len(ctext)
                    cue_spans.append((start, pos, ctext.strip()))
                    if child.tail:
                        parts.append(child.tail)
                        pos += len(child.tail)
                else:
                    walk(child)
                    if child.tail:
                        parts.append(child.tail)
                        pos += len(child.tail)

        walk(sent)
        text = "".join(parts)
        text = re.sub(r"\s+", " ", text).strip()
        sentences.append(
            {"text": text, "gold_spec": 1 if cue_spans else 0, "cue_spans": cue_spans}
        )
    return sentences


def spans_overlap(a, b):
    return not (a[1] <= b[0] or b[1] <= a[0])


def evaluate(sentences, detector):
    tp = fp = fn = tn = 0
    cue_total = 0
    cue_hit = 0
    for s in sentences:
        matches = detector(s["text"])
        pred = 1 if matches else 0
        gold = s["gold_spec"]
        if gold and pred:
            tp += 1
        elif gold and not pred:
            fn += 1
        elif not gold and pred:
            fp += 1
        else:
            tn += 1
        # cue-level recall (re-detect on cleaned text; approximate by phrase match)
        for (_, _, phrase) in s["cue_spans"]:
            cue_total += 1
            p = phrase.lower()
            if any(p in m.text.lower() or m.text.lower() in p for m in matches):
                cue_hit += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(len(sentences), 1)
    return {
        "n_sentences": len(sentences),
        "n_speculative": sum(s["gold_spec"] for s in sentences),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "accuracy": round(acc, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cue_level_recall": round(cue_hit / cue_total, 3) if cue_total else None,
        "cue_total": cue_total,
    }


def main():
    detectors = {"base_find_hedges": find_hedges}
    try:
        from src.hedge_screener_scientific import find_scientific_hedges
        detectors["scientific"] = find_scientific_hedges
    except Exception as e:
        print(f"(scientific screener not loaded: {e})")
    try:
        from src.hedge_screener_contextual import find_contextual_hedges
        detectors["contextual_clinical"] = lambda t: find_contextual_hedges(t, domain="clinical")
    except Exception as e:
        print(f"(contextual screener not loaded: {e})")

    out = {}
    require(BIOSCOPE / "abstracts.xml", "BioScope abstracts.xml", "EE_BIOSCOPE_DIR")
    for corpus in ["abstracts.xml", "full_papers.xml"]:
        path = BIOSCOPE / corpus
        if not path.exists():
            print(f"skip {corpus} (missing)")
            continue
        sents = parse_bioscope(path)
        out[corpus] = {}
        for name, det in detectors.items():
            res = evaluate(sents, det)
            out[corpus][name] = res
            print(f"\n=== {corpus} / {name} ===")
            print(json.dumps(res, indent=2))

    (RESULTS_DIR / "bioscope").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "bioscope" / "bioscope_extractor_validation.json").write_text(json.dumps(out, indent=2))
    names = {"base_find_hedges": "clinical", "scientific": "scientific", "contextual_clinical": "contextual"}
    checks = {f"J1.bioscope.{names[d]}.{k}": r[k] for d, r in out.get("abstracts.xml", {}).items()
              for k in ("n_sentences", "precision", "recall", "f1")}
    json.dump(checks, open(RESULTS_DIR / "analysis" / "bioscope.json", "w"), indent=1)
    print(f"\nWrote {RESULTS_DIR / 'bioscope' / 'bioscope_extractor_validation.json'}")


if __name__ == "__main__":
    main()
