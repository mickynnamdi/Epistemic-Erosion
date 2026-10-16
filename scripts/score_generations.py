"""Score every generation of a study with the EFS pipeline.

Reads the generation JSONs of a study (locations in configs/studies.py), pairs
each with its source document from the rebuilt samples in EE_DATA_DIR
(truncated to 12,000 characters, as at generation time), and writes one row per
generation to results/scores/<study>.csv (or <study>_greedy.csv).

Usage
    python scripts/score_generations.py --study depth          # Table 1, A18 ...
    python scripts/score_generations.py --study all
    python scripts/score_generations.py --study depth --aligner greedy   # Appendix J ablations

Row status
    ok          scored
    error       the generation call failed or returned no text (e.g. a content-filter
                rejection of a hosted model); excluded from every analysis
    no_source   the document is not in the sample file; excluded
Columns `source_hash_ok` (clinical runs only) records whether the text the model saw
equals the truncated source note used for scoring.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import OUTPUTS_DIR, RESULTS_DIR, require  # noqa: E402
from configs.studies import STUDIES, sample_path  # noqa: E402

MAX_SOURCE_CHARS = 12000
TYPES = ["omission", "escalation", "attenuation", "fabrication", "scope_shift", "severity_warp"]
MODEL_ALIASES = {"gpt-4o-2": "gpt-4o"}   # one hosted seed used a second deployment name

_SRC = {}          # (sample file, id) -> truncated source text; filled per worker
_ALIGNER = "hungarian"


def _load_sources(part):
    df = pd.read_parquet(require(sample_path(part["sample"]), f"sample {part['sample']}", "EE_DATA_DIR"))
    if "discharge_text" in df:   # MIMIC samples, keyed by hadm_id
        return {str(int(h)): t[:MAX_SOURCE_CHARS] for h, t in zip(df["hadm_id"], df["discharge_text"])}
    return {str(i): t[:MAX_SOURCE_CHARS] for i, t in zip(df[part["id_col"]], df["source_text"])}


def _doc_id(d, path):
    # doc_id when present (breadth / cross-domain / hosted files); hadm_id otherwise
    for key in ("doc_id", "hadm_id"):
        if d.get(key) not in (None, "", "None"):
            v = d[key]
            return str(int(v)) if isinstance(v, (int, float)) else str(v)
    return os.path.basename(path).split("_")[0]


def _init(sources, aligner):
    global _SRC, _ALIGNER
    _SRC, _ALIGNER = sources, aligner


def _score(job):
    from src.efs import compute_efs, compute_efs_greedy
    path, key, meta = job
    d = json.load(open(path))
    doc = _doc_id(d, path)
    # breadth-study files carry an unrelated number in hadm_id; the file name holds the episode id
    if meta.get("id_from_filename"):
        doc = os.path.basename(path).split("_")[0]
    out = d.get("output_text") or ""
    err = d.get("error")
    row = dict(meta, file=os.path.basename(path), doc_id=doc,
               model=MODEL_ALIASES.get(d.get("model"), d.get("model")), task=d.get("task"),
               condition=(f"T={d['temperature']}" if "temperature" in d else (d.get("mitigation") or "none")),
               output_tokens=d.get("output_tokens", d.get("completion_tokens")),
               out_words=len(out.split()))
    src = _SRC.get((key, doc))
    if src is None:
        return dict(row, status="no_source")
    if d.get("source_text_hash"):
        row["source_hash_ok"] = hashlib.md5(src.encode()).hexdigest() == d["source_text_hash"]
    if (err and str(err) != "None") or not out.strip():
        return dict(row, status="error", error=str(err)[:200] if err else "empty output")
    fn = compute_efs_greedy if _ALIGNER == "greedy" else compute_efs
    r = fn(src, out, domain=meta["pipeline_domain"])
    row.update(status="ok", efs=r["efs_score"], n_distortions=r["distortion_count"],
               source_hedges=r["source_hedges"], **{t: r["by_type"].get(t, 0) for t in TYPES})
    return row


def jobs_for(study):
    spec = STUDIES[study]
    sources, jobs = {}, []
    for i, part in enumerate(spec["parts"]):
        domain = part.get("domain", spec.get("domain"))
        pipeline_domain = part.get("pipeline", domain)
        key = f"{study}:{i}"
        for doc, text in _load_sources(part).items():
            sources[(key, doc)] = text
        gdir = require(OUTPUTS_DIR / part["dir"], f"generations of study '{study}'", "EE_OUTPUTS_DIR")
        meta = dict(study=study, domain=domain, pipeline_domain=pipeline_domain, seed=part.get("seed"),
                    id_from_filename=(study == "breadth"))
        for f in sorted(glob.glob(str(gdir / "*" / "*.json"))):
            jobs.append((f, key, meta))
    return sources, jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True, choices=list(STUDIES) + ["all"])
    ap.add_argument("--aligner", default="hungarian", choices=["hungarian", "greedy"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    out_dir = RESULTS_DIR / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    for study in (STUDIES if a.study == "all" else [a.study]):
        try:
            sources, jobs = jobs_for(study)
        except SystemExit as e:   # an input is missing: with --study all, score the other studies
            if a.study != "all":
                raise
            print(f"{study:12s} skipped: {e}")
            continue
        if not jobs:
            print(f"{study:12s} skipped: no generation files found under EE_OUTPUTS_DIR")
            continue
        with Pool(a.workers, initializer=_init, initargs=(sources, a.aligner)) as pool:
            rows = pool.map(_score, jobs, chunksize=16)
        df = pd.DataFrame(rows).drop(columns=["id_from_filename"])
        suffix = "_greedy" if a.aligner == "greedy" else ""
        path = out_dir / f"{study}{suffix}.csv"
        df.to_csv(path, index=False)
        print(f"{study:12s} {len(df):6d} generations -> {path}   status: {df['status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
