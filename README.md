# Epistemic Erosion: An Error Class Beyond Hallucination in LLM Text Generation

Code for the AACL-IJCNLP 2026 paper. When an LLM summarises or rewrites professional text, it
often changes how certain the source is ("possible pneumonia" → "pneumonia"), even when the facts
survive. The paper calls this **epistemic erosion** and measures it with the **Epistemic
Faithfulness Score (EFS)**:

    EFS = max(0, 1 − |D(S, G)| / max(|H(S)|, 1))

Here `H(S)` is the set of uncertainty markers in the source, and `D(S, G)` is the set of
distortions detected in the output. There are six distortion types: escalation, attenuation,
omission, fabrication, scope shift and severity warp.

This repository contains the EFS pipeline and all code behind the paper's numbers: sampling,
generation, scoring, statistics, validations, ablations, figures, and a check that compares your
results with every value printed in the paper. It contains no data and no model weights
(see [docs/DATA.md](docs/DATA.md) and [docs/MODELS.md](docs/MODELS.md)).

## Install

```bash
git clone https://github.com/mickynnamdi/Epistemic-Erosion && cd Epistemic-Erosion
bash setup.sh                       # creates venv/ from requirements.txt  (--locked: exact tested versions)
source venv/bin/activate
cp configs/paths.example.env .env   # optional: where your data, outputs and PhysioNet files live
python -m pytest tests/             # 12 tests run immediately; the others run once data exist
```

Python 3.10 or later (tested on 3.13). A GPU is needed only for generating model outputs, the
token probe and the learned detector.

## Score a pair

```python
from src.efs import compute_efs

r = compute_efs(source_text, output_text, domain="clinical")   # or "legal", "scientific"
r["efs_score"]      # 0..1, 1 = no distortion detected
r["by_type"]        # {"omission": 2, "escalation": 1, ...}
r["distortions"]    # one record per detected distortion, with source and output spans
```

Always pass `domain`. Scoring is rule-based and CPU-only: about 20 ms for a 12,000-character clinical note.

## Reproduce the paper

[docs/REPRODUCE.md](docs/REPRODUCE.md) maps every table, figure and appendix number to the
script that produces it, and says which need PhysioNet access, a GPU or a paid API. In short:

```bash
bash scripts/data/download_public_data.sh && python scripts/data/sample_external_datasets.py
python -m src.build_mimic_sample --seed 123          # MIMIC-IV samples (PhysioNet credentialed)
python scripts/generate/generate.py --study depth --seed 123 --model llama-3.1-8b   # model outputs
python scripts/score_generations.py --study all      # EFS for every output
python scripts/make_tables.py                        # all tables + results/PAPER_CHECK.md
```

`results_summary/` holds the result of this check on the paper's own generations.

## Repository layout

```
src/                 EFS pipeline: hedge lexicons, claim extraction, one-to-one claim aligner,
                     distortion typing (efs.py is the entry point); MIMIC sampler; generator; detector
configs/             paths.py (all locations), models.py (models + pinned revisions), studies.py
prompts/             every prompt used in the paper (Appendix S; hosted_prompts.py for Table 2)
scripts/data/        download public corpora, draw the samples
scripts/generate/    open-weight generation for every study
scripts/hosted/      hosted-model (Azure) generation for Table 2
scripts/score_generations.py   EFS for every generation of a study
scripts/analysis/    statistics, ablations and robustness analyses
scripts/validation/  extractor, ground-truth, annotator, RadGraph and metric validations
scripts/judges/      LLM-judge analyses (App. I, J.7, J.8)
scripts/decision/    clinician-proxy decision studies (App. F.5, F.6)
scripts/probe/       token-distribution probe (Sec. 6.8, App. M)
scripts/detector/    learned detector training and transfer (App. L, N)
scripts/figures/     Figure A1 (Figure 2: scripts/analysis/metric_blindness.py)
scripts/make_tables.py         every table, and the comparison with the paper
data/hosted_samples/ document ids of the hosted-model study (public ids only)
docs/                DATA.md, MODELS.md, REPRODUCE.md
results_summary/     result of the check on the paper's generations (aggregate numbers only)
tests/               pytest suite
```

## Citation

```bibtex
@inproceedings{nnamdi2026epistemic,
  title     = {Epistemic Erosion: An Error Class Beyond Hallucination in LLM Text Generation},
  author    = {Nnamdi, Micky C. and Wang, Shaunna and Nnamdi, Chinaza and Li, Jieru and Marteau, Benoit L. and Tamo, J. Ben and Yuan, Yining and Wang, May D.},
  booktitle = {Proceedings of AACL-IJCNLP 2026 (Main Conference)},
  year      = {2026}
}
```

Code: MIT licence ([LICENSE](LICENSE)). Data and models keep their own licences (docs/DATA.md, docs/MODELS.md).
