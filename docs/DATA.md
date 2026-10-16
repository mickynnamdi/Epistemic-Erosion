# Data

**This repository ships no data.** Every input is either downloaded from its public source by
a script, or rebuilt from PhysioNet data that you obtain under your own credentialed access.
The only data files in the repository are `data/hosted_samples/seed{0,1,2}.json`: the lists of
public document ids (Multi-LexSum case ids, SciTLDR paper ids, PubMed ids) used in the
hosted-model study.

All locations are set in one place, `configs/paths.py`. You can override any of them with an
environment variable, either exported in your shell or written in `.env` (copy
`configs/paths.example.env`). A script that cannot find an input stops and names the variable to set.

## Overview

| Dataset | Used for | Access | Variable | How to get it | Sample file(s) built |
|---|---|---|---|---|---|
| MIMIC-IV v3.1 + MIMIC-IV-Note v2.2 | clinical studies (Sec. 6.1-6.4, 6.8, most appendices) | **PhysioNet credentialed** | `EE_MIMIC_ROOT` | physionet.org, see below | `data/full_sample.parquet` (seed 123, 192 episodes), `full_sample_seed0.parquet` (184), `full_sample_seed1.parquet` (184) |
| RadGraph 1.0.0 | Sec. 6.7, App. J.1, J.2 | **PhysioNet credentialed** | `EE_RADGRAPH_ROOT` | physionet.org/content/radgraph/1.0.0 | read directly |
| Multi-LexSum v20230518 | legal domain (Sec. 6.5, Table 2) | public | `EE_EXTERNAL_DIR` | `scripts/data/download_public_data.sh` | `data/legal_sample.parquet` (192), pool `legal_sample_full.parquet` (1,881) |
| SciTLDR (AIC) | scientific domain (Sec. 6.5, Table 2) | public | `EE_EXTERNAL_DIR` | `scripts/data/download_public_data.sh` | `data/scientific_sample.parquet` (192), pool `scientific_sample_full.parquet` (2,615) |
| PubMedQA `pqa_labeled` | biomedical domain of Table 2 | public | — | downloaded from the Hugging Face Hub by the builder | `data/medical_sample.parquet` (300) |
| GEM Cochrane simplification (train) | multilingual probe (App. F.7) | public | — | downloaded by `multilingual_probe.py --stage data` | `data/cochrane_sample.parquet` (400) |
| NUBes / IULA+ | Spanish extractor validation (App. J.1) | public | `EE_NUBES_DIR` | `scripts/data/download_public_data.sh` | read directly |
| BioScope | extractor validation (App. J.1) | public (research use) | `EE_BIOSCOPE_DIR` | manual download, see below | read directly |

Every builder was checked against the files used for the paper: the samples it rebuilds are
identical (same ids, same order, same text).

## PhysioNet data (credentialed)

1. **Get access.** Obtain credentialed access at https://physionet.org (CITI "Data or Specimens
   Only Research" training). Sign the data use agreements of **MIMIC-IV v3.1**,
   **MIMIC-IV-Note v2.2** and, for Section 6.7 / Appendix J.1-J.2, **RadGraph 1.0.0**. All three
   are under the PhysioNet Credentialed Health Data License 1.5.0.
2. **Download.** Use PhysioNet's `wget -r` layout. `EE_MIMIC_ROOT` must contain
   `mimic-iv-note/2.2/note/discharge.csv.gz`, `mimiciv/3.1/hosp/{admissions,services,diagnoses_icd,d_icd_diagnoses}.csv.gz`
   and `mimiciv/3.1/icu/icustays.csv.gz`.
3. **Rebuild the three depth-study samples.** This takes a few minutes each and about 16 GB of RAM:

   ```bash
   python -m src.build_mimic_sample --seed 123   # data/full_sample.parquet        (192 episodes)
   python -m src.build_mimic_sample --seed 0     # data/full_sample_seed0.parquet  (184)
   python -m src.build_mimic_sample --seed 1     # data/full_sample_seed1.parquet  (184)
   ```

**The 1,000-episode breadth sample** (`data/full_sample_1000.parquet`) has no builder in this
package. It is needed for the breadth study (Appendix B specialty test, Table A1) and for the
identity notes of the judge reference set (Appendix I, Table A8).

**Anything derived from MIMIC-IV or RadGraph is covered by the same agreement.** That includes:
- model outputs on clinical notes and radiology reports;
- the judge reference sets;
- the detector's training pairs and checkpoints;
- the clinical annotation sheet;
- the files the scripts write under `results/`.

Keep them local and never publish them. PhysioNet's MIMIC-IV page states: "Any derived datasets
or models should be treated as containing sensitive information. If you wish to share these
resources, they should be shared on PhysioNet under the same agreement as the source data."

**Sending MIMIC-derived text to hosted APIs.** Some validation scripts send MIMIC-derived text
to the Anthropic API (Claude judges and the clinician proxy). PhysioNet's post "Responsible use
of MIMIC data with online services like GPT" forbids sending credentialed data to services that
may use or review it. It lists Anthropic Claude among the acceptable options, because prompts are
not used for training by default and are not routinely reviewed. Check your own agreement and
institutional rules before running those stages.

## Public data

```bash
bash scripts/data/download_public_data.sh        # Multi-LexSum, SciTLDR, NUBes (pinned versions)
python scripts/data/sample_external_datasets.py  # legal + scientific samples (add --pools for Table A14)
python scripts/data/build_medical_sample.py      # PubMedQA sample
python scripts/analysis/multilingual_probe.py --stage data   # Cochrane sample (Appendix F.7 only)
```

**BioScope** has no direct download link. Download it from https://rgai.inf.u-szeged.hu/node/105,
then put `abstracts.xml` and `full_papers.xml` in `data/external/bioscope/`, or set
`EE_BIOSCOPE_DIR`.

**Licences.** These corpora are not redistributed here; check each licence before any reuse.

| Corpus | Licence |
|---|---|
| Multi-LexSum | Dataset ODC-By 1.0; summaries and metadata CC BY-NC 4.0; source documents public domain |
| SciTLDR | Apache-2.0 |
| PubMedQA | MIT |
| GEM Cochrane simplification | CC BY 4.0 |
| NUBes / IULA+ | CC BY-SA 3.0 ES |
| BioScope | Publicly available for research purposes |


## Sampling details

- **Legal / scientific:** seed 42, 48 documents per stratum, and at least 3 domain hedges in the
  source. Legal strata are the 4 most frequent case types; scientific strata are hedge-count quartiles.
- **PubMedQA:** abstracts of 600-12,000 characters with at least 2 clinical hedges (378
  candidates), stratified into hedge-density terciles, `random_state=42`.
- **Hosted-model seeds** (`data/hosted_samples/`):
  - Seed 0 draws 100 documents per domain with `random.Random(42)`.
  - Seeds 1 and 2 use `random.Random(seed)`. The biomedical lists exclude the documents of
    earlier seeds; the legal and scientific samples have only 192 documents, too few to exclude
    100 per seed, so their lists are drawn from the whole sample and overlap across seeds.
  - `python scripts/hosted/build_samples.py` re-draws them and confirms they match the shipped
    lists. The exception is seed 0's biomedical list, which was drawn from an earlier version of
    the PubMedQA sample and is therefore shipped as data.
