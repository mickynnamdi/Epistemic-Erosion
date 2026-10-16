# Reproducing the paper

The pipeline has five steps. Each can be run on its own, and each writes under `data/`,
`outputs/` or `results/`.

```
1. data       rebuild the samples                       scripts/data/*, src/build_mimic_sample.py
2. generate   model outputs (GPU / paid API)            scripts/generate/generate.py, scripts/hosted/generate.py
3. score      EFS for every output                      scripts/score_generations.py
4. analyse    statistics, validations, ablations        scripts/analysis/*, scripts/validation/*, ...
5. check      every table + comparison with the paper   scripts/make_tables.py  ->  results/PAPER_CHECK.md
```

`scripts/make_tables.py` recomputes every table from your outputs and lists each printed paper
value next to your value, marked `match`, `MISMATCH` or `approximate` (with the reason).
Checks whose inputs you have not produced are listed as "not computed".
- **With the paper's generations** (not distributed: most are MIMIC-derived), every value
  reproduces; `PAPER_CHECK.md` gives the reason for the few printed values that differ.
- **With outputs you generate yourself**, values are close but not identical, because greedy GPU
  decoding is not bit-exact. In addition, 192 of the paper's 18,480 depth generations
  (Llama-3.1-8B and Med42-v2-8B, seed 123, 32 long notes x 3 tasks) were produced from a source
  longer than 12,000 characters; `generate.py` truncates at 12,000, so their input differs. The
  score tables flag them in the `source_hash_ok` column.

## Access levels

- **public**: runs with public data only.
- **credentialed**: needs PhysioNet access (MIMIC-IV or RadGraph).
- **GPU**: needs model inference; *70B* means about 140 GB of GPU memory.
- **API**: needs paid Anthropic or Azure calls.

## Step 1-3: data, generations, scores

| Study | Generate | Access | Score |
|---|---|---|---|
| Depth (Table 1, A18; 11 models x 3 tasks x 3 samples) | `generate.py --study depth --seed {123,0,1} --model M` | credentialed, GPU (up to 70B) | `score_generations.py --study depth` |
| Breadth (App. B specialty test) | `generate.py --study breadth --model M` | credentialed, GPU; needs `full_sample_1000.parquet` (no builder) | `--study breadth` |
| Mitigation (Tables A5-A7) | `generate.py --study mitigation --model M --strategy S` | credentialed, GPU | `--study mitigation` |
| Temperature (Table A5) | `generate.py --study temperature --temperature {0.3,0.7}` | credentialed, GPU | `--study temperature` |
| Cross-domain (Sec. 6.5) | `generate.py --study crossdomain --domain {legal,scientific} --model M` | public, GPU | `--study crossdomain` |
| Hosted models (Table 2) | `scripts/hosted/generate.py --deployment D --seed {0,1,2}` | public, API (Azure) | `--study hosted` |

Model names and deployments are listed in docs/MODELS.md. The Appendix J ablations need a second
scoring pass with the greedy baseline aligner:
`score_generations.py --study {depth,crossdomain,hosted} --aligner greedy`.

## Step 4: analyses, by paper element

| Paper element | Command | Access |
|---|---|---|
| Table 1, A18; Sec. 6.1-6.2; Table A1; App. B; Tables A5-A7; Sec. 6.5; Table 2, A2; Tables A4 and App. C, E, F.1, J.3 (clamp), Q; Figure A1 | computed by `scripts/make_tables.py` from the scores; `scripts/figures/make_figures.py` for Figure A1 (Figure 2: `scripts/analysis/metric_blindness.py --figure`) | as the scores |
| App. D; J.1 (marker shares); J.4 (marker counts); J.5 (greedy aligner, threshold sweep); J.6; Table A11 | `scripts/analysis/aligner_and_markers.py` | credentialed |
| App. J.4 restricted marker set (EFS 0.366) | `scripts/analysis/restricted_markers.py` | credentialed |
| Sec. 6.3 / Figure 2 (ROUGE-L, BERTScore) | `scripts/analysis/metric_blindness.py --stage metrics`, then `--stage analyze --figure` | credentialed, GPU (BERTScore) |
| App. F.2 metric sensitivity | `scripts/validation/metric_sensitivity.py` | credentialed |
| App. F.3 hedge restoration | `scripts/analysis/hedge_restoration.py` | credentialed |
| App. F.4 compression control | `scripts/analysis/compression_control.py` | public; exact values need the earlier generation run (see below) |
| App. F.5 clinician proxy | `scripts/decision/clinician_proxy.py --stage pairs / proxy / identity / harm / analyze` | credentialed, API, 70B |
| App. F.6 de-hedging decisions | `scripts/decision/dehedge_decisions.py --stage run / analyze` | credentialed, 70B |
| App. F.7 multilingual probe | `scripts/analysis/multilingual_probe.py --stage data / generate / analyze` | public, GPU |
| App. I, Table A8 (judge reference set) | `scripts/judges/build_judge_set.py --truncation-stress-test`, then `reference_set_judge.py --stage judge --stratum {identity,positives,truncation}` and `--stage analyze`, and `positive_control.py --stage judge / analyze` | credentialed, 70B, API; needs `full_sample_1000.parquet` |
| Sec. 6.6, App. J.8 (directional judge) | `scripts/judges/directional_judge.py --stage judge --domain D --model M`, then `--stage analyze` | credentialed, 70B; scientific shards: see below |
| Sec. 6.7, App. J.2, Table A9 (RadGraph erosion) | `scripts/validation/radgraph_erosion.py --stage summarize / judge / analyze` | credentialed, GPU, 70B |
| App. J.1 extractor vs radiologists (RadGraph) | `scripts/validation/radgraph_extractor.py` | credentialed |
| App. J.1 extractor vs BioScope | `scripts/validation/bioscope_extractor_validation.py` | public |
| App. J.1 Spanish (NUBes / IULA+) | `scripts/validation/nubes_spanish.py` | public |
| App. J.3 properties; Table A10 | `scripts/validation/efs_properties.py`; `scripts/validation/discriminance.py` | credentialed |
| App. J.3 marker coverage (39%, 68%) | `scripts/analysis/claim_coverage.py` | credentialed |
| App. J.7 / Table A12 controlled ground truth (and J.5, greedy) | `scripts/validation/controlled_groundtruth_validation.py [--aligner greedy]` | credentialed |
| App. J.7 Claude severity judge | `scripts/judges/controlled_gt_claude.py --stage judge / analyze` | credentialed, API |
| App. J.9 / Table A13 AI annotators | `scripts/validation/multi_annotator.py --model M`, then `multi_annotator_agreement.py` | the clinical sheet is credentialed; API / GPU |
| App. K human pilot | `scripts/validation/human_pilot.py --raters R1 R2 R3` | annotator files (withheld: they identify the raters) |
| Sec. 6.8, App. M, Table A15 (probe) | `scripts/probe/certainty_prior_probe.py --model {llama-3.1-8b,mistral-7b,med42-8b,gemma-3-4b}` | credentialed, GPU |
| Sec. 6.9, App. N, Table A16 (detector) | `python -m src.learned_detector generate_data`, `scripts/detector/train.py --all-seeds`, `scripts/analysis/detector_table.py` | credentialed, GPU |
| App. L, Table A14 (detector transfer) | `sample_external_datasets.py --pools`; `generate.py --study crossdomain --pool --domain D --model llama-3.1-8b --limit 200`; `scripts/detector/transfer.py` | public + detector checkpoints |

Scripts with several stages (`--stage`) can re-run their final `analyze` stage on intermediate
files, so the statistics can be checked without GPUs or API calls.

## Step 5: check

```bash
python scripts/make_tables.py        # writes results/tables/*.csv and results/PAPER_CHECK.md
python -m pytest tests/              # includes the headline values once results/scores exist
```

`results_summary/` holds the result of this check on the paper's own
generations: of 783 printed values it recomputes, 774 match and 9
are approximate (token-probe numerical precision).

## If generation is unexpectedly slow

Set `EE_FORCE_GPU=1` before `generate.py`. The Hugging Face loader's default device map
can place layers on the CPU when it judges GPU memory insufficient (for example
when a leftover process still holds memory on the device); generation then runs 20-30x
slower with no error. With the flag, the whole model is placed on GPU 0 and loading fails
with an error if it does not fit.

## Not reproducible from this package

- **Claude judge identity at 6,000 characters** (0.5% flagged, App. I): not included.
- **Breadth sample** (`full_sample_1000.parquet`): its builder is not included. The breadth
  study (Table A1) and the judge reference set (App. I, Table A8) run once you have the file.
- **Earlier cross-domain generation run** (`outputs/cross_domain/`): App. F.4 reads model compression
  from it, and the scientific shards of App. J.8 were judged on it. It is not distributed and
  `generate.py` does not produce it; without it both scripts use the scored cross-domain
  generations instead, which gives close but not identical values (conclusions unchanged).
- **Human annotations** (App. K, Table A13 vs human): withheld, because the files identify the
  raters. The scripts run on your own annotation files.
