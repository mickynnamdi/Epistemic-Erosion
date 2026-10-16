# Models

No model weights are included in this repository.

- **Open-weight models** are used exactly as published, without fine-tuning. Download them from
  the Hugging Face Hub at the pinned commit below.
- **Hosted models** were called through Azure.
- **The learned detector** was trained on MIMIC-derived data, so its weights may not be
  published outside PhysioNet (see the last section).

## Open-weight models

`configs/models.py` holds this registry. `src/generator.py` loads a model from `EE_MODELS_ROOT/<local folder>`
when that folder exists, and otherwise from the Hub at the pinned commit. Each commit below was
checked against the Hub; it is the snapshot of the checkpoints used for the
paper's runs.

| Name in code | Paper name | Hugging Face repository | Pinned commit | Licence | Used in |
|---|---|---|---|---|---|
| `llama-3.1-8b` | Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct (gated) | `0e9e39f249a16976918f6564b8830bc894c89659` | Llama 3.1 | all clinical studies, cross-domain, probe |
| `llama-3.3-70b` | Llama-3.3-70B | meta-llama/Llama-3.3-70B-Instruct (gated) | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` | Llama 3.3 | depth study; LLM judge (App. I, J.2, J.8); clinician proxy (F.5, F.6); AI annotator (J.9) |
| `med42-8b` | Med42-v2-8B | m42-health/Llama3-Med42-8B | `ceab7e7ee4b9dbde7ba82867f34274db51487d83` | Llama 3 | as Llama-3.1-8B |
| `med42-70b` | Med42-v2-70B | m42-health/Llama3-Med42-70B | `867064e18aad7bf3f4795f20dcb25a1108952543` | Llama 3 | depth study |
| `openbio-70b` | OpenBioLLM-70B | aaditya/Llama3-OpenBioLLM-70B | `7ad17ef0d2185811f731f89d20885b2f99b1e994` | Llama 3 | depth study |
| `mistral-7b` | Mistral-7B-Instruct | mistralai/Mistral-7B-Instruct-v0.3 | `c170c708c41dac9275d15a8fff4eca08d52bab71` | Apache-2.0 | depth, breadth, mitigation, cross-domain, RadGraph, probe |
| `biomistral-7b` | BioMistral-7B | BioMistral/BioMistral-7B-DARE | `588fca722fbffc46eab97cbe610d4203f303add8` | Apache-2.0 | depth study, RadGraph |
| `medgemma-27b` | MedGemma-27B | google/medgemma-27b-it | `2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f` | Health AI Developer Foundations | depth study |
| `gemma-4-4b` | Gemma-4-E4B | google/gemma-4-E4B-it | `ab4d2025e9fbb0b0dbbf5dbde081d3e546155586` | Apache-2.0 | depth, mitigation, multilingual |
| `gemma-4-31b` | Gemma-4-31B | google/gemma-4-31B-it | `e51e7dcdb6febd74c182fe0cb41c236363ae2ac5` | Apache-2.0 | depth; AI annotator (J.9) |
| `gemma-4-moe` | Gemma-4-26B-MoE | google/gemma-4-26B-A4B-it | `1db3cff1840c2ae59759d8e842ff37831cf8cb63` | Apache-2.0 | depth study |
| `gemma-3-4b` | Gemma-3-4B | google/gemma-3-4b-it (gated) | `093f9f388b31de276ce2de164bdc2081324b9767` | Gemma | probe (App. M) |
| `gpt-oss-20b` | gpt-oss-20B | openai/gpt-oss-20b | `6cee5e81ee83917806bbde320786a8fb61efebee` | Apache-2.0 | AI annotator (J.9) |

**Hardware.**
- The 7-8B models run on one GPU with 24 GB of memory.
- The 26-31B models need roughly 60-80 GB.
- The 70B models need about 140 GB in bfloat16 (2 x 80 GB). A single 141 GB card is not enough
  once activations are included.

**Decoding.** Greedy decoding (`do_sample=False`) is not bit-exact across GPU types and library
versions. Regenerated outputs therefore differ in wording from the paper's, while study-level EFS
agrees closely.

## Hosted models (Table 2)

All four were called through Azure OpenAI / Azure AI Foundry (API version 2024-10-21) by
`scripts/hosted/generate.py`. Temperature was 0 with at most 1,024 output tokens. The GPT-5.x
family accepts no temperature parameter, so it ran at the API default with
`max_completion_tokens=1024`.

| Paper name | Azure deployment used | Notes |
|---|---|---|
| Phi-4 | `Phi-4` | |
| GPT-4o | `gpt-4o-2` (seed 0), `gpt-4o` (seeds 1-2) | Two deployments; whether they served the same model version was not recorded. |
| Mistral-Large-3 | `Mistral-Large-3` | |
| "GPT-5" | `gpt-5.4` | Azure lists gpt-5.4 (version 2026-03-05) as a model distinct from gpt-5. The outputs of Table 2 come from the `gpt-5.4` deployment. |

The hosted models received the prompts in `prompts/hosted_prompts.py`, not those of
Appendix S.

## Learned detector (Section 4.3, Appendix N, Tables A14/A16)

**The checkpoints are not distributed.** The encoders were fine-tuned on sentence pairs that
contain verbatim MIMIC-IV discharge-note text. They are therefore derived from PhysioNet
credentialed data. PhysioNet's MIMIC-IV page (announcement "Guidelines for creating datasets
and models from MIMIC") says that derived models "should be treated as containing sensitive
information" and may be shared only "on PhysioNet under the same agreement as the source data".
The authors release the code and the exact recipe instead.

**To retrain** (about 2 hours per seed for the four encoders on one H200, i.e. about 6 GPU-hours for all 12 runs):

```bash
python -m src.learned_detector generate_data     # results/learned_detector/{train,val}.csv (62k pairs)
python scripts/detector/train.py --all-seeds     # 4 encoders x seeds 42, 123, 2024
python scripts/analysis/detector_table.py        # Table A16
python scripts/detector/transfer.py --seed 42    # Table A14 (needs the pool generations, see REPRODUCE.md)
```

**Settings and encoders.** Hyperparameters are AdamW, lr 3e-5, 5 epochs, maximum length 256,
batch 16, class-weighted cross-entropy and 10% linear warm-up. The encoders, whose Hugging Face
revisions were not recorded, are:
- `roberta-base` (MIT)
- `roberta-large` (MIT)
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (MIT)
- `allenai/scibert_scivocab_uncased`

**Expected agreement.** Retrained mean accuracy lands within about half a point of Table A16
(88.5 / 88.4 / 88.5 / 88.7 for RoBERTa-base / RoBERTa-large / PubMedBERT / SciBERT over the three
seeds, against 88.7 / 88.5 / 88.8 / 88.8 in the paper). Macro-F1 lands within 0.01 and per-type
F1 within about 0.02 of the paper (attenuation F1 shows the largest gap, about 0.02 lower). It is not exact, because the train/validation split depends on the order of the input files.
