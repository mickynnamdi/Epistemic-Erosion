#!/usr/bin/env bash
# Download the public corpora used by the paper into $EE_EXTERNAL_DIR (default: $EE_DATA_DIR/external,
# i.e. data/external). The directory is resolved by configs/paths.py, so .env and EE_ROOT apply.
# Nothing here needs credentials. MIMIC-IV and RadGraph are not downloaded (PhysioNet
# credentialed access; see docs/DATA.md). BioScope has no direct download link: see below.
#
# Usage:  bash scripts/data/download_public_data.sh
# Then:   python scripts/data/sample_external_datasets.py      # legal + scientific samples
#         python scripts/data/build_medical_sample.py          # PubMedQA sample (downloads from the HF Hub)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXT="$(cd "$ROOT" && python3 -c 'from configs.paths import EXTERNAL_DIR; print(EXTERNAL_DIR)')"
mkdir -p "$EXT"

# Multi-LexSum release v20230518 (Shen et al., 2022). Dataset ODC-By 1.0; summaries and metadata
# CC BY-NC 4.0; source documents public domain.
MLS_REV=055f5fa5300a5fdefb5b691c4d6c1e58f485b379
MLS="$EXT/multilexsum/releases/v20230518"
mkdir -p "$MLS"
for f in train dev test sources; do
    if [ ! -s "$MLS/$f.json" ]; then
        echo "Multi-LexSum: $f.json"
        curl -fL --retry 3 -o "$MLS/$f.json" \
            "https://huggingface.co/datasets/allenai/multi_lexsum/resolve/$MLS_REV/releases/v20230518/$f.json"
    fi
done

# SciTLDR (Cachola et al., 2020), Apache-2.0 repository.
SCI="$EXT/scitldr/scitldr_data"
if [ ! -d "$SCI/.git" ]; then
    git clone -q https://github.com/allenai/scitldr "$SCI"
fi
git -C "$SCI" checkout -q 5ccad9c00a60ad75c9e04abf7f27d0f53f983b20

# NUBes / IULA+ (Lima Lopez et al., 2020), CC BY-SA 3.0 ES. Used only by Appendix J.1.
NUB="$EXT/nubes"
if [ ! -d "$NUB/.git" ]; then
    git clone -q https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus "$NUB"
fi
git -C "$NUB" checkout -q 91c81c1f5eee2b5b9d377cc78c2442e8e8b198a1

# BioScope (Vincze et al., 2008): "publicly available for research purposes", no direct link.
# Used only by Appendix J.1. Download the corpus from https://rgai.inf.u-szeged.hu/node/105 and
# place abstracts.xml and full_papers.xml in "$EXT/bioscope" (or set EE_BIOSCOPE_DIR).
if [ ! -f "$EXT/bioscope/abstracts.xml" ]; then
    echo "BioScope: download manually from https://rgai.inf.u-szeged.hu/node/105 into $EXT/bioscope/"
fi

echo "Public corpora are in $EXT"
