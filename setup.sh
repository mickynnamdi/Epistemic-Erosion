#!/usr/bin/env bash
# Create a virtual environment with every dependency of the package.
#   bash setup.sh            # requirements.txt (compatible versions)
#   bash setup.sh --locked   # requirements-lock.txt (the exact versions the release was tested with)
set -euo pipefail

REQ=requirements.txt
[ "${1:-}" = "--locked" ] && REQ=requirements-lock.txt

if [ ! -d venv ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r "$REQ"

cat <<'EOF'

Setup complete. Next:
    source venv/bin/activate
    cp configs/paths.example.env .env      # then edit .env (optional; see docs/DATA.md)
    python -m pytest tests/                # 12 tests run without data; the rest skip until data exist
    See docs/REPRODUCE.md for the full pipeline.
EOF
