#!/usr/bin/env bash
# One-shot bootstrap for a fresh clone: create a virtual environment, install
# the CPU verification stack, and run the smoke test + artifact verification.
# GPU users: afterwards run `make setup-llm` inside the same environment.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
VENV=${VENV:-.venv}

if [ ! -d "$VENV" ]; then
    "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

make smoke
make verify
echo "Bootstrap complete. Next: 'make reproduce-paper' (CPU) or see docs/EXPERIMENTS.md for GPU re-execution."
