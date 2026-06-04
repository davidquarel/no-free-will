#!/usr/bin/env bash
#
# One-time setup: install uv and create a local .venv with all dependencies
# (torch, transformers, bitsandbytes, …). Run this once after cloning; then
# use run.sh to start the server.
#
#   bash setup.sh
#
set -euo pipefail
cd "$(dirname "$0")"

# Build an ISOLATED venv for this project (so its newer transformers, needed for
# Gemma 4, never disturbs a shared conda env like arena-env). We deliberately use
# plain `python -m venv` + pip rather than `uv run`, because uv prefers an active
# conda env (CONDA_PREFIX) over .venv and would run against the wrong interpreter.
#
# PY picks the interpreter to base the venv on. On a box that already has a CUDA
# torch in a conda env you can reuse it (and skip a multi-GB torch download) with:
#   PY=/opt/conda/envs/arena-env/bin/python3 SYSTEM_SITE_PACKAGES=1 bash setup.sh
PY="${PY:-python3}"
VENV_ARGS=""
[ "${SYSTEM_SITE_PACKAGES:-0}" = "1" ] && VENV_ARGS="--system-site-packages"

echo ">> creating .venv (base: $PY ${VENV_ARGS:-isolated}) …"
rm -rf .venv
"$PY" -m venv $VENV_ARGS .venv

echo ">> installing dependencies (this can take a few minutes)…"
# --ignore-installed so the venv gets its OWN copy of each dep even when it's
# inheriting an older one from a system/conda env via --system-site-packages.
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install --ignore-installed -r requirements.txt

echo ">> setup complete. Start the server with:  bash run.sh"
