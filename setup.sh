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

# 1. uv (fast installer + venv manager)
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

# 2. venv + dependencies
echo ">> creating .venv and installing dependencies (this can take a few minutes)…"
uv venv .venv
uv pip install -r requirements.txt

echo ">> setup complete. Start the server with:  bash run.sh"
