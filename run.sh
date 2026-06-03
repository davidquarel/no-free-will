#!/usr/bin/env bash
#
# One-shot: install deps into a local .venv (with uv) and serve the app.
# Defaults to Qwen3-14B-Base in 4-bit, which fits a single 16GB GPU (~9GB)
# while keeping ~Qwen2.5-32B-Base quality.
#
# Usage (after cloning the repo):
#   bash run.sh                 # serve on port 8080
#   bash run.sh --port 9000     # serve on a different port
#
# Override anything via env vars, e.g.:
#   MODEL_NAME=Qwen/Qwen3-8B-Base QUANTIZE=4bit bash run.sh
#   QUANTIZE= MODEL_NAME=Qwen/Qwen3-4B-Base bash run.sh   # no quantization (fits bf16)
#   TUNNEL=1 bash run.sh                                   # also open a public cloudflared URL
#
set -euo pipefail
cd "$(dirname "$0")"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B-Base}"
QUANTIZE="${QUANTIZE-4bit}"     # set QUANTIZE= (empty) to disable
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

# CLI args (override the env defaults above).
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --host=*) HOST="${1#*=}"; shift ;;
    -h|--help)
      echo "Usage: bash run.sh [--port PORT] [--host HOST]"
      echo "Env: MODEL_NAME, QUANTIZE (4bit|8bit|empty), TUNNEL=1"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# 1. uv (fast installer + venv manager)
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

# 2. venv + dependencies
echo ">> creating .venv and installing dependencies (torch, transformers, bitsandbytes…)"
uv venv .venv
uv pip install -r requirements.txt

# 3. optional public URL via cloudflared
if [ "${TUNNEL:-0}" = "1" ]; then
  if [ ! -x ./cloudflared ]; then
    echo ">> fetching cloudflared…"
    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
    chmod +x cloudflared
  fi
  echo ">> opening public tunnel (URL will print below)…"
  ./cloudflared tunnel --url "http://localhost:${PORT}" &
fi

# 4. serve. First run downloads the model weights from HuggingFace.
echo ">> serving ${MODEL_NAME} (QUANTIZE='${QUANTIZE}') on ${HOST}:${PORT}"
echo ">> open http://localhost:${PORT}  (or use an SSH tunnel / TUNNEL=1 for a public URL)"
exec env MODEL_NAME="$MODEL_NAME" QUANTIZE="$QUANTIZE" \
  uv run uvicorn server:app --host "$HOST" --port "$PORT"
