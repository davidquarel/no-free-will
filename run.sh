#!/usr/bin/env bash
#
# Serve the app. Run setup.sh once first to install dependencies.
# Defaults to Gemma 4 12B base, loaded 8-bit (~13GB, near-lossless) so it fits a
# 16GB card. Other models auto-quantize per their hint (bf16 ≤4B, 8-bit 7–8B);
# switch models from the in-page admin panel (admin password comes from the
# ADMIN_PASSWORD env var — see below). Runs via this project's own .venv.
#
#   bash run.sh                 # serve on port 8080
#   bash run.sh --port 9000     # serve on a different port
#
# Override anything via env vars, e.g.:
#   MODEL_NAME=Qwen/Qwen3-8B-Base bash run.sh   # start on a bigger model
#   QUANTIZE=4bit bash run.sh                    # force 4-bit for EVERY model
#   TUNNEL=1 bash run.sh                         # also open a public cloudflared URL
#   ADMIN_PASSWORD='secret' bash run.sh          # admin password (else one is
#                                                # generated into admin_password.txt)
#
set -euo pipefail
cd "$(dirname "$0")"

MODEL_NAME="${MODEL_NAME:-google/gemma-4-12B}"
QUANTIZE="${QUANTIZE-}"        # empty = per-model default (bf16 ≤4B, 4-bit for 8B/14B)
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
LOG_FILE="${APP_LOG:-app.log}"

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

# Make sure uv is on PATH (setup.sh may have installed it this shell ago).
# shellcheck disable=SC1091
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

if [ ! -d .venv ]; then
  echo "No .venv found — run setup first:  bash setup.sh" >&2
  exit 1
fi

# Use THIS project's own venv (isolated from any conda env, e.g. arena-env).
# It carries the newer transformers (>=5.10, for Gemma 4) without touching other
# environments. We call its python directly rather than `uv run`, because uv
# would otherwise prefer an active conda env (CONDA_PREFIX) over .venv.
SERVE=(env MODEL_NAME="$MODEL_NAME" QUANTIZE="$QUANTIZE" APP_LOG="$LOG_FILE"
       ./.venv/bin/python -m uvicorn server:app --host "$HOST" --port "$PORT")

echo ">> serving ${MODEL_NAME} (QUANTIZE='${QUANTIZE:-per-model}') on ${HOST}:${PORT}"
echo ">> logs stream to the terminal AND ${LOG_FILE} (the in-page terminal tails it)"

# Without a tunnel, just serve in the foreground. tee so the in-page log
# terminal can tail app.log while you still see everything here.
if [ "${TUNNEL:-0}" != "1" ]; then
  echo ">> open http://localhost:${PORT}  (or use an SSH tunnel / TUNNEL=1 for a public URL)"
  "${SERVE[@]}" 2>&1 | tee "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi

# --- TUNNEL path: start the server, wait until it answers, THEN open the
#     tunnel so the public URL never points at a not-yet-listening port. ---
"${SERVE[@]}" > >(tee "$LOG_FILE") 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; kill "${CF_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo ">> waiting for the server to come up on port ${PORT}…"
for _ in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}/api/config" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "!! server exited before it started listening — check the logs above" >&2
    exit 1
  fi
  sleep 1
done

if [ ! -x ./cloudflared ]; then
  echo ">> fetching cloudflared…"
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
  chmod +x cloudflared
fi

echo ">> opening public tunnel…"
./cloudflared tunnel --url "http://localhost:${PORT}" > cloudflared.log 2>&1 &
CF_PID=$!

URL=""
for _ in $(seq 1 30); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' cloudflared.log | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done

echo
echo "============================================================"
if [ -n "$URL" ]; then
  echo "  PUBLIC URL:  $URL"
else
  echo "  Tunnel URL not detected yet — check cloudflared.log"
fi
echo "  (the model downloads on first use; first prediction lags)"
echo "============================================================"
echo

wait "$SERVER_PID"
