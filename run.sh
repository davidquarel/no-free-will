#!/usr/bin/env bash
#
# Serve the app. Run setup.sh once first to install dependencies.
# Defaults to Qwen3-14B-Base in 4-bit, which fits a single 16GB GPU (~9GB)
# while keeping ~Qwen2.5-32B-Base quality.
#
#   bash run.sh                 # serve on port 8080
#   bash run.sh --port 9000     # serve on a different port
#
# Override anything via env vars, e.g.:
#   MODEL_NAME=Qwen/Qwen3-8B-Base bash run.sh
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

# Make sure uv is on PATH (setup.sh may have installed it this shell ago).
# shellcheck disable=SC1091
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

if [ ! -d .venv ]; then
  echo "No .venv found — run setup first:  bash setup.sh" >&2
  exit 1
fi

SERVE=(env MODEL_NAME="$MODEL_NAME" QUANTIZE="$QUANTIZE"
       uv run uvicorn server:app --host "$HOST" --port "$PORT")

echo ">> serving ${MODEL_NAME} (QUANTIZE='${QUANTIZE}') on ${HOST}:${PORT}"

# Without a tunnel, just serve in the foreground.
if [ "${TUNNEL:-0}" != "1" ]; then
  echo ">> open http://localhost:${PORT}  (or use an SSH tunnel / TUNNEL=1 for a public URL)"
  exec "${SERVE[@]}"
fi

# --- TUNNEL path: start the server, wait until it answers, THEN open the
#     tunnel so the public URL never points at a not-yet-listening port. ---
"${SERVE[@]}" &
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
