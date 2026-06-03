"""FastAPI server: live next-token prediction over a WebSocket.

Run:
    pip install -r requirements.txt
    MODEL_NAME=gpt2 uvicorn server:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000  (or point a browser at the host's IP).

The page has a model dropdown (see MODELS below). Only one model is held in
memory at a time: switching evicts the previous model first, so a 16GB GPU
never has to hold two large models at once. Weights load in bf16/fp16 on CUDA.

Set MODEL_NAME=mock to run without torch/transformers.
"""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from predictors import build_predictor, free_predictor

app = FastAPI(title="no-free-will")

# Curated base (pretrained, non-instruct) models for next-token prediction.
# Sizes are bf16 weight footprints. On a multi-GPU box these shard across all
# cards automatically (device_map="auto"), so the larger ones fit a 4xA4000
# (~64GB total). Qwen3-14B-Base is the recommended "strong but reliable" pick:
# Qwen reports it matches Qwen2.5-32B-Base while leaving plenty of headroom.
MODELS = [
    {"id": "gpt2", "label": "GPT-2 small · 124M (baseline)"},
    {"id": "Qwen/Qwen3-0.6B-Base", "label": "Qwen3 0.6B base · ~1.4GB"},
    {"id": "Qwen/Qwen3-1.7B-Base", "label": "Qwen3 1.7B base · ~3.8GB"},
    {"id": "Qwen/Qwen3-4B-Base", "label": "Qwen3 4B base · ~8GB"},
    {"id": "Qwen/Qwen3-8B-Base", "label": "Qwen3 8B base · ~16GB"},
    {"id": "Qwen/Qwen3-14B-Base", "label": "Qwen3 14B base · ~28GB (recommended)"},
    {"id": "Qwen/Qwen3-30B-A3B-Base", "label": "Qwen3 30B-A3B MoE base · ~61GB (tight)"},
    {"id": "HuggingFaceTB/SmolLM2-1.7B", "label": "SmolLM2 1.7B base · ~3.8GB"},
    {"id": "mock", "label": "Mock (no GPU — UI test)"},
]

DEFAULT_MODEL = os.environ.get("MODEL_NAME", "gpt2")

# Make sure a custom MODEL_NAME is always selectable in the dropdown.
if DEFAULT_MODEL not in {m["id"] for m in MODELS}:
    MODELS.insert(0, {"id": DEFAULT_MODEL, "label": f"{DEFAULT_MODEL} (custom)"})

ALLOWED = {m["id"] for m in MODELS}


class ModelManager:
    """Holds at most one loaded model. Loading runs in a worker thread (so the
    event loop keeps serving other sockets) and is serialized by an async lock.
    Switching models evicts the current one first to bound VRAM usage."""

    def __init__(self) -> None:
        self.predictor = None
        self.current_name: str | None = None
        self.lock = asyncio.Lock()

    async def get(self, name: str):
        async with self.lock:
            if self.current_name == name and self.predictor is not None:
                return self.predictor
            if self.predictor is not None:
                old, self.predictor, self.current_name = self.predictor, None, None
                await asyncio.to_thread(free_predictor, old)
            predictor = await asyncio.to_thread(build_predictor, name)
            self.predictor, self.current_name = predictor, name
            return predictor


manager = ModelManager()


@app.get("/api/config")
def config():
    return {"models": MODELS, "default": DEFAULT_MODEL}


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    await socket.send_text(
        json.dumps({"models": MODELS, "default": DEFAULT_MODEL})
    )

    try:
        while True:
            raw = await socket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            text = msg.get("text", "")
            k = int(msg.get("k", 10))
            seq = msg.get("seq")  # echoed back so the client can drop stale results
            requested = msg.get("model") or DEFAULT_MODEL

            if requested not in ALLOWED:
                await socket.send_text(
                    json.dumps({"error": f"Unknown model '{requested}'", "seq": seq})
                )
                continue

            # Tell the client we're (down)loading before the blocking work.
            if manager.current_name != requested:
                await socket.send_text(
                    json.dumps({"status": "loading", "model": requested, "seq": seq})
                )

            try:
                predictor = await manager.get(requested)
                result = await asyncio.to_thread(predictor.predict, text, k)
            except Exception as exc:
                await socket.send_text(
                    json.dumps({"error": f"{type(exc).__name__}: {exc}", "seq": seq})
                )
                continue

            result["seq"] = seq
            await socket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        return


# Static frontend. Mounted last so /ws and /api/* take precedence.
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
else:  # pragma: no cover
    @app.get("/")
    def _missing():
        return JSONResponse({"error": "static/ directory missing"}, status_code=500)
