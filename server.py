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

# Curated base (pretrained, non-instruct) models for next-token prediction on a
# single 16GB GPU (e.g. 1xA4000). Models up to ~4B run in FULL bf16 precision
# (no quantization — quantization measurably degrades next-token calibration).
# The 8B/14B entries carry a "quant" hint because they only fit 16GB in 4-bit;
# everything else loads bf16. Qwen3-4B-Base is the recommended default: strong
# yet comfortably full-precision at ~8GB.
MODELS = [
    {"id": "gpt2", "label": "GPT-2 small · 124M (baseline)"},
    {"id": "Qwen/Qwen3-0.6B-Base", "label": "Qwen3 0.6B base · ~1.4GB bf16"},
    {"id": "Qwen/Qwen3-1.7B-Base", "label": "Qwen3 1.7B base · ~3.8GB bf16"},
    {"id": "Qwen/Qwen3-4B-Base", "label": "Qwen3 4B base · ~8GB bf16 (recommended)"},
    {"id": "Qwen/Qwen3-8B-Base", "label": "Qwen3 8B base · ~5.5GB (4-bit — quantized)", "quant": "4bit"},
    {"id": "Qwen/Qwen3-14B-Base", "label": "Qwen3 14B base · ~9GB (4-bit — quantized)", "quant": "4bit"},
    {"id": "mock", "label": "Mock (no GPU — UI test)"},
]

DEFAULT_MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Base")

# Make sure a custom MODEL_NAME is always selectable in the dropdown.
if DEFAULT_MODEL not in {m["id"] for m in MODELS}:
    MODELS.insert(0, {"id": DEFAULT_MODEL, "label": f"{DEFAULT_MODEL} (custom)"})

ALLOWED = {m["id"] for m in MODELS}
# Per-model quantization hint ("4bit"/"8bit"/None). The QUANTIZE env var, if set
# non-empty, overrides this for every model.
MODEL_QUANT = {m["id"]: m.get("quant") for m in MODELS}


def resolve_quant(name: str) -> str | None:
    env_q = os.environ.get("QUANTIZE", "").strip()
    return env_q if env_q else MODEL_QUANT.get(name)


# Dynamic-batching knobs (overridable via env).
MAX_BATCH = int(os.environ.get("MAX_BATCH", "8"))
BATCH_WINDOW = float(os.environ.get("BATCH_WINDOW_MS", "8")) / 1000.0

# File the live in-page log terminal tails. run.sh tees the server's stdout+
# stderr here (so it captures model-download/load progress too).
LOG_FILE = os.environ.get("APP_LOG", os.path.join(os.path.dirname(__file__), "app.log"))


class ModelManager:
    """Holds at most ONE loaded model, shared by every connected user. Only one
    model lives in VRAM at a time; switching evicts the current one first (so a
    16GB GPU never holds two), and because everyone shares it, their requests
    can be batched into a single forward pass. Loading runs in a worker thread
    and is serialized by an async lock."""

    def __init__(self) -> None:
        self.predictor = None
        self.current_name: str | None = None
        self.desired_name: str = DEFAULT_MODEL  # the global selection
        self.lock = asyncio.Lock()

    async def _ensure_locked(self):
        """Load `desired_name` if it isn't the one already resident. Caller MUST
        hold self.lock (the batch engine holds it across inference so the model
        can't be swapped out mid-forward)."""
        if self.predictor is not None and self.current_name == self.desired_name:
            return self.predictor
        if self.predictor is not None:
            old, self.predictor, self.current_name = self.predictor, None, None
            await asyncio.to_thread(free_predictor, old)
        name = self.desired_name
        predictor = await asyncio.to_thread(build_predictor, name, resolve_quant(name))
        self.predictor, self.current_name = predictor, name
        return predictor

    async def switch(self, name: str):
        """Make `name` the global model for everyone."""
        async with self.lock:
            self.desired_name = name
            return await self._ensure_locked()


class BatchEngine:
    """Coalesces concurrent predict requests into one padded forward pass.

    Each request is parked on a future; a single worker drains the queue,
    waits a short window for stragglers, then runs up to MAX_BATCH of them
    through `predict_batch` while holding the model lock (so a switch can't
    free the weights mid-flight). Batches run one at a time — there is only
    one GPU — but new requests accumulate while the current batch computes."""

    def __init__(self, manager: ModelManager) -> None:
        self.manager = manager
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def submit(self, text: str, k: int):
        fut = asyncio.get_event_loop().create_future()
        await self.queue.put((text, k, fut))
        return await fut

    async def _run(self) -> None:
        while True:
            batch = [await self.queue.get()]
            if BATCH_WINDOW > 0:
                await asyncio.sleep(BATCH_WINDOW)  # let concurrent users coalesce
            while len(batch) < MAX_BATCH and not self.queue.empty():
                batch.append(self.queue.get_nowait())

            reqs = [(t, k) for (t, k, _f) in batch]
            try:
                async with self.manager.lock:
                    predictor = await self.manager._ensure_locked()
                    results = await asyncio.to_thread(predictor.predict_batch, reqs)
                for (_t, _k, fut), res in zip(batch, results):
                    if not fut.done():
                        fut.set_result(res)
            except Exception as exc:  # fail the whole batch, clients see the error
                for (_t, _k, fut) in batch:
                    if not fut.done():
                        fut.set_exception(exc)


manager = ModelManager()
engine = BatchEngine(manager)
clients: set[WebSocket] = set()


async def broadcast(msg: dict) -> None:
    payload = json.dumps(msg)
    for ws in list(clients):
        try:
            await ws.send_text(payload)
        except Exception:
            clients.discard(ws)


@app.get("/api/config")
def config():
    return {"models": MODELS, "default": DEFAULT_MODEL, "current": manager.desired_name}


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    engine.start()  # idempotent; ensures the worker runs on the live event loop
    clients.add(socket)
    await socket.send_text(
        json.dumps({"models": MODELS, "default": DEFAULT_MODEL, "current": manager.desired_name})
    )

    try:
        while True:
            raw = await socket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # --- explicit global model switch: changes the model for EVERYONE ---
            if "set_model" in msg:
                name = msg.get("set_model")
                if name not in ALLOWED:
                    await socket.send_text(json.dumps({"error": f"Unknown model '{name}'"}))
                    continue
                if name != manager.desired_name:
                    manager.desired_name = name  # reflect immediately for new requests
                    await broadcast({"status": "loading", "model": name})
                    try:
                        await manager.switch(name)
                    except Exception as exc:
                        await broadcast({"error": f"{type(exc).__name__}: {exc}"})
                        continue
                    await broadcast({"model_switched": name})  # sync everyone's dropdown
                continue

            # --- prediction on the CURRENT global model (batched) ---
            text = msg.get("text", "")
            k = int(msg.get("k", 10))
            seq = msg.get("seq")  # echoed back so the client can drop stale results

            # If the model isn't resident yet, tell this client before it blocks.
            if manager.predictor is None or manager.current_name != manager.desired_name:
                await socket.send_text(
                    json.dumps({"status": "loading", "model": manager.desired_name, "seq": seq})
                )

            try:
                result = await engine.submit(text, k)
            except Exception as exc:
                await socket.send_text(
                    json.dumps({"error": f"{type(exc).__name__}: {exc}", "seq": seq})
                )
                continue

            result["seq"] = seq
            await socket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        return
    finally:
        clients.discard(socket)


@app.websocket("/logs")
async def logs(socket: WebSocket):
    """Stream the server log file to the in-page terminal: send a tail to start,
    then follow appended bytes (and re-open if the file is truncated/rotated)."""
    await socket.accept()
    TAIL = 16384
    f = None
    try:
        while True:
            if f is None:
                if not os.path.exists(LOG_FILE):
                    await asyncio.sleep(0.5)
                    continue
                f = open(LOG_FILE, "rb")
                size = f.seek(0, 2)
                f.seek(max(0, size - TAIL))
                if size > TAIL:
                    f.readline()  # drop a partial first line
            chunk = f.read()
            if chunk:
                await socket.send_text(chunk.decode("utf-8", "replace"))
            else:
                # Detect truncation/rotation (file got smaller) and re-open.
                try:
                    if os.path.getsize(LOG_FILE) < f.tell():
                        f.close()
                        f = None
                        continue
                except OSError:
                    f = None
                await asyncio.sleep(0.25)
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        if f is not None:
            f.close()


# Static frontend. Mounted last so /ws and /api/* take precedence.
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
else:  # pragma: no cover
    @app.get("/")
    def _missing():
        return JSONResponse({"error": "static/ directory missing"}, status_code=500)
