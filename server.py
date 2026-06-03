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
import hashlib
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
    # 7B–8B run in 8-bit (LLM.int8) — ~near-lossless and fits 16GB with headroom,
    # unlike 4-bit. Only 14B still needs 4-bit (8-bit 14B is ~14GB, too tight).
    {"id": "Qwen/Qwen3-8B-Base", "label": "Qwen3 8B base · ~8.5GB (8-bit)", "quant": "8bit"},
    {"id": "Qwen/Qwen3-14B-Base", "label": "Qwen3 14B base · ~9GB (4-bit)", "quant": "4bit"},
    # Alternative / larger base models. 7B fits 16GB in full bf16 (~14GB) now
    # that we don't materialize fp32 logits; 8B needs 8-bit to leave headroom.
    {"id": "openbmb/MiniCPM5-1B", "label": "MiniCPM5 1B base · bf16"},
    {"id": "Qwen/Qwen2.5-7B", "label": "Qwen2.5 7B base · ~14GB bf16 (full precision)"},
    {"id": "tiiuae/Falcon3-7B-Base", "label": "Falcon3 7B base · ~14GB bf16 (full precision)"},
    {"id": "meta-llama/Llama-3.1-8B", "label": "Llama 3.1 8B base · ~8.5GB (8-bit; gated)", "quant": "8bit"},
    {"id": "google/gemma-4-E4B", "label": "Gemma 4 E4B · experimental (multimodal — may not load)"},
    {"id": "mock", "label": "Mock (no GPU — UI test)"},
]

DEFAULT_MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B")

# Light gate on model switching (changing the model affects everyone). Not meant
# to be strong — just enough to stop casual visitors flipping the model. Set a
# real one via ADMIN_PASSWORD; defaults to "banana".
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "banana")

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

# Cap on tokens scored per request, to bound VRAM (logits are B×S×vocab). Text
# beyond it is truncated; the client shows current/max in the status. Adjustable
# at runtime from the admin panel (clamped to MAX_TOKENS_RANGE).
MAX_TOKENS_DEFAULT = int(os.environ.get("MAX_TOKENS", "8192"))
MAX_TOKENS_RANGE = (16, 32768)


class Runtime:
    """Mutable, admin-adjustable runtime settings (shared by all users)."""

    max_tokens = MAX_TOKENS_DEFAULT

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
                    results = await asyncio.to_thread(predictor.predict_batch, reqs, Runtime.max_tokens)
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


# --- live "conversations": one streaming file per active user, viewable on a
#     separate page (/viewer.html) over the /sessions WebSocket. -----------
SESSIONS_DIR = os.environ.get("SESSIONS_DIR", os.path.join(os.path.dirname(__file__), "sessions"))
sessions: dict[str, dict] = {}        # id -> {"text", "ip", "file", "path"}
viewers: set[WebSocket] = set()       # /sessions viewer sockets
_session_seq = 0


async def _viewer_broadcast(event: dict) -> None:
    payload = json.dumps(event)
    for v in list(viewers):
        try:
            await v.send_text(payload)
        except Exception:
            viewers.discard(v)


def _session_open(ip: str, socket: WebSocket) -> str:
    global _session_seq
    _session_seq += 1
    sid = f"u{_session_seq}"
    try:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        path = os.path.join(SESSIONS_DIR, f"{sid}.txt")
        f = open(path, "w", encoding="utf-8")
    except Exception:
        f, path = None, None
    sessions[sid] = {"text": "", "ip": ip, "file": f, "path": path, "socket": socket}
    return sid


async def _clear_session(sid: str) -> None:
    """Admin delete of one live conversation: wipe its text/file/viewer panel and
    tell that user's editor to clear too (otherwise it repopulates on keystroke)."""
    s = sessions.get(sid)
    if s is None:
        return
    _session_write(sid, "")
    await _viewer_broadcast({"type": "update", "id": sid, "text": ""})
    sock = s.get("socket")
    if sock is not None:
        try:
            await sock.send_text(json.dumps({"cleared": True}))
        except Exception:
            pass


def _session_write(sid: str, text: str) -> None:
    s = sessions.get(sid)
    if s is None:
        return
    s["text"] = text
    f = s.get("file")
    if f is not None:
        try:
            f.seek(0)
            f.truncate()
            f.write(text)
            f.flush()
        except Exception:
            pass


def _session_close(sid: str) -> None:
    s = sessions.pop(sid, None)
    if s is None:
        return
    f = s.get("file")
    if f is not None:
        try:
            f.close()
        except Exception:
            pass
    if s.get("path"):
        try:
            os.remove(s["path"])
        except OSError:
            pass


@app.get("/api/config")
def config():
    return {
        "models": MODELS,
        "default": DEFAULT_MODEL,
        "current": manager.desired_name,
        "max_tokens": Runtime.max_tokens,
    }


# Version stamps so the page can show whether it's running the latest code.
# Backend files need a server restart; static files only need a page reload.
_HERE = os.path.dirname(__file__)
_PY_FILES = ["server.py", "predictors.py"]
_STATIC_FILES = ["static/index.html", "static/app.js", "static/style.css", "static/viewer.html"]


def _hash_files(rel_files) -> str:
    h = hashlib.sha1()
    for rel in rel_files:
        try:
            with open(os.path.join(_HERE, rel), "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(b"<missing>")
        h.update(b"\0")
    return h.hexdigest()[:12]


# Captured once at startup = the code this process is actually running.
RUNNING_PY = _hash_files(_PY_FILES)
RUNNING_STATIC = _hash_files(_STATIC_FILES)


@app.get("/api/version")
def version():
    """running_* = what this process started with; disk_* = what's on disk now.
    If disk_py != running_py a server RESTART is needed; if disk_static differs
    from what the page loaded, a browser RELOAD picks it up."""
    return {
        "running_py": RUNNING_PY,
        "running_static": RUNNING_STATIC,
        "disk_py": _hash_files(_PY_FILES),
        "disk_static": _hash_files(_STATIC_FILES),
    }


@app.websocket("/sessions")
async def sessions_ws(socket: WebSocket):
    """Viewer feed: snapshot of every active user's current text, then live
    open/update/close events as people type and come/go."""
    await socket.accept()
    viewers.add(socket)
    await socket.send_text(json.dumps({
        "type": "snapshot",
        "sessions": [{"id": sid, "ip": s["ip"], "text": s["text"]} for sid, s in sessions.items()],
    }))
    try:
        while True:
            raw = await socket.receive_text()
            try:
                m = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # Verify the admin password so the viewer page can unlock delete.
            if "admin_check" in m:
                await socket.send_text(json.dumps({"admin_ok": m.get("admin_check") == ADMIN_PASSWORD}))
            # Admin: delete (clear) one conversation.
            elif m.get("action") == "delete":
                if m.get("password") != ADMIN_PASSWORD:
                    await socket.send_text(json.dumps({"error": "wrong admin password"}))
                else:
                    await _clear_session(m.get("id"))
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        viewers.discard(socket)


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    engine.start()  # idempotent; ensures the worker runs on the live event loop
    clients.add(socket)
    ip = socket.client.host if socket.client else "?"
    sid = _session_open(ip, socket)  # one streaming file per active user
    await _viewer_broadcast({"type": "open", "id": sid, "ip": ip})
    await socket.send_text(
        json.dumps({
            "models": MODELS, "default": DEFAULT_MODEL,
            "current": manager.desired_name, "max_tokens": Runtime.max_tokens,
        })
    )

    try:
        while True:
            raw = await socket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # --- admin: verify the password so the client can unlock the panel ---
            if "admin_check" in msg:
                await socket.send_text(
                    json.dumps({"admin_ok": msg.get("admin_check") == ADMIN_PASSWORD})
                )
                continue

            # --- admin: adjust the per-request token cap (affects EVERYONE) ---
            if "set_max_tokens" in msg:
                if msg.get("password") != ADMIN_PASSWORD:
                    await socket.send_text(
                        json.dumps({"error": "Wrong admin password — unlock the admin panel to change the token cap."})
                    )
                    continue
                try:
                    n = int(msg.get("set_max_tokens"))
                except (TypeError, ValueError):
                    await socket.send_text(json.dumps({"error": "max tokens must be a number"}))
                    continue
                lo, hi = MAX_TOKENS_RANGE
                Runtime.max_tokens = max(lo, min(hi, n))
                await broadcast({"max_tokens_changed": Runtime.max_tokens})
                continue

            # --- explicit global model switch: admin-only, changes EVERYONE ---
            if "set_model" in msg:
                if msg.get("password") != ADMIN_PASSWORD:
                    await socket.send_text(
                        json.dumps({"error": "Wrong admin password — unlock the admin panel to switch models."})
                    )
                    continue
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

            # Stream this user's current text to their session file + viewers.
            _session_write(sid, text)
            await _viewer_broadcast({"type": "update", "id": sid, "text": text})

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
        _session_close(sid)  # close + remove this user's streaming file
        await _viewer_broadcast({"type": "close", "id": sid})


def _gpu_frame() -> str:
    """One nvtop-style snapshot via gpustat (nvml). Real nvtop is a TUI and
    can't be piped to a browser, so we render per-GPU util/mem/temp + a bar +
    the compute processes ourselves."""
    try:
        import gpustat

        stats = gpustat.GPUStatCollection.new_query()
        lines = []
        for g in stats.gpus:
            used, total = int(g.memory_used), int(g.memory_total)
            frac = used / total if total else 0.0
            fill = int(frac * 28)
            bar = "█" * fill + "░" * (28 - fill)
            util = g.utilization if g.utilization is not None else "?"
            temp = g.temperature if g.temperature is not None else "?"
            lines.append(f"GPU {g.index}  {g.entry.get('name', '')}")
            lines.append(f"  util {str(util):>3}%   temp {str(temp):>3}°C   mem {used:>6} / {total} MiB")
            lines.append(f"  [{bar}] {frac * 100:4.1f}%")
            procs = g.processes or []
            if procs:
                parts = [
                    f"{p.get('pid')}:{p.get('gpu_memory_usage', '?')}MiB"
                    + (f"({p['command']})" if p.get("command") else "")
                    for p in procs[:6]
                ]
                lines.append("  procs  " + "  ".join(parts))
        return "\n".join(lines) or "(no GPU data)"
    except Exception as exc:
        return f"gpu monitor unavailable: {type(exc).__name__}: {exc}"


@app.websocket("/gpu")
async def gpu(socket: WebSocket):
    """Push a fresh GPU snapshot ~once a second; the client replaces the panel
    each frame (a live gauge rather than a scrolling log)."""
    await socket.accept()
    try:
        while True:
            await socket.send_text(await asyncio.to_thread(_gpu_frame))
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, RuntimeError):
        return


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
