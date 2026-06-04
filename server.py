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
import secrets
import sys
import urllib.request

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from predictors import build_predictor, free_predictor

app = FastAPI(title="no-free-will")


@app.middleware("http")
async def _no_cache_assets(request, call_next):
    """Stop the frontend (HTML/JS/CSS) from being cached — by the browser AND by
    a Cloudflare tunnel, which otherwise caches .js/.css by extension and serves
    a stale app.js after every deploy. `no-cache` allows efficient 304s via the
    ETag StaticFiles already sends, while guaranteeing the latest code is used."""
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp

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
    # Gemma 4 12B base (the default). It's a multimodal "unified" checkpoint
    # (Gemma4UnifiedForConditionalGeneration), but transformers maps it under
    # AutoModelForCausalLM, so the text-only causal-LM path works unchanged.
    # 24GB in bf16; we load 8-bit (LLM.int8, near-lossless) at ~13GB, which fits a
    # 16GB card with headroom and keeps next-token calibration far better than
    # 4-bit. Needs transformers>=5.10 (only in this project's .venv, not arena).
    {"id": "google/gemma-4-12B", "label": "Gemma 4 12B base · ~13GB (8-bit; multimodal)", "quant": "8bit"},
    {"id": "mock", "label": "Mock (no GPU — UI test)"},
]

DEFAULT_MODEL = os.environ.get("MODEL_NAME", "google/gemma-4-12B")

# Admin password — taken ONLY from the ADMIN_PASSWORD env var, so it's never
# hardcoded in source. If it's unset, we generate a random one and write it to
# admin_password.txt (gitignored). We deliberately do NOT print the value: the
# server log is streamed publicly to the in-page terminal.
#   Set your own:  ADMIN_PASSWORD='whatever' bash run.sh
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = secrets.token_urlsafe(9)
    try:
        with open(os.path.join(os.path.dirname(__file__), "admin_password.txt"), "w", encoding="utf-8") as _pf:
            _pf.write(ADMIN_PASSWORD + "\n")
    except OSError:
        pass
    print("[admin] ADMIN_PASSWORD not set — wrote a generated one to admin_password.txt", flush=True)

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
MAX_TOKENS_DEFAULT = int(os.environ.get("MAX_TOKENS", "1024"))
MAX_TOKENS_RANGE = (16, 32768)

# Whether the live conversation viewer (/viewer.html via the /sessions feed) is
# on by default. Admins can flip this at runtime from the in-page admin panel.
VIEWER_ENABLED_DEFAULT = os.environ.get("VIEWER_ENABLED", "1").strip().lower() not in (
    "0", "false", "no", "off",
)


class Runtime:
    """Mutable, admin-adjustable runtime settings (shared by all users)."""

    max_tokens = MAX_TOKENS_DEFAULT
    viewer_enabled = VIEWER_ENABLED_DEFAULT

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
admin_viewers: set[WebSocket] = set()  # viewers that unlocked (get the ban list)
_session_seq = 0

# Identifies this server process; sent in the hello so the client can tell when
# the server has restarted (and clear its editor instead of resurrecting text).
SERVER_ID = os.getpid()


def _reboot_server() -> None:
    """Kill-and-restart this server by re-execing the process in place.

    os.execv replaces the current image with a fresh `python -m uvicorn …` (same
    PID), so all code AND the model reload — a genuine restart that even picks up
    edited .py files. The listening socket is close-on-exec, so the port frees and
    the new uvicorn rebinds; stdout/stderr (and thus run.sh's tee to app.log) are
    inherited, so logs keep flowing. This never returns."""
    os.execv(sys.executable, [sys.executable, "-m", "uvicorn", *sys.argv[1:]])

# Bans (admin, from the viewer page). We ban a per-browser client token rather
# than an IP: it targets the individual browser instead of everyone sharing the
# IP (NAT/household), and is intentionally evadable (clear storage/incognito) —
# annoying, not nuclear. Persisted so bans survive restarts. (A web server can't
# see a client's MAC address — that never leaves the local network.)
BANNED_FILE = os.environ.get("BANNED_FILE", os.path.join(os.path.dirname(__file__), "banned_clients.txt"))


def _load_bans() -> set[str]:
    try:
        with open(BANNED_FILE, encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    except OSError:
        return set()


BANNED_CIDS: set[str] = _load_bans()


def _save_bans() -> None:
    try:
        with open(BANNED_FILE, "w", encoding="utf-8") as f:
            for c in sorted(BANNED_CIDS):
                f.write(c + "\n")
    except OSError:
        pass


def _ban_cid(cid: str | None) -> None:
    if not cid or cid in BANNED_CIDS:
        return
    BANNED_CIDS.add(cid)
    _save_bans()


def _unban_cid(cid: str | None) -> None:
    if cid and cid in BANNED_CIDS:
        BANNED_CIDS.discard(cid)
        _save_bans()


def _reset_sessions_dir() -> None:
    """Wipe leftover conversation files from a previous run on startup, so old
    text can't survive a server restart."""
    try:
        for fn in os.listdir(SESSIONS_DIR):
            if fn.endswith(".txt"):
                try:
                    os.remove(os.path.join(SESSIONS_DIR, fn))
                except OSError:
                    pass
    except OSError:
        pass


_reset_sessions_dir()


async def _viewer_broadcast(event: dict) -> None:
    # When an admin has turned the live viewer off, stop streaming conversation
    # data to the feed entirely (open/update/close/score/flag all flow through
    # here). Control messages to viewers go through their own helpers below.
    if not Runtime.viewer_enabled:
        return
    payload = json.dumps(event)
    for v in list(viewers):
        try:
            await v.send_text(payload)
        except Exception:
            viewers.discard(v)


def _sessions_snapshot() -> dict:
    """Current state of every active conversation, for a (re)connecting viewer."""
    return {
        "type": "snapshot",
        "sessions": [
            {"id": sid, "flag": s.get("flag", "🌐"), "text": s["text"], "score": s.get("score")}
            for sid, s in sessions.items()
        ],
    }


async def _broadcast_viewer_state() -> None:
    """Tell every connected viewer whether the live feed is on. On enable, follow
    immediately with a fresh snapshot so panels repopulate; on disable, the page
    clears itself and shows an 'off' notice."""
    state = json.dumps({"type": "viewer_state", "enabled": Runtime.viewer_enabled})
    snap = json.dumps(_sessions_snapshot()) if Runtime.viewer_enabled else None
    for v in list(viewers):
        try:
            await v.send_text(state)
            if snap is not None:
                await v.send_text(snap)
        except Exception:
            viewers.discard(v)


async def _broadcast_bans() -> None:
    """Push the ban list to unlocked admin viewers only (not regular viewers)."""
    payload = json.dumps({"type": "bans", "bans": sorted(BANNED_CIDS)})
    for v in list(admin_viewers):
        try:
            await v.send_text(payload)
        except Exception:
            admin_viewers.discard(v)


# Coarse IP -> country flag, so the viewer shows "where", not a precise IP.
_geo_cache: dict[str, str | None] = {}


def _client_ip(socket: WebSocket) -> str:
    """Real client IP — behind the cloudflare tunnel the socket sees localhost,
    so prefer the forwarded headers."""
    h = socket.headers
    fwd = h.get("cf-connecting-ip") or h.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return socket.client.host if socket.client else "?"


def _geo_country(ip: str) -> str | None:
    """ISO country code for an IP (cached). None for private/loopback/unknown."""
    if ip in _geo_cache:
        return _geo_cache[ip]
    cc = None
    try:
        if ip not in ("?", "127.0.0.1", "::1") and not ip.startswith(("10.", "192.168.", "172.")):
            url = f"http://ip-api.com/json/{ip}?fields=status,countryCode"
            with urllib.request.urlopen(url, timeout=3) as r:
                d = json.loads(r.read().decode())
            if d.get("status") == "success":
                cc = d.get("countryCode")
    except Exception:
        cc = None
    _geo_cache[ip] = cc
    return cc


def _flag(country_code: str | None) -> str:
    """Country code -> flag emoji (regional indicator letters). 🌐 if unknown."""
    if not country_code or len(country_code) != 2 or not country_code.isalpha():
        return "🌐"
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in country_code.upper())


async def _geolocate_session(sid: str, ip: str) -> None:
    cc = await asyncio.to_thread(_geo_country, ip)
    s = sessions.get(sid)
    if s is None:
        return
    s["flag"] = _flag(cc)
    await _viewer_broadcast({"type": "flag", "id": sid, "flag": s["flag"]})


def _session_open(ip: str, cid: str, socket: WebSocket) -> str:
    global _session_seq
    _session_seq += 1
    sid = f"u{_session_seq}"
    try:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        path = os.path.join(SESSIONS_DIR, f"{sid}.txt")
        f = open(path, "w", encoding="utf-8")
    except Exception:
        f, path = None, None
    # Keep ip/cid only internally (geolocation / bans); the viewer never sees them.
    sessions[sid] = {
        "text": "", "ip": ip, "cid": cid, "flag": "🌐", "file": f, "path": path,
        "socket": socket, "score": None,
    }
    return sid


def _word_accuracy(tokens: list) -> float | None:
    """Fraction of words the model fully predicted (every token its #1 guess),
    matching the frontend: a new word starts at a whitespace-leading token; a
    word counts if it has non-whitespace + a scored token, and is correct only
    if all its scored tokens have rank 0. Returns None if nothing scorable."""
    words: list[list] = []
    cur = None
    for t in tokens:
        txt = t.get("text", "")
        if cur is None or txt[:1].isspace():
            cur = []
            words.append(cur)
        cur.append(t)
    total = hits = 0
    for w in words:
        if not any(tok.get("text", "").strip() for tok in w):
            continue
        scored = [tok for tok in w if tok.get("rank") is not None]
        if not scored:
            continue
        total += 1
        if all(tok.get("rank") == 0 for tok in scored):
            hits += 1
    return hits / total if total else None


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
        "viewer_enabled": Runtime.viewer_enabled,
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
    # If an admin has the live feed turned off, don't reveal conversations — just
    # tell the page it's disabled (admin unlock / ban management still work).
    if Runtime.viewer_enabled:
        await socket.send_text(json.dumps(_sessions_snapshot()))
    else:
        await socket.send_text(json.dumps({"type": "viewer_state", "enabled": False}))
    try:
        while True:
            raw = await socket.receive_text()
            try:
                m = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # Verify the admin password so the viewer page can unlock delete/ban.
            if "admin_check" in m:
                ok = m.get("admin_check") == ADMIN_PASSWORD
                if ok:
                    admin_viewers.add(socket)
                    await socket.send_text(json.dumps({"admin_ok": True, "bans": sorted(BANNED_CIDS)}))
                else:
                    admin_viewers.discard(socket)
                    await socket.send_text(json.dumps({"admin_ok": False}))
            # Admin: delete (clear) one conversation.
            elif m.get("action") == "delete":
                if m.get("password") != ADMIN_PASSWORD:
                    await socket.send_text(json.dumps({"error": "wrong admin password"}))
                else:
                    await _clear_session(m.get("id"))
            # Admin: ban one user's browser (disconnect now + block reconnects).
            elif m.get("action") == "ban":
                if m.get("password") != ADMIN_PASSWORD:
                    await socket.send_text(json.dumps({"error": "wrong admin password"}))
                else:
                    s = sessions.get(m.get("id"))
                    if s is not None:
                        _ban_cid(s.get("cid"))
                        sock = s.get("socket")
                        if sock is not None:
                            try:
                                await sock.send_text(json.dumps({"banned": True}))
                                await sock.close(code=1008)
                            except Exception:
                                pass
                        await _broadcast_bans()
            # Admin: lift a ban.
            elif m.get("action") == "unban":
                if m.get("password") != ADMIN_PASSWORD:
                    await socket.send_text(json.dumps({"error": "wrong admin password"}))
                else:
                    _unban_cid(m.get("cid"))
                    await _broadcast_bans()
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        viewers.discard(socket)
        admin_viewers.discard(socket)


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    ip = _client_ip(socket)
    cid = socket.query_params.get("cid", "")
    if cid and cid in BANNED_CIDS:  # blocked browser — don't create a session
        try:
            await socket.send_text(json.dumps({"banned": True}))
            await socket.close(code=1008)
        except Exception:
            pass
        return
    engine.start()  # idempotent; ensures the worker runs on the live event loop
    clients.add(socket)
    sid = _session_open(ip, cid, socket)  # one streaming file per active user
    await _viewer_broadcast({"type": "open", "id": sid, "flag": sessions[sid]["flag"]})
    asyncio.create_task(_geolocate_session(sid, ip))  # resolve the flag in the background
    await socket.send_text(
        json.dumps({
            "models": MODELS, "default": DEFAULT_MODEL,
            "current": manager.desired_name, "max_tokens": Runtime.max_tokens,
            "viewer_enabled": Runtime.viewer_enabled,
            "server_id": SERVER_ID,
        })
    )

    # A forward pass costs far more than a keystroke, so we DON'T run one per
    # keystroke. Instead a reader coroutine ingests every message (doing the cheap
    # per-keystroke work — session file + viewer stream — for all of them) but only
    # stashes the *latest* prediction request; a worker coroutine runs the model on
    # whatever the latest text is. When someone types faster than the GPU, the
    # intermediate prefixes are skipped: the worker always jumps to the newest text
    # the moment it frees up, instead of grinding through every stale prefix.
    pending: tuple | None = None   # most recent (text, k, seq) not yet predicted
    have_req = asyncio.Event()

    async def reader() -> None:
        nonlocal pending
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

            # --- admin: turn the live conversation viewer on/off (EVERYONE) ---
            if "set_viewer_enabled" in msg:
                if msg.get("password") != ADMIN_PASSWORD:
                    await socket.send_text(
                        json.dumps({"error": "Wrong admin password — unlock the admin panel to toggle the live viewer."})
                    )
                    continue
                Runtime.viewer_enabled = bool(msg.get("set_viewer_enabled"))
                await broadcast({"viewer_enabled": Runtime.viewer_enabled})  # sync admin panels
                await _broadcast_viewer_state()  # flip the /sessions feed for viewers
                continue

            # --- admin: kill + restart the whole server process (EVERYONE) ---
            if "reboot" in msg:
                if msg.get("password") != ADMIN_PASSWORD:
                    await socket.send_text(
                        json.dumps({"error": "Wrong admin password — unlock the admin panel to reboot the server."})
                    )
                    continue
                await broadcast({"status": "rebooting"})  # tell everyone before we go
                # Let the broadcast flush, then re-exec in place (does not return).
                asyncio.get_event_loop().call_later(0.3, _reboot_server)
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

            # --- prediction request: stream text to the viewer for EVERY keystroke,
            #     but only queue the latest for the (expensive) forward pass. ---
            text = msg.get("text", "")
            k = int(msg.get("k", 10))
            seq = msg.get("seq")  # echoed back so the client can drop stale results

            _session_write(sid, text)
            await _viewer_broadcast({"type": "update", "id": sid, "text": text})

            pending = (text, k, seq)  # overwrite: a newer keystroke supersedes older
            have_req.set()

    async def worker() -> None:
        nonlocal pending
        while True:
            await have_req.wait()
            have_req.clear()
            # Atomic grab (no await between read and reset, so the reader can't
            # slip a newer value in unnoticed): take the newest request, drop the
            # rest. Anything typed while we were busy is already coalesced here.
            req, pending = pending, None
            if req is None:
                continue
            text, k, seq = req

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

            # Update this user's live word-prediction score for the viewer page.
            score = _word_accuracy(result.get("tokens", []))
            if sid in sessions:
                sessions[sid]["score"] = score
            await _viewer_broadcast({"type": "score", "id": sid, "score": score})

    worker_task = asyncio.create_task(worker())
    try:
        await reader()  # returns/raises WebSocketDisconnect when the client leaves
    except WebSocketDisconnect:
        pass
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except (asyncio.CancelledError, Exception):
            pass
        clients.discard(socket)
        _session_close(sid)  # close + remove this user's streaming file
        await _viewer_broadcast({"type": "close", "id": sid})


def _gpu_stats() -> dict:
    """A compact per-GPU snapshot via gpustat (nvml) as plain JSON. The browser
    keeps a rolling history and draws a util-over-time bar chart, so we just send
    the numbers (util/mem/temp/power + top compute processes) each tick."""
    try:
        import gpustat

        stats = gpustat.GPUStatCollection.new_query()
        gpus = []
        for g in stats.gpus:
            e = g.entry
            gpus.append({
                "index": g.index,
                "name": e.get("name", ""),
                "util": g.utilization,
                "mem_used": int(g.memory_used),
                "mem_total": int(g.memory_total),
                "temp": g.temperature,
                "power": e.get("power.draw"),
                "power_max": e.get("enforced.power.limit") or e.get("power.limit"),
                "procs": [
                    {"pid": p.get("pid"), "mem": p.get("gpu_memory_usage", 0), "cmd": p.get("command", "")}
                    for p in (g.processes or [])[:5]
                ],
            })
        return {"gpus": gpus}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.websocket("/gpu")
async def gpu(socket: WebSocket):
    """Push a compact GPU snapshot ~once a second; the client keeps the history
    and renders the util-over-time bar chart."""
    await socket.accept()
    try:
        while True:
            await socket.send_text(json.dumps(await asyncio.to_thread(_gpu_stats)))
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
