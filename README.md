# no free will

A website where you type freely while a language model races to predict your
next token **before you commit it**. As you type:

- the model's **#1 next-token guess** trails your cursor as faded ghost text;
- each **token is highlighted by surprisal** — green where the model predicted
  you (it was *bored*), red where you surprised it — so subword splits are
  visible, not just whole words;
- a side panel shows the **top-k next-token distribution** with live bars, the
  **token count** (`current / max`), and a **word-level score**: *"X% of your
  words the model nailed"* — a word only counts if **every** token in it was the
  model's #1 guess.

Everyone shares **one model at a time** (kept in VRAM), and concurrent typists
are **batched into a single forward pass**. The default model is
**Gemma 4 12B (base)**, loaded **8-bit** (near-lossless, ~13 GB) so it fits a
16 GB card.

There's also a **live feed** at `/viewer.html` showing every connected person's
typing in real time, with a country flag and their live score.

## Quick start

```bash
bash setup.sh    # one-time: installs uv + a .venv with all deps
bash run.sh      # serve on http://localhost:8080  (re-run to serve)
```

First run downloads the model from HuggingFace (~15 GB for the 7B). Logs stream
to the terminal **and** to an in-page panel. Then open <http://localhost:8080>.

Common overrides (env vars + `--port`):

```bash
bash run.sh --port 9000                      # different port
MODEL_NAME=Qwen/Qwen3-4B-Base bash run.sh    # start on a different model
TUNNEL=1 bash run.sh                         # also open a public cloudflared URL
ADMIN_PASSWORD='secret' bash run.sh          # set the admin password (see below)
MODEL_NAME=mock bash run.sh                  # UI only, no torch/GPU
```

### Manual run

The app runs from its **own** `.venv` (isolated so its newer `transformers`
doesn't disturb any conda env). `setup.sh` builds it; then:

```bash
MODEL_NAME=google/gemma-4-12B ./.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8080
```

There's also a CLI smoke-test — load any model + check it predicts before wiring
it into the server:

```bash
./.venv/bin/python testbed.py --model google/gemma-4-12B --quant 8bit "some text"
./.venv/bin/python testbed.py --list
```

## Models & quantization

Switch the model from the **admin panel dropdown** in the page (see below). Only
one model is loaded at a time — switching **evicts** the previous one and
changes it for **everyone** (all clients re-sync). The curated list is
base/pretrained models, sized for a single **16 GB** GPU:

| Model | How it loads on 16 GB |
|-------|-----------------------|
| GPT-2 small · Qwen3 0.6B/1.7B/4B · MiniCPM5-1B | full **bf16** |
| Qwen2.5-7B, Falcon3-7B | full **bf16** (~14 GB) |
| Qwen3-8B, Llama-3.1-8B | **8-bit** (near-lossless) |
| **Gemma 4 12B** (base, multimodal · **default**) | **8-bit** (~13 GB) |
| Qwen3-14B | **4-bit** (only thing that needs it) |

Quantization is chosen **per model** automatically (small → bf16, 8B → 8-bit,
14B → 4-bit), because 4-bit measurably hurts next-token calibration and we only
use it when nothing else fits. Override globally with `QUANTIZE=4bit|8bit` (CUDA
only); leave it empty for per-model defaults. `MODEL_NAME` sets the startup
model and may be any HuggingFace causal LM id (custom ids join the dropdown).

> Llama-3.1-8B is **gated** — needs an accepted license and a `HF_TOKEN`.
> `google/gemma-4-12B` is a multimodal `*ForConditionalGeneration`, but
> transformers maps it under `AutoModelForCausalLM`, so the text-only path works.
> It needs **transformers ≥ 5.10**, which is why this project ships its own
> isolated `.venv` (see below) rather than relying on a system/conda env.

## Admin panel

Some actions are gated by an admin password, enforced **server-side** (the
dropdown/buttons are just cosmetic — the server rejects any action without the
password). The password is read **only** from the `ADMIN_PASSWORD` env var; if
unset, a random one is generated into `admin_password.txt` (gitignored) and
**not printed** (the log is public). Start with your own:

```bash
ADMIN_PASSWORD='whatever' bash run.sh
```

Unlock the **🔒 admin** panel in the side bar to:

- **switch the model** (affects everyone);
- **adjust the token cap** — `MAX_TOKENS` (default **1024**) bounds VRAM by
  truncating each request; the status shows `current / max tok (truncated)`.
- **enable/disable the live conversation viewer** — a checkbox that turns the
  `/viewer.html` feed on or off for everyone. When off, the server stops
  streaming any conversation text to the feed (enforced server-side, not just
  hidden) and the viewer page shows an "off" notice; admin unlock / ban
  management on the viewer page still work. Default is on; override the startup
  default with `VIEWER_ENABLED=0`.

## Live conversations (`/viewer.html`)

A separate page streams every connected user's current text live (one panel
each, appearing/updating/disappearing as people type and come/go). Each panel
shows a **country flag** (geolocated from the IP via ip-api — the raw **IP is
never sent to the browser**) and the user's **live word-prediction score**.

Unlock with the admin password to get, per panel:

- **✕ delete** — wipe that conversation (and clear that user's editor);
- **ban** — block that **browser** (a per-browser token, *not* the IP, so it
  won't catch NAT/household neighbours; evadable via incognito). Bans persist to
  `banned_clients.txt` and there's an **unban list** in the same panel.

Conversations are **ephemeral**: each active user maps to `sessions/<id>.txt`,
rewritten per keystroke and **deleted on disconnect**; the dir is also wiped on
startup. Nothing is archived, and the text appears in no logs.

## Observability

Below the editor are two live terminals:

- **server log** — streams `app.log` (model download/load progress included);
- **GPU** — a compact live readout: the server streams a small `gpustat`/NVML
  JSON snapshot each second (`/gpu`), and the page draws one info line per GPU
  plus horizontal ASCII bars for **MEM** and current **Util%**, and a fixed
  32-bar ASCII sparkline of utilisation over time.

A pill in the header shows whether you're running the latest code: **green** =
up to date, **yellow** = a static change needs a page reload *or* a backend
(`.py`) change needs a **server restart** (a browser can't restart the server,
so that one only clears when you restart the process).

## Hosting for visitors

A single FastAPI process serves static files plus WebSockets (`/ws`, `/logs`,
`/gpu`, `/sessions`), so anything that proxies WebSockets works:

- **Quick share:** `TUNNEL=1 bash run.sh` starts cloudflared and prints a public
  URL; or run `cloudflared tunnel --url http://localhost:8080` yourself.
- **Behind nginx/Caddy:** proxy `/` and the WebSocket paths to the uvicorn port;
  the client auto-uses `wss://` over HTTPS.

Dynamic batching means a single GPU serves a handful of simultaneous typists
comfortably. Tune with `MAX_BATCH` (default 8) and `BATCH_WINDOW_MS` (default 8).

## How it works

- **`predictors.py`** — `HFPredictor` loads any causal LM. `predict_batch` runs
  several requests through **one left-padded forward pass** (correct
  `position_ids` + attention mask, so results match the unbatched path even on
  absolute-position models like GPT-2). It returns the top-k next-token
  distribution *and* per-token probability/rank for the text already typed, and
  only `log_softmax`es the rows it reads (so the full `B×S×vocab` logits never
  hit fp32 — that's what lets a 7B run in bf16 on 16 GB). A `MockPredictor`
  mirrors the interface with zero heavy deps.
- **`server.py`** — FastAPI app. A `BatchEngine` coalesces concurrent `/ws`
  requests *across users*; a `ModelManager` holds the one global model and
  serializes swaps. Each `/ws` connection is a **reader + worker** pair: the
  reader streams every keystroke to the viewer but only keeps the *latest*
  prediction request, and the worker runs the model on the newest text — so
  typing faster than the GPU **skips the stale intermediate prefixes** instead of
  running a forward pass per keystroke. Also serves `/logs`, `/gpu`, `/sessions`
  (the viewer feed), `/api/config`, and `/api/version`.
- **`static/`** — a transparent `<textarea>` over a styled backdrop that renders
  surprisal-colored token chips + the ghost prediction. While you type, the
  already-scored prefix keeps its colors and only the new tail is plain, so the
  highlighting never flickers. `viewer.html` is the live feed.

### Beginning-of-sequence handling

Text is encoded **without** the tokenizer's automatic specials, then exactly one
start token is prepended so the first typed token has the context the model
expects: a model's real BOS where it has one (Llama/SmolLM style), or
`<|endoftext|>` (the pretraining document separator) for GPT-2 / Qwen. This
avoids the double-BOS that naive `encode()` causes.

### Multi-GPU / sharding

When more than one CUDA device is visible, models load with `device_map="auto"`
and shard across all of them automatically. Pin which GPUs with
`CUDA_VISIBLE_DEVICES=0,1,...`. The server auto-selects `cuda` → `mps` → `cpu`.

### Mock mode

```bash
MODEL_NAME=mock bash run.sh
```

Runs the whole site with a fake predictor and **no torch/transformers** — handy
for developing the frontend.
