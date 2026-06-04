---
name: no-free-will
description: Work on the "no free will" live next-token-prediction web app — run/kill the server, swap or add HuggingFace models, debug model loading/quantization, and use the CLI testbed. Use whenever the task touches server.py, predictors.py, run.sh, the model list, GPU/VRAM, or "make model X work".
---

# no free will — maintainer's guide

A FastAPI app where users type and a base LM races to predict their next token
(ghost text + per-token surprisal coloring + top-k panel). One model is held in
VRAM at a time, shared by everyone, with dynamic batching. There's a live viewer
feed at `/viewer.html`. Read `README.md` for the product details.

## Layout
- `predictors.py` — `HFPredictor` (any HF causal LM) + `MockPredictor`. The core
  is `predict_batch`: one left-padded forward pass, reads only the rows it needs
  so the `B×S×vocab` logits never go fp32 (that's what lets a 7B run bf16 on 16GB).
- `server.py` — FastAPI app. `MODELS` list (curated, with per-model `quant`
  hints), `ModelManager` (one global model, serialized swaps), `BatchEngine`
  (coalesces concurrent `/ws` requests *across users* into one forward pass).
  `DEFAULT_MODEL` + the admin panel.
- `/ws` per-connection is a **reader + worker** pair: the reader ingests every
  keystroke (cheap session-file + viewer-stream updates for all of them) but only
  stashes the *latest* prediction request; the worker runs the model on whatever
  the newest text is. So when a user types faster than the GPU, stale intermediate
  prefixes are **skipped** — the worker jumps straight to the current text instead
  of grinding through each prefix. (Verified: a 12-keystroke burst → 2 forwards.)
  The reader/worker share `pending`/`have_req`; the grab is atomic because there's
  no `await` between reading `pending` and resetting it.
- `static/` — frontend. `testbed.py` — CLI smoke-test (see below).
- `run.sh` (serve, default port 8080) / `setup.sh` (one-time deps).

## Environment — IMPORTANT
- This project has its **own** venv at `./.venv`, isolated on purpose. Use it:
  `./.venv/bin/python ...`. `run.sh` calls `./.venv/bin/python -m uvicorn`
  directly (NOT `uv run` — uv prefers an active conda env over `.venv`).
- The venv was built with `--system-site-packages` against the conda env
  `/opt/conda/envs/arena-env` (python 3.11), so it **inherits** that env's
  `torch` (cu126), `bitsandbytes`, `accelerate`, `fastapi`, `uvicorn`, `gpustat`
  — no multi-GB torch re-download — but installs its **own** newer
  `transformers` (5.10.x, needed for Gemma 4) + `huggingface_hub` on top, which
  shadow the inherited ones. Verify with:
  `./.venv/bin/python -c "import transformers; print(transformers.__file__)"`
  (must point inside `./.venv/...`, not the conda env).
- **DO NOT upgrade transformers in arena-env.** It's pinned there for ARENA
  course code (transformer-lens needs numpy<2, inspect-ai needs click<8.2.2,
  etc.). The whole point of the project venv is to leave arena-env alone. As of
  this writing arena-env is: transformers 4.57.6, huggingface-hub 0.36.2,
  click 8.2.1 — keep it that way.
- Rebuild the venv the same way with:
  `PY=/opt/conda/envs/arena-env/bin/python3 SYSTEM_SITE_PACKAGES=1 bash setup.sh`
- GPU: a single **RTX A4000, 16 GB**. Everything is sized to fit 16 GB.

## The CLI testbed — use this BEFORE touching the server
`testbed.py` loads a model exactly like the server (same `build_predictor` +
per-model quant hint) and runs one prediction. Exit code 0 only if it produced
valid predictions, so it's a real smoke test.

```bash
./.venv/bin/python testbed.py --model gpt2 "Hello world"
./.venv/bin/python testbed.py --model google/gemma-4-12B --quant 8bit "some text"
./.venv/bin/python testbed.py --list      # curated models + quant
```
It sets `ADMIN_PASSWORD`/`SESSIONS_DIR` to throwaways before importing `server`,
so it won't clobber a live server's password file or wipe live conversations.
(Importing `server.py` runs module-level side effects: regenerates
`admin_password.txt` and wipes `sessions/*.txt`. Keep that in mind elsewhere.)

## Running / killing the server
```bash
bash run.sh                          # serve on :8080, foreground, tees app.log
MODEL_NAME=google/gemma-4-12B bash run.sh
QUANTIZE=4bit bash run.sh            # force a quant for EVERY model (else per-model)
```
To kill: `pkill -f "run.sh"` often leaves the `uv run`/uvicorn **children**
alive holding VRAM — kill those PIDs directly (`ps aux | grep "port 8080"`,
then `kill -9 <uvicorn-child-pid>`). Confirm VRAM freed with
`nvidia-smi --query-gpu=memory.used --format=csv`. There may be other unrelated
servers (e.g. :8099) — don't kill those unless asked.

## Adding / fixing a model
1. Add `{"id": "...", "label": "...", "quant": "4bit"|"8bit"|None}` to `MODELS`
   in `server.py`. Per-model quant: bf16 for ≤4B, 8-bit for 7–8B, 4-bit only when
   nothing else fits 16 GB (4-bit measurably hurts next-token calibration).
2. Size rule of thumb on 16 GB: bf16 ≈ 2 GB/B params, 8-bit ≈ 1 GB/B, 4-bit ≈
   0.55 GB/B. A 12B is ~24 GB bf16 → needs 4-bit (~7 GB).
3. **Test with `testbed.py` first.** Only set `DEFAULT_MODEL` (in `server.py` and
   `run.sh`) after the testbed reports `OK ✓`.

### Gemma 4 12B specifically (`google/gemma-4-12B`) — the current default
- It's a **multimodal "unified"** checkpoint (`Gemma4UnifiedForConditionalGeneration`,
  model_type `gemma4_unified`), but **transformers maps it under
  `AutoModelForCausalLM`**, so the existing text-only causal-LM path works with no
  predictor code change. Text-only `input_ids` → `.logits` over the 262k vocab.
- Requires **transformers >= 5.10** (4.57 raises "model type `gemma4_unified` not
  recognized") — that's exactly why this project has its own `.venv`.
- 24 GB bf16 single safetensors. We load **8-bit** (`"quant": "8bit"`, LLM.int8,
  near-lossless) → ~13 GB VRAM, fits the 16 GB card with headroom and keeps
  next-token calibration far better than 4-bit (4-bit also worked, ~7 GB, but
  calibration is worse — only use it if 8-bit ever stops fitting). Not gated.
  Loading is layer-by-layer and takes ~70 s from cache. Expect a benign bnb
  warning: "MatMul8bitLt: inputs will be cast from bfloat16 to float16".
- Community **GGUF** quants exist (e.g. Abiray/gemma-4-12b-it-GGUF, Q6_K 9.8 GB /
  Q8_0 12.7 GB) but they're (a) the **instruct** variant, wrong for this
  base-model app, and (b) need a llama.cpp path, not transformers. bnb 8-bit of
  the base model is the right fit here.
- **Xet download flakiness:** the big single file sometimes dies with
  `Internal Writer Error: Background writer channel closed`. Retry with
  `HF_HUB_DISABLE_XET=1` (plain HTTPS) — more reliable for the 24 GB blob.
- **Disk:** the HF cache (`~/.cache/huggingface/hub`) is shared across envs and
  was near-full; freeing space meant deleting an unused cached model. The 100 GB
  overlay fills fast with a few big models.

## GPU monitor (compact JSON + ASCII bars)
`/gpu` streams a small `gpustat`/NVML JSON snapshot (`_gpu_stats`) ~1/s. The
client (`renderGpu` in app.js) draws, per GPU: one info line (name/temp/power/
procs), a horizontal ASCII fill bar for **MEM** and one for current **Util%**,
plus a fixed **32-bar** ASCII util sparkline (`gpuHist`, prefilled with zeros so
it never grows). Panel is `#gpupanel`, monospace `white-space: pre`. (Earlier this
was a real nvtop streamed via a PTY into xterm.js — removed in favour of this
lighter, shorter, dependency-free panel.)

**Caching gotcha (important):** the site is usually reached through a Cloudflare
**trycloudflare** tunnel, which edge-caches `.js`/`.css` by extension — so a stale
`app.js` keeps being served after a deploy no matter how often you reload. Fixed
two ways: a `@app.middleware` sets `Cache-Control: no-cache, must-revalidate` on
`/`+`.html`/`.js`/`.css`, and `index.html` carries `?v=N` query strings on its
asset URLs (bump N to bypass a copy already stuck in the edge cache).

## Memory: vocab-chunked head (CHUNK_VOCAB, opt-in)
`predictors.py` has an optional "online softmax" scoring path (`_score_chunked`)
that streams the LM head over the vocab in `CHUNK_VOCAB`-wide slices so the full
`(B,S,vocab)` logits are never materialized — the big VRAM driver on Gemma's 262k
vocab. Off by default (`CHUNK_VOCAB=0`); set e.g. `CHUNK_VOCAB=8192` to enable.
It gets `last_hidden_state` from `_decoder()` and applies the tied head itself
(with Gemma's `final_logit_softcapping`), keeping running max/Z (probs), a `>`
count (ranks), and a running top-K (next token). Falls back to the full path if it
errors. Measured on the A4000/16GB with gemma-4-12B 8-bit:
- Full path ceiling: `B×S ≤ 2048`. Chunked: `B×S ≤ 4096` for `S≤1024` (≈2× more
  concurrent short-text users). For `S≥2048` the bottleneck shifts to the decoder's
  O(S²) attention, so chunking doesn't extend long-sequence limits.
- Cost: ~2× latency under big batches (32-slice Python loop + fp32 elementwise);
  negligible for typical short single-user inputs. Bigger `CHUNK_VOCAB` = fewer
  slices = faster but more per-slice memory.
- Correctness: probs match the full path to ~1e-6 in fp32; in bf16/int8 the top-k
  ordering matches except occasional tail near-tie swaps (acceptable/approximate).

## Git workflow (REQUIRED)
For every new feature/change like the ones in this project, do NOT commit straight
to `main`. Create a **new branch**, commit the work there, and **push the branch**.
Only merge into `main` when the user **explicitly orders it** — never merge to main
on your own initiative. Treat `main` as protected; branches are where work lands
until the user says to merge.

## Layout note
The right-hand `aside` order is: word-accuracy `.stat` → `Next token, predicted`
(`#predictions`) → status → one collapsible `<details class="settings">` holding
top-k + current model + all admin controls → token count. Keeping admin inside
the collapsible keeps the page short.

## Admin controls
Admin actions are gated by `ADMIN_PASSWORD` (env, or random → `admin_password.txt`)
and enforced **server-side** in the `/ws` reader (the UI is cosmetic). Pattern for
adding one: handle a `{"set_X": ..., "password": ...}` message in the reader, store
on `Runtime`, `broadcast({...})` to sync `/ws` admin panels, add it to the `/ws`
hello + `/api/config`, and wire a control in `static/index.html` + `app.js`.
Existing ones: `set_model`, `set_max_tokens`, `set_viewer_enabled`.

- `set_viewer_enabled` (`Runtime.viewer_enabled`, env `VIEWER_ENABLED=0` to start
  off) turns the `/viewer.html` live feed on/off for everyone. Enforcement:
  `_viewer_broadcast` early-returns when off (no conversation data leaks), new
  `/sessions` connects get `{type:"viewer_state",enabled:false}` instead of a
  snapshot, and toggling broadcasts `viewer_state` to viewers (+snapshot on
  re-enable). Admin unlock/ban management on the viewer page still works while off.

## Gotchas
- transformers 5.x deprecated `torch_dtype=` → use `dtype=` (already changed in
  `predictors.py`). bitsandbytes 4-bit needs CUDA; placed by accelerate
  (`device_map="auto"`), can't be `.to()`'d afterward.
- The `/api/config` `current` reflects `manager.desired_name`, which may differ
  from what's actually resident while a swap loads.
- VRAM: only ONE model fits. A failed/oversized load can OOM — `free_predictor`
  is called before each swap, but verify VRAM after switching big models.
