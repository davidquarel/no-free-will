# no free will

A website where you type freely while a language model races to predict your
next token **before you commit it**. As you type:

- the model's **#1 next-token guess** trails your cursor as faded ghost text;
- your words are **tinted by surprisal** — green where the model predicted you,
  red where you surprised it;
- a side panel shows the **top-k next-token distribution** with live bars and a
  running score: *"X% of your tokens were the model's #1 guess"*.

Default model is **gpt2** (a.k.a. gpt2-small), but any HuggingFace causal LM
works — just set `MODEL_NAME`.

## Run it

```bash
pip install -r requirements.txt

# default: gpt2-small on GPU if available, else CPU
MODEL_NAME=gpt2 uvicorn server:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. To let other people visit, run it on your
GPU box and point them at that machine's address (or front it with a reverse
proxy / tunnel — see below).

### Choosing a model

There's a **dropdown in the page** to switch models live (it changes the model
for everyone — only one model is held in memory at a time, and switching evicts
the previous one). The curated list is base/pretrained models suited to
next-token prediction. **Qwen3-14B-Base** is the recommended strong pick:
Qwen reports it matching Qwen2.5-32B-Base quality, and at ~28GB (bf16) it shards
comfortably across a 4xA4000 box.

`MODEL_NAME` sets the default and may be any HuggingFace causal LM id (custom
ids are added to the dropdown automatically):

```bash
MODEL_NAME=Qwen/Qwen3-14B-Base uvicorn server:app --host 0.0.0.0 --port 8000
```

### Multi-GPU / sharding

When more than one CUDA device is visible, models load with
`device_map="auto"` and shard across all of them automatically — no flags
needed. So on 4xA4000 (~64GB total) a 14B model just works; inputs go to
`cuda:0` and activations hop between cards.

Weights load in **bf16** (or fp16 if bf16 is unsupported), so a 14B model is
~28GB rather than ~56GB. The server auto-selects `cuda` → `mps` → `cpu`.

To pin which GPUs are used: `CUDA_VISIBLE_DEVICES=0,1,2,3 uvicorn ...`.

### Beginning-of-sequence handling

Text is encoded **without** the tokenizer's automatic special tokens, then
exactly one start token is prepended so the first typed token has the context
the model expects: a model's real BOS where it has one (Llama/SmolLM style), or
`<|endoftext|>` (the pretraining document separator) for GPT-2 / Qwen, which
have no dedicated BOS. This avoids the double-BOS that naive `encode()` causes.

### No GPU / no torch? Mock mode

```bash
MODEL_NAME=mock uvicorn server:app --port 8000
```

Runs the full website with a fake predictor and **no torch/transformers
dependency** — useful for trying the UI or developing the frontend.

## Hosting for visitors

The app is a single FastAPI process serving static files plus a WebSocket
(`/ws`), so anything that proxies WebSockets works:

- **Quick share:** `cloudflared tunnel --url http://localhost:8000` or
  `ngrok http 8000` gives you a public URL in seconds.
- **Behind nginx/Caddy:** proxy `/` and `/ws` to the uvicorn port; the client
  auto-uses `wss://` when served over HTTPS.

Inference is serialized with a lock and recomputed per keystroke (debounced
client-side to ~14 req/s), so a single GPU comfortably serves a handful of
simultaneous typists with gpt2-small. For heavier load, run multiple workers
behind the proxy.

## How it works

- **`predictors.py`** — `HFPredictor` loads any causal LM. For each request it
  runs one forward pass over the current text and returns the top-k next-token
  distribution *and* per-token probability/rank for the text already typed (so
  the UI can color surprisal and score the model). A `MockPredictor` mirrors the
  same interface with zero heavy deps.
- **`server.py`** — FastAPI app. `/ws` streams predictions; the model is loaded
  lazily on first connect so the page comes up instantly. Static frontend is
  served from `static/`.
- **`static/`** — a transparent `<textarea>` stacked over a styled backdrop that
  renders surprisal-colored token spans + the ghost prediction. The backdrop
  updates instantly to plain text on each keystroke and is *upgraded* to colored
  tokens once the matching model result arrives, so the caret never drifts.

### Notes

- Predictions are at the **token** level (BPE), so the ghost text may be a word
  fragment — that's the model's actual unit of prediction.
- For non-ASCII text where a character spans multiple BPE tokens, per-token
  coloring gracefully falls back to plain (uncolored) text for that snapshot.
