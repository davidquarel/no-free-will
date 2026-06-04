"""Next-token predictors.

A predictor takes the current text and returns:
  * `predictions`: the model's top-k guesses for the NEXT token (token + prob)
  * `tokens`: per-token analysis of the text already typed, where each entry
    reports the token's probability and rank under the model's distribution
    at that position (i.e. how well the model predicted *you*).

Two implementations are provided:
  * `HFPredictor`  - any HuggingFace causal LM (default: gpt2 / "gpt2-small").
  * `MockPredictor`- deterministic fake, so the server/frontend run without
                     torch installed. Select with MODEL_NAME=mock.
"""

from __future__ import annotations

import os
import math
import threading
from typing import Any


def build_predictor(model_name: str | None = None, quantize: str | None = None) -> "BasePredictor":
    """Factory: pick a predictor from MODEL_NAME (or the given name).

    `quantize` ("4bit"/"8bit"/"" for none) is resolved per-model by the caller;
    if left None, HFPredictor falls back to the QUANTIZE env var."""
    name = model_name or os.environ.get("MODEL_NAME", "gpt2")
    if name == "mock":
        return MockPredictor()
    return HFPredictor(name, quantize=quantize)


class BasePredictor:
    name: str = "base"

    def predict(self, text: str, k: int = 10) -> dict[str, Any]:
        raise NotImplementedError

    def predict_batch(
        self, reqs: list[tuple[str, int]], max_tokens: int | None = None
    ) -> list[dict[str, Any]]:
        """Predict several (text, k) requests at once. Default: just loop. The
        HF predictor overrides this with a single padded forward pass so that
        concurrent users share one GPU call."""
        out = []
        for text, k in reqs:
            r = self.predict(text, k)
            r.setdefault("n_tokens", len(r.get("tokens", [])))
            r.setdefault("n_tokens_total", r["n_tokens"])
            r["max_tokens"] = max_tokens
            out.append(r)
        return out


class HFPredictor(BasePredictor):
    """Wraps a HuggingFace causal language model.

    GPT-2 (and friends) have no beginning-of-sequence token, so we prepend the
    model's bos/eos id when available. That gives even the very first typed
    token a real probability and rank, and keeps token indices aligned with the
    text the user sees.
    """

    def __init__(self, model_name: str = "gpt2", quantize: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_name

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        )

        # On GPU, load weights in half precision so multi-billion-param models
        # fit in modest VRAM (a 4B model is ~16GB in fp32 but only ~8GB in
        # bf16). Prefer bf16 where supported (Qwen3 is trained in bf16), else
        # fp16. CPU stays fp32 for correctness/speed.
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Optional bitsandbytes quantization (CUDA only), set via QUANTIZE=
        # "4bit" or "8bit". 4-bit NF4 lets a 14B model run in ~9GB so it fits a
        # single 16GB card while keeping most of its quality.
        # Per-model choice wins; fall back to the QUANTIZE env var if unset.
        quant = quantize if quantize is not None else os.environ.get("QUANTIZE", "")
        quant = (quant or "").lower().replace("-", "")
        quant_config = None
        if self.device == "cuda" and quant in ("4bit", "4", "nf4"):
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif self.device == "cuda" and quant in ("8bit", "8"):
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # transformers 5.x renamed `torch_dtype` -> `dtype` (the old name warns).
        common = dict(dtype=dtype, low_cpu_mem_usage=True)
        n_gpus = torch.cuda.device_count() if self.device == "cuda" else 0

        if quant_config is not None:
            # Quantized weights are placed by accelerate and can't be .to()'d.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quant_config, device_map="auto", **common
            )
            self.input_device = "cuda:0"
        elif n_gpus > 1:
            # Shard across all visible GPUs (e.g. a 14B model across 4xA4000).
            # device_map="auto" splits layers and moves activations between
            # cards automatically; inputs go to cuda:0.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", **common
            )
            self.input_device = "cuda:0"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **common).to(self.device)
            self.input_device = self.device
        self.model.eval()

        # Beginning-of-sequence handling. We encode content WITHOUT the
        # tokenizer's automatic specials (add_special_tokens=False) and then
        # prepend exactly one start token ourselves, so the first typed token
        # always has the context the model was pretrained to expect:
        #   * Llama/SmolLM-style models -> their real BOS token.
        #   * GPT-2 / Qwen (no dedicated BOS) -> <|endoftext|>, the document
        #     separator used during pretraining, which acts as "start of doc".
        # This also avoids the double-BOS bug you'd get from letting encode()
        # add a BOS and then prepending another.
        self.prefix_id = self.tokenizer.bos_token_id
        if self.prefix_id is None:
            self.prefix_id = self.tokenizer.eos_token_id

        # Inference is not thread-safe across requests; serialize it.
        self._lock = threading.Lock()

    def _decode(self, token_id: int) -> str:
        # Decode a single id so leading spaces ("Ġ") render naturally.
        return self.tokenizer.decode([token_id])

    def predict(self, text: str, k: int = 10) -> dict[str, Any]:
        return self.predict_batch([(text, k)])[0]

    def predict_batch(
        self, reqs: list[tuple[str, int]], max_tokens: int | None = None
    ) -> list[dict[str, Any]]:
        """Run several requests through a single padded forward pass.

        Sequences are LEFT-padded so every row's final real token lands at
        index -1 (uniform next-token slice), and explicit position_ids + an
        attention mask keep results identical to the unbatched path — correct
        even for models with learned absolute positions (e.g. GPT-2), not just
        RoPE models. The per-token bookkeeping is the unbatched logic shifted by
        each row's left-pad width."""
        torch = self.torch
        with self._lock, torch.no_grad():
            # Encode every request, cap to max_tokens (drop the rest), and
            # prepend the one start token.
            seqs = []  # (full_input_ids, content_ids, total_tokens_before_cap)
            for text, _k in reqs:
                if max_tokens is not None:
                    text = text[: max_tokens * 16]  # bound tokenizer work on huge pastes
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                total = len(ids)
                if max_tokens is not None and total > max_tokens:
                    ids = ids[:max_tokens]  # keep the first max_tokens, truncate after
                full = ([self.prefix_id] if self.prefix_id is not None else []) + ids
                if not full:  # need >=1 token to get a next-token distribution
                    full = [self.prefix_id if self.prefix_id is not None else 0]
                seqs.append((full, ids, total))

            B = len(seqs)
            S = max(len(full) for full, _ids, _total in seqs)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            if pad_id is None:
                pad_id = 0

            input_ids = torch.full((B, S), pad_id, dtype=torch.long)
            attn = torch.zeros((B, S), dtype=torch.long)
            for b, (full, _ids, _total) in enumerate(seqs):
                L = len(full)
                input_ids[b, S - L:] = torch.tensor(full, dtype=torch.long)  # left pad
                attn[b, S - L:] = 1
            input_ids = input_ids.to(self.input_device)
            attn = attn.to(self.input_device)
            # Correct 0-based positions for each (left-padded) row.
            position_ids = (attn.long().cumsum(-1) - 1).clamp(min=0)

            # Keep the (B, S, vocab) logits in the model's dtype — upcasting the
            # whole tensor to fp32 would cost gigabytes for a 150k vocab and is
            # what tips a 7B/8B model over 16GB. log_softmax is per-row, so we
            # upcast only the individual rows we actually read, below.
            logits = self.model(
                input_ids=input_ids, attention_mask=attn, position_ids=position_ids
            ).logits  # (B, S, vocab), model dtype

            offset = 1 if self.prefix_id is not None else 0
            out = []
            for b, (full, ids, total) in enumerate(seqs):
                k = reqs[b][1]
                pad_b = S - len(full)

                # --- top-k prediction for the NEXT token (last position) ---
                last = torch.log_softmax(logits[b, -1].float(), dim=-1)
                top = torch.topk(last, min(k, last.shape[-1]))
                predictions = [
                    {"token": self._decode(int(tid)), "prob": float(math.exp(lp))}
                    for lp, tid in zip(top.values.tolist(), top.indices.tolist())
                ]

                # --- per-token analysis of the already-typed text ---
                # Content token i is predicted by unbatched row (offset+i-1),
                # which sits at (pad_b + offset + i - 1) once left-padded.
                tokens = []
                for i, tid in enumerate(ids):
                    j = offset + i - 1
                    if j < 0:
                        tokens.append({"text": self._decode(tid), "prob": None, "rank": None})
                        continue
                    row = logits[b, pad_b + j]  # (vocab,), model dtype
                    lp = float(torch.log_softmax(row.float(), dim=-1)[tid])
                    # rank = tokens strictly more likely; softmax is monotonic so
                    # comparing raw logits gives the same ordering.
                    rank = int((row > row[tid]).sum().item())
                    tokens.append(
                        {"text": self._decode(tid), "prob": float(math.exp(lp)), "rank": rank}
                    )

                out.append({
                    "model": self.name,
                    "predictions": predictions,
                    "tokens": tokens,
                    "n_tokens": len(ids),        # tokens actually scored (after cap)
                    "n_tokens_total": total,     # tokens the user actually typed
                    "max_tokens": max_tokens,
                })
            return out


class MockPredictor(BasePredictor):
    """No-dependency stand-in. Splits on whitespace and fabricates plausible
    numbers so the UI is fully exercisable without torch."""

    name = "mock"

    _VOCAB = [
        " the", " a", " of", " to", " and", " in", " is", " that", " it",
        " model", " token", " next", " predict", ".", ",", "\n",
    ]

    def predict(self, text: str, k: int = 10) -> dict[str, Any]:
        import random

        seed = abs(hash(text)) % (2**32)
        rng = random.Random(seed)

        # Fake but normalized top-k.
        picks = rng.sample(self._VOCAB, min(k, len(self._VOCAB)))
        raw = sorted((rng.random() for _ in picks), reverse=True)
        s = sum(raw) or 1.0
        predictions = [
            {"token": tok, "prob": p / s} for tok, p in zip(picks, raw)
        ]

        # Re-split text into pseudo-tokens (keep leading spaces, GPT-2 style).
        toks = _mock_tokenize(text)
        tokens = []
        for t in toks:
            r = random.Random(abs(hash(t)) % (2**32))
            rank = r.randint(0, 50)
            prob = max(1e-4, r.random() ** (1 + rank / 5))
            tokens.append({"text": t, "prob": prob, "rank": rank})

        return {"model": self.name, "predictions": predictions, "tokens": tokens}


def free_predictor(predictor: "BasePredictor") -> None:
    """Release a predictor's model and reclaim VRAM. Used before loading a
    different model so we never hold two large models at once on a 16GB GPU."""
    import gc

    torch = getattr(predictor, "torch", None)
    model = getattr(predictor, "model", None)
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
    for attr in ("model", "tokenizer"):
        if hasattr(predictor, attr):
            setattr(predictor, attr, None)
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mock_tokenize(text: str) -> list[str]:
    out, cur = [], ""
    for ch in text:
        if ch == " " and cur:
            out.append(cur)
            cur = " "
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out
