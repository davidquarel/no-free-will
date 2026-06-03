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


def build_predictor(model_name: str | None = None) -> "BasePredictor":
    """Factory: pick a predictor from MODEL_NAME (or the given name)."""
    name = model_name or os.environ.get("MODEL_NAME", "gpt2")
    if name == "mock":
        return MockPredictor()
    return HFPredictor(name)


class BasePredictor:
    name: str = "base"

    def predict(self, text: str, k: int = 10) -> dict[str, Any]:
        raise NotImplementedError


class HFPredictor(BasePredictor):
    """Wraps a HuggingFace causal language model.

    GPT-2 (and friends) have no beginning-of-sequence token, so we prepend the
    model's bos/eos id when available. That gives even the very first typed
    token a real probability and rank, and keeps token indices aligned with the
    text the user sees.
    """

    def __init__(self, model_name: str = "gpt2"):
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

        # Shard across all visible GPUs when there's more than one (e.g. a
        # 14B model across 4xA4000). accelerate's device_map="auto" splits the
        # layers and moves activations between cards automatically; inputs go
        # to cuda:0. With a single GPU we just .to() it; CPU/MPS likewise.
        n_gpus = torch.cuda.device_count() if self.device == "cuda" else 0
        if n_gpus > 1:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="auto"
            )
            self.input_device = "cuda:0"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, low_cpu_mem_usage=True
            ).to(self.device)
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
        torch = self.torch
        with self._lock, torch.no_grad():
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids = ([self.prefix_id] if self.prefix_id is not None else []) + ids
            # Need at least one token to get a "next token" distribution.
            if not input_ids:
                input_ids = [self.prefix_id if self.prefix_id is not None else 0]

            tensor = torch.tensor([input_ids], device=self.input_device)
            # Upcast logits to fp32 before softmax for stable probabilities
            # even when the model runs in bf16/fp16.
            logits = self.model(tensor).logits[0].float()  # (seq, vocab)
            logprobs = torch.log_softmax(logits, dim=-1)

            # --- top-k prediction for the NEXT token (from the last position) ---
            last = logprobs[-1]
            top = torch.topk(last, min(k, last.shape[-1]))
            predictions = [
                {"token": self._decode(int(tid)), "prob": float(math.exp(lp))}
                for lp, tid in zip(top.values.tolist(), top.indices.tolist())
            ]

            # --- per-token analysis of the text the user already typed ---
            # Token at position i (in `ids`) is predicted by row (offset + i - 1).
            offset = 1 if self.prefix_id is not None else 0
            tokens = []
            for i, tid in enumerate(ids):
                row = logprobs[offset + i - 1] if (offset + i - 1) >= 0 else None
                if row is None:
                    tokens.append({"text": self._decode(tid), "prob": None, "rank": None})
                    continue
                lp = float(row[tid])
                # rank = number of tokens strictly more likely than this one.
                rank = int((row > row[tid]).sum().item())
                tokens.append(
                    {"text": self._decode(tid), "prob": float(math.exp(lp)), "rank": rank}
                )

        return {"model": self.name, "predictions": predictions, "tokens": tokens}


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
