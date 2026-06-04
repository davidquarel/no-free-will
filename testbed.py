#!/usr/bin/env python3
"""CLI testbed for the next-token predictors.

Give it a model and a string; it loads the model exactly the way the server
does (via build_predictor + the server's per-model quant hint) and runs one
prediction, printing the top-k next-token guesses and the per-token surprisal
of the text you gave it. Use it to check a model loads and behaves *before*
wiring it into the live server.

    python testbed.py --model google/gemma-4-12B "The quick brown fox"
    python testbed.py --model gpt2 "Hello world"        # uses the server's
                                                          # quant hint for known
                                                          # models
    python testbed.py --model Qwen/Qwen3-14B-Base --quant 4bit "some text"
    python testbed.py --list                              # show the curated list

Exit code is 0 only if the model loaded AND produced a sane prediction, so it
doubles as a smoke test in scripts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# Importing server.py runs its module-level startup (writes admin_password.txt,
# wipes the sessions/ dir). Point those at harmless throwaways BEFORE the import
# so the testbed never disturbs a live server's password file or conversations.
os.environ.setdefault("ADMIN_PASSWORD", "testbed")
os.environ.setdefault("SESSIONS_DIR", os.path.join(os.path.dirname(__file__), ".testbed_sessions"))

from predictors import build_predictor, free_predictor

# Reuse the server's curated list + per-model quant hints so the testbed loads a
# model the same way the live server would (same quant, same id).
try:
    from server import MODELS, resolve_quant
except Exception:  # server import shouldn't pull in torch, but be defensive
    MODELS, resolve_quant = [], lambda name: None  # type: ignore


DEFAULT_TEXT = "The quick brown fox jumps over the lazy"


def _fmt_tok(s: str) -> str:
    """Make whitespace visible so token boundaries are obvious."""
    return s.replace("\n", "\\n").replace("\t", "\\t")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Load a model and check it predicts.")
    ap.add_argument("text", nargs="?", default=DEFAULT_TEXT, help="text to score")
    ap.add_argument("--model", "-m", default="gpt2", help="HF model id (or 'mock')")
    ap.add_argument(
        "--quant", "-q", default=None,
        help="force '4bit'/'8bit'/'' (none). Default: the server's per-model hint.",
    )
    ap.add_argument("-k", type=int, default=10, help="top-k to show (default 10)")
    ap.add_argument("--list", action="store_true", help="print the curated model list and exit")
    args = ap.parse_args(argv)

    if args.list:
        for m in MODELS:
            q = m.get("quant") or "bf16"
            print(f"  {m['id']:<32} [{q}]  {m.get('label','')}")
        return 0

    # If the user didn't force a quant, use the server's per-model hint so the
    # testbed mirrors production loading exactly.
    quant = args.quant if args.quant is not None else resolve_quant(args.model)

    print(f">> model:  {args.model}")
    print(f">> quant:  {quant or 'bf16 (none)'}")
    print(f">> text:   {args.text!r}")
    print(">> loading… (first run downloads weights; can be slow)", flush=True)

    t0 = time.time()
    try:
        predictor = build_predictor(args.model, quantize=quant)
    except Exception as exc:
        print(f"\n!! LOAD FAILED: {type(exc).__name__}: {exc}\n", file=sys.stderr)
        traceback.print_exc()
        return 1
    load_s = time.time() - t0
    dev = getattr(predictor, "input_device", getattr(predictor, "device", "?"))
    print(f">> loaded in {load_s:.1f}s on {dev}", flush=True)

    try:
        t1 = time.time()
        res = predictor.predict(args.text, k=args.k)
        infer_s = time.time() - t1
    except Exception as exc:
        print(f"\n!! PREDICT FAILED: {type(exc).__name__}: {exc}\n", file=sys.stderr)
        traceback.print_exc()
        free_predictor(predictor)
        return 1

    preds = res.get("predictions", [])
    toks = res.get("tokens", [])

    print(f"\n>> inference {infer_s * 1000:.0f}ms — {len(toks)} tokens scored\n")
    print(f"top-{args.k} next-token guesses:")
    for i, p in enumerate(preds):
        bar = "█" * int(round(p["prob"] * 30))
        print(f"  {i:>2}. {p['prob']*100:5.1f}%  {bar:<30} {_fmt_tok(p['token'])!r}")

    print("\nper-token surprisal of your text (rank 0 = model's #1 guess):")
    for t in toks:
        rank = t.get("rank")
        prob = t.get("prob")
        mark = "·" if rank == 0 else (" " if rank is None else "✗")
        prob_s = "  n/a" if prob is None else f"{prob*100:5.1f}%"
        rank_s = "  -" if rank is None else f"{rank:>3}"
        print(f"  {mark} rank {rank_s}  p={prob_s}  {_fmt_tok(t.get('text',''))!r}")

    # Sanity: a working model returns >=1 prediction whose probs are finite and
    # sum to <= ~1. Anything else is a broken load even if it didn't throw.
    ok = bool(preds) and all(
        isinstance(p.get("prob"), float) and 0.0 <= p["prob"] <= 1.0001 for p in preds
    )
    print(f"\n>> RESULT: {'OK ✓' if ok else 'BROKEN ✗ (no/invalid predictions)'}")
    free_predictor(predictor)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
