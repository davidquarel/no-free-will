"""FastAPI server: live next-token prediction over a WebSocket.

Run:
    pip install -r requirements.txt
    MODEL_NAME=gpt2 uvicorn server:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000  (or point a browser at the host's IP).

Set MODEL_NAME=mock to run without torch/transformers.
The model is loaded lazily on the first WebSocket connection so the HTTP
server (and a friendly error page) is available immediately.
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from predictors import build_predictor

app = FastAPI(title="no-free-will")

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2")

_predictor = None
_predictor_error: str | None = None


def get_predictor():
    """Load the model once, lazily. Cache the error if it fails."""
    global _predictor, _predictor_error
    if _predictor is None and _predictor_error is None:
        try:
            _predictor = build_predictor(MODEL_NAME)
        except Exception as exc:  # surface load failures to the client
            _predictor_error = f"{type(exc).__name__}: {exc}"
    return _predictor


@app.get("/api/config")
def config():
    return {"model": MODEL_NAME}


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    predictor = get_predictor()
    if predictor is None:
        await socket.send_text(
            json.dumps({"error": f"Failed to load model '{MODEL_NAME}': {_predictor_error}"})
        )
        await socket.close()
        return

    await socket.send_text(json.dumps({"ready": True, "model": predictor.name}))

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
            try:
                result = predictor.predict(text, k=k)
            except Exception as exc:
                await socket.send_text(json.dumps({"error": f"{type(exc).__name__}: {exc}", "seq": seq}))
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
