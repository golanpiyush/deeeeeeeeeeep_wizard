"""
server.py
==========
FastAPI backend for DepthWizard.

Endpoints:
    GET  /
        Health check. Returns status + active model info.

    POST /predict
        Body: an uploaded image file (multipart/form-data)
        Returns: JSON point cloud { points, colors, count }
        (No live progress here — use the WebSocket version below for that.)

    WS   /ws/predict
        Same job as POST /predict, but over a WebSocket so the frontend
        can receive LIVE progress events while the model runs — model
        loading, inference timing, point-cloud chunk-by-chunk placement —
        instead of just waiting on a spinner. This is what powers the
        live terminal view in the frontend.

        Protocol (all messages are JSON text frames):
            Client -> Server:  {"image_b64": "<base64 image bytes>"}
            Server -> Client:  {"type": "model_load", "message": "...", ...}
                               {"type": "inference_start", ...}
                               {"type": "pointcloud_chunk", ...}
                               ...
                               {"type": "result", "points": [...], "colors": [...], "count": N}
                               {"type": "error", "message": "..."}   (on failure)

Run modes:
    - Real model (default):  uses depth_model.DepthEstimator (Depth Anything V2)
    - Fake/dev mode:         set USE_FAKE_DEPTH=1 to use the CPU-only
                              brightness-based placeholder instead.
    - Model size:            controlled by model_switch.py (or the
                              DEPTHWIZARD_MODEL_SIZE env var override).

Start the server:
    python server.py
    # then visit http://localhost:8000/docs for interactive API docs
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from pointcloud import image_and_depth_to_pointcloud
from shadow_height import estimate_height_from_shadow, HeightEstimate
from progress_events import ProgressReporter
from model_switch import get_active_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USE_FAKE_DEPTH = os.environ.get("USE_FAKE_DEPTH", "0") == "1"

_ACTIVE_MODEL_SIZE, _ACTIVE_MODEL_ID, _ACTIVE_MODEL_INFO = get_active_model()

# Max resolution we resize incoming images to before running depth
# estimation. Large photos make inference slower and point clouds heavier
# for no real visual benefit on a screen-sized 3D viewer.
MAX_IMAGE_DIM = 768

# ---------------------------------------------------------------------------
# App + model setup
# ---------------------------------------------------------------------------

app = FastAPI(title="DepthWizard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a hackathon demo; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# The depth estimator is expensive to load, so we load it ONCE at startup
# and reuse it for every request rather than reloading per-request.
_estimator = None


def _get_estimator(reporter: ProgressReporter = None):
    """Lazily creates the real DepthEstimator on first use (not at import
    time), so USE_FAKE_DEPTH mode never needs torch/transformers installed
    at all. Reused across all requests after the first."""
    global _estimator
    if _estimator is None:
        from depth_model import DepthEstimator
        from progress_events import NULL_REPORTER

        _estimator = DepthEstimator(reporter=reporter or NULL_REPORTER)
    return _estimator


def _resize_if_needed(image: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    """Downscales an image so its longest side is at most max_dim, keeping
    aspect ratio. Keeps inference fast and point clouds a reasonable size."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS)


def _run_pipeline(image: Image.Image, reporter: ProgressReporter) -> dict:
    """Shared core: resize -> depth -> pointcloud. Used by both the plain
    REST endpoint (no live events) and the WebSocket endpoint (live events)."""
    reporter.emit("resize", f"Preparing image (max dimension {MAX_IMAGE_DIM}px)...")
    image = _resize_if_needed(image)

    if USE_FAKE_DEPTH:
        from generate_sample_depth import fake_depth_from_brightness

        reporter.emit("mode", "Running in FAKE DEPTH mode (dev/testing) — no ML model used.")
        depth_map = fake_depth_from_brightness(image, reporter=reporter)
    else:
        estimator = _get_estimator(reporter=reporter)
        depth_map = estimator.predict(image, reporter=reporter)

    pointcloud = image_and_depth_to_pointcloud(
        image=image,
        depth_map=depth_map,
        max_points=150_000,
        depth_scale=40.0,
        reporter=reporter,
    )
    return pointcloud


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def health_check():
    """Simple endpoint to confirm the server is running + report active model."""
    return {
        "status": "ok",
        "mode": "fake_depth (dev/testing)" if USE_FAKE_DEPTH else f"real_model ({_ACTIVE_MODEL_SIZE})",
        "model_id": None if USE_FAKE_DEPTH else _ACTIVE_MODEL_ID,
        "model_info": None if USE_FAKE_DEPTH else _ACTIVE_MODEL_INFO,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Plain REST endpoint: takes an uploaded image, returns a 3D point cloud.
    No live progress events here — use WS /ws/predict for that.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    from progress_events import NULL_REPORTER
    pointcloud = await asyncio.to_thread(_run_pipeline, image, NULL_REPORTER)
    return pointcloud




@app.websocket("/ws/predict")
async def predict_ws(websocket: WebSocket):
    """
    Live-progress version of /predict. The frontend's terminal view connects
    here, sends one image, and receives a stream of progress events followed
    by a final "result" event containing the point cloud.
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)
        image_b64 = payload["image_b64"]
        if "," in image_b64 and image_b64.strip().startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"Bad request: {exc}"})
        await websocket.close()
        return

    # Events are produced on a background thread (since model inference is
    # CPU/GPU-bound, blocking code) — we bridge them back to the asyncio
    # event loop via call_soon_threadsafe so send_json() is safe to call.
    def on_event(event: dict):
        asyncio.run_coroutine_threadsafe(websocket.send_json(event), loop)

    reporter = ProgressReporter(callback=on_event)

    try:
        pointcloud = await asyncio.to_thread(_run_pipeline, image, reporter)
        await websocket.send_json({"type": "result", **pointcloud})
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


class HeightMeasureRequest(BaseModel):
    feature_top_px: tuple[float, float]
    shadow_tip_px: tuple[float, float]
    sun_elevation_deg: float
    pixel_resolution_m: float
    click_uncertainty_px: float = 3.0


@app.post("/measure_height")
async def measure_height(req: HeightMeasureRequest):
    """
    Shadow-length based real-world height estimate for a lunar feature.
    Takes two pixel coordinates the frontend user clicked (feature top,
    shadow tip) plus the image's own sun_elevation and pixel_resolution
    from its PDS4 label, and returns a physically-grounded height in
    meters with an honest uncertainty range.
    """
    result: HeightEstimate = estimate_height_from_shadow(
        feature_top_px=req.feature_top_px,
        shadow_tip_px=req.shadow_tip_px,
        sun_elevation_deg=req.sun_elevation_deg,
        pixel_resolution_m=req.pixel_resolution_m,
        click_uncertainty_px=req.click_uncertainty_px,
    )
    return {
        "height_m": result.height_m,
        "height_uncertainty_m": result.height_uncertainty_m,
        "shadow_length_px": result.shadow_length_px,
        "shadow_length_m": result.shadow_length_m,
        "sun_elevation_deg": result.sun_elevation_deg,
        "pixel_resolution_m": result.pixel_resolution_m,
        "is_reliable": result.is_reliable,
        "warning": result.warning,
    }

if __name__ == "__main__":
    import uvicorn

    print(f"[server] USE_FAKE_DEPTH={USE_FAKE_DEPTH}  MODEL_SIZE={_ACTIVE_MODEL_SIZE} (edit model_switch.py to change)")
    uvicorn.run(app, host="0.0.0.0", port=8000)