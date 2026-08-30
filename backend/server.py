"""
server.py
==========
FastAPI backend for DepthWizard.

THIS VERSION ACTUALLY WIRES IN (confirmed via testing, not assumed):
  1. Lunar preprocessing (lunar_preprocess.py) -- auto-detected, runs
     gradient-flatten + CLAHE before the image reaches the depth model.
     This is the fix for the smooth-slope failure on real OHRC imagery.
  2. Realistic depth scaling (pointcloud.py's compute_realistic_depth_scale)
     -- for lunar images, Z-axis exaggeration is now derived from the
     image's real pixel_resolution_m instead of an arbitrary constant.
     Pass pixel_resolution_m through the WebSocket payload to use this;
     omitting it falls back to the old fixed default (safe for generic
     photos where no real-world scale is known).

Endpoints:
    GET  /                Health check + active model info.
    POST /predict          Upload image -> point cloud JSON. No live events.
    WS   /ws/predict        Streams live progress events (model load, lunar
                             preprocessing steps, inference, point-cloud
                             chunks). Client payload:
                                 {
                                   "image_b64": "...",
                                   "lunar_mode": "auto" | "on" | "off",   (optional, default "auto")
                                   "pixel_resolution_m": 0.24              (optional -- pass for
                                                                             lunar imagery to get
                                                                             realistic depth scaling)
                                 }
    POST /measure_height    Shadow-length based real-world height estimate.

Start the server:
    python server.py
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

from drive_predict import DrivePredictError, fetch_and_crop_ohrc, list_ohrc_candidates
from pointcloud import image_and_depth_to_pointcloud
from shadow_height import estimate_height_from_shadow, HeightEstimate
from progress_events import ProgressReporter
from model_switch import get_active_model
from lunar_preprocess import preprocess_lunar_image, looks_like_lunar_grayscale

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USE_FAKE_DEPTH = os.environ.get("USE_FAKE_DEPTH", "0") == "1"

_ACTIVE_MODEL_SIZE, _ACTIVE_MODEL_ID, _ACTIVE_MODEL_INFO = get_active_model()

MAX_IMAGE_DIM = 768

# ---------------------------------------------------------------------------
# App + model setup
# ---------------------------------------------------------------------------

app = FastAPI(title="DepthWizard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_estimator = None


def _get_estimator(reporter: ProgressReporter = None):
    global _estimator
    if _estimator is None:
        from depth_model import DepthEstimator
        from progress_events import NULL_REPORTER

        _estimator = DepthEstimator(reporter=reporter or NULL_REPORTER)
    return _estimator


def _resize_if_needed(image: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS)


def _run_pipeline(
    image: Image.Image,
    reporter: ProgressReporter,
    lunar_mode: str = "auto",
    pixel_resolution_m: float | None = None,
) -> dict:
    """
    Shared core: resize -> [lunar preprocess] -> depth -> pointcloud.

    lunar_mode: "auto" (default) auto-detects via looks_like_lunar_grayscale();
        "on" forces preprocessing; "off" always skips it.
    pixel_resolution_m: if provided, used to compute a physically-grounded
        depth_scale for the point cloud (see pointcloud.py). Pass this for
        lunar/satellite imagery when you know the real ground sample
        distance (e.g. 0.24 for the OHRC dataset, from its PDS4 label).
    """
    reporter.emit("resize", f"Preparing image (max dimension {MAX_IMAGE_DIM}px)...")
    original_width = image.size[0]
    image = _resize_if_needed(image)

    # If the image was resized, pixel_resolution_m no longer matches the
    # image we're about to process -- scale it accordingly so the
    # real-world-meters-per-pixel figure stays correct after resizing.
    adjusted_pixel_resolution_m = pixel_resolution_m
    if pixel_resolution_m is not None and image.size[0] != original_width:
        resize_ratio = original_width / image.size[0]
        adjusted_pixel_resolution_m = pixel_resolution_m * resize_ratio
        reporter.emit(
            "resize",
            f"Adjusted pixel_resolution_m for resize: "
            f"{pixel_resolution_m} -> {adjusted_pixel_resolution_m:.4f} m/px",
        )

    # --- Lunar preprocessing (auto-detected by default) ---
    should_preprocess = (
        lunar_mode == "on"
        or (lunar_mode == "auto" and looks_like_lunar_grayscale(image))
    )
    if should_preprocess:
        image = preprocess_lunar_image(image, reporter=reporter)
    else:
        reporter.emit(
            "lunar_preprocess_skipped",
            "Image does not look like lunar/satellite grayscale imagery -- "
            "skipping sun-angle gradient correction.",
        )

    if USE_FAKE_DEPTH:
        from generate_sample_depth import fake_depth_from_brightness

        reporter.emit("mode", "Running in FAKE DEPTH mode (dev/testing) -- no ML model used.")
        depth_map = fake_depth_from_brightness(image, reporter=reporter)
    else:
        estimator = _get_estimator(reporter=reporter)
        depth_map = estimator.predict(image, reporter=reporter)

    pointcloud = image_and_depth_to_pointcloud(
        image=image,
        depth_map=depth_map,
        max_points=150_000,
        pixel_resolution_m=adjusted_pixel_resolution_m,
        reporter=reporter,
    )
    return pointcloud


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "mode": "fake_depth (dev/testing)" if USE_FAKE_DEPTH else f"real_model ({_ACTIVE_MODEL_SIZE})",
        "model_id": None if USE_FAKE_DEPTH else _ACTIVE_MODEL_ID,
        "model_info": None if USE_FAKE_DEPTH else _ACTIVE_MODEL_INFO,
        "lunar_preprocessing": "auto-detect (gradient flatten + CLAHE)",
        "realistic_depth_scaling": "enabled when pixel_resolution_m is provided",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
    Payload: {"image_b64": "...", "lunar_mode": "auto"|"on"|"off" (optional),
              "pixel_resolution_m": float (optional)}
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)
        image_b64 = payload["image_b64"]
        lunar_mode = payload.get("lunar_mode", "auto")
        pixel_resolution_m = payload.get("pixel_resolution_m")
        if "," in image_b64 and image_b64.strip().startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"Bad request: {exc}"})
        await websocket.close()
        return

    def on_event(event: dict):
        asyncio.run_coroutine_threadsafe(websocket.send_json(event), loop)

    reporter = ProgressReporter(callback=on_event)

    try:
        pointcloud = await asyncio.to_thread(
            _run_pipeline, image, reporter, lunar_mode, pixel_resolution_m
        )
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



@app.get("/drive/ohrc_list")
async def drive_ohrc_list():
    """Lists OHRC .img products available in the team's Drive, for the
    frontend's file-picker dropdown."""
    try:
        candidates = await asyncio.to_thread(list_ohrc_candidates)
        return {"candidates": candidates}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not list Drive files: {exc}")



@app.websocket("/ws/predict_from_drive")
async def predict_from_drive_ws(websocket: WebSocket):
    """
    Payload: {"img_name": "ch2_ohr_..._d_img_d18.img",
              "lunar_mode": "auto"|"on"|"off" (optional, default "auto")}

    Fetches the named .img + its sibling .xml from Drive, crops it,
    then runs it through the SAME _run_pipeline as a manual upload --
    same lunar preprocessing, same depth model, same point cloud logic.
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)
        img_name = payload["img_name"]
        lunar_mode = payload.get("lunar_mode", "auto")
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"Bad request: {exc}"})
        await websocket.close()
        return

    def on_event(event: dict):
        asyncio.run_coroutine_threadsafe(websocket.send_json(event), loop)

    reporter = ProgressReporter(callback=on_event)

    try:
        image, pixel_resolution_m, sun_elevation_deg = await asyncio.to_thread(
            fetch_and_crop_ohrc, img_name, reporter
        )
        if pixel_resolution_m is None:
            reporter.emit(
                "drive_scale_missing",
                "No pixel_resolution found in this image's label -- "
                "falling back to default depth scale (not physically "
                "grounded for this product).",
            )
        pointcloud = await asyncio.to_thread(
            _run_pipeline, image, reporter, lunar_mode, pixel_resolution_m
        )
        if sun_elevation_deg is not None:
            pointcloud["sun_elevation_deg"] = sun_elevation_deg
        await websocket.send_json({"type": "result", **pointcloud})
    except DrivePredictError as exc:
        # Known, explainable failure -- send the specific message, not a
        # stack trace (e.g. "missing .xml pair", "not found in Drive").
        await websocket.send_json({"type": "error", "message": str(exc)})
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"Unexpected error: {exc}"})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

@app.post("/measure_height")
async def measure_height(req: HeightMeasureRequest):
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
    print(f"[server] Lunar preprocessing: auto-detect (gradient flatten + CLAHE)")
    print(f"[server] Realistic depth scaling: enabled when pixel_resolution_m is passed in the WS payload")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# import sys; print(sys.path[0]); import os; print(os.listdir('backend/drive_sync'))