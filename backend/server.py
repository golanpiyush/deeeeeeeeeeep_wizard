"""
server.py
==========
FastAPI backend for DepthWizard.

Exposes a single main endpoint:

    POST /predict
        Body: an uploaded image file (multipart/form-data)
        Returns: JSON point cloud { points, colors, count }
                 ready to be rendered by the Three.js frontend.

Run modes:
    - Real model (default):  uses depth_model.DepthEstimator (Depth Anything V2)
    - Fake/dev mode:         set USE_FAKE_DEPTH=1 to use the CPU-only
                              brightness-based placeholder instead, useful
                              for quickly testing the server/frontend without
                              waiting on model download or a GPU.

Start the server:
    python server.py
    # then visit http://localhost:8000/docs for interactive API docs
"""

from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from pointcloud import image_and_depth_to_pointcloud

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Toggle this via environment variable if you want to run without the real
# model (e.g. USE_FAKE_DEPTH=1 python server.py) — handy for fast iteration
# on the frontend/server plumbing before the real model is set up.
USE_FAKE_DEPTH = os.environ.get("USE_FAKE_DEPTH", "0") == "1"

# Which Depth Anything V2 checkpoint to use when USE_FAKE_DEPTH is off.
# "small" is recommended for a 6GB laptop GPU — see README for details.
MODEL_SIZE = os.environ.get("DEPTHWIZARD_MODEL_SIZE", "small")

# Max resolution we resize incoming images to before running depth
# estimation. Large photos make inference slower and point clouds heavier
# for no real visual benefit on a screen-sized 3D viewer.
MAX_IMAGE_DIM = 768

# ---------------------------------------------------------------------------
# App + model setup
# ---------------------------------------------------------------------------

app = FastAPI(title="DepthWizard API")

# Allow the frontend (served from a different origin, e.g. a plain
# `python -m http.server`) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a hackathon demo; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# The depth estimator is expensive to load, so we load it ONCE at startup
# and reuse it for every request rather than reloading per-request.
_estimator = None


def _get_estimator():
    """Lazily creates the real DepthEstimator on first use (not at import
    time), so USE_FAKE_DEPTH mode never needs torch/transformers installed
    at all."""
    global _estimator
    if _estimator is None:
        from depth_model import DepthEstimator

        _estimator = DepthEstimator(model_size=MODEL_SIZE)
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def health_check():
    """Simple endpoint to confirm the server is running."""
    return {
        "status": "ok",
        "mode": "fake_depth (dev/testing)" if USE_FAKE_DEPTH else f"real_model ({MODEL_SIZE})",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main endpoint: takes an uploaded image, returns a 3D point cloud.

    Frontend usage (see frontend/index.html):
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        const res = await fetch("http://localhost:8000/predict", {
            method: "POST",
            body: formData,
        });
        const pointcloud = await res.json();
    """
    # --- Validate upload ---
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    image = _resize_if_needed(image)

    # --- Run depth estimation (real model or fake dev fallback) ---
    if USE_FAKE_DEPTH:
        from generate_sample_depth import fake_depth_from_brightness

        depth_map = fake_depth_from_brightness(image)
    else:
        estimator = _get_estimator()
        depth_map = estimator.predict(image)

    # --- Convert to point cloud JSON ---
    pointcloud = image_and_depth_to_pointcloud(
        image=image,
        depth_map=depth_map,
        max_points=60_000,
        depth_scale=40.0,
    )

    return pointcloud


if __name__ == "__main__":
    import uvicorn

    print(f"[server] USE_FAKE_DEPTH={USE_FAKE_DEPTH}  MODEL_SIZE={MODEL_SIZE}")
    uvicorn.run(app, host="0.0.0.0", port=8000)