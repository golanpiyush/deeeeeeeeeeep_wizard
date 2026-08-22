"""
depth_model.py
================
Wraps the Depth Anything V2 model so the rest of the app doesn't need to
know anything about Hugging Face / PyTorch internals.

Depth Anything V2 is a pretrained monocular depth estimation model — it was
trained on millions of images to predict "how far away is each pixel" from
a single 2D photo, with no need for stereo cameras or LIDAR. We use it
as-is (no training needed) via the `transformers` library's pipeline API.

Usage:
    from depth_model import DepthEstimator

    estimator = DepthEstimator(model_size="small")   # "small" | "base" | "large"
    depth_map = estimator.predict(pil_image)          # returns a numpy array
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# Hugging Face model IDs for each Depth Anything V2 size.
# "small" is recommended for a 6GB laptop GPU (see README).
_MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthEstimator:
    """
    Thin wrapper around a Depth Anything V2 checkpoint.

    Loading the model is slow (a few seconds to a minute, plus a one-time
    download), so this class is designed to be instantiated ONCE when the
    server starts, then reused for every request — not re-created per image.
    """

    def __init__(self, model_size: str = "small", device: str | None = None):
        if model_size not in _MODEL_IDS:
            raise ValueError(
                f"model_size must be one of {list(_MODEL_IDS)}, got {model_size!r}"
            )

        # Import torch/transformers lazily so this module can be imported
        # (e.g. for type-checking or by generate_sample_depth.py's CPU
        # fallback path) even in environments where torch isn't installed.
        import torch
        from transformers import pipeline

        if device is None:
            # Auto-detect: use the GPU if available (your 4060), else CPU.
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_size = model_size
        model_id = _MODEL_IDS[model_size]

        print(f"[DepthEstimator] Loading {model_id} on device={device} ...")
        # transformers' `pipeline` helper handles all the preprocessing/
        # postprocessing plumbing for us — we just call it like a function.
        self._pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if device == "cuda" else -1,  # 0 = first GPU, -1 = CPU
        )
        print("[DepthEstimator] Model loaded.")

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Run depth estimation on a single PIL image.

        Returns:
            A 2D numpy float32 array, same (H, W) as the input image, where
            each value represents the model's relative depth estimate at
            that pixel (larger = generally closer for Depth Anything V2 —
            this is a RELATIVE depth, not metric/absolute distance, which
            is expected and fine for our point-cloud visualization use case).
        """
        # Ensure 3-channel RGB — some uploaded images (e.g. lunar grayscale
        # imagery) may come in as single-channel or RGBA.
        if image.mode != "RGB":
            image = image.convert("RGB")

        result = self._pipe(image)
        # transformers' depth-estimation pipeline returns a dict with a
        # "predicted_depth" tensor and a convenience "depth" PIL image.
        # We want the raw numeric values, not the auto-normalized PIL image,
        # so we pull from predicted_depth and convert to numpy ourselves.
        depth_tensor = result["predicted_depth"]

        # predicted_depth may come back as (1, H, W) or (H, W) depending on
        # version — squeeze any batch dimension to be safe.
        depth_np = depth_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

        return depth_np


def normalize_depth_for_display(depth_map: np.ndarray) -> np.ndarray:
    """
    Scales a raw depth map to 0-255 uint8, purely for saving/viewing as a
    grayscale image (e.g. to sanity-check output visually). This is NOT
    used for the actual 3D point cloud math — pointcloud.py uses the raw
    float depth values for better precision.
    """
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min < 1e-6:
        # Avoid divide-by-zero on a completely flat (degenerate) depth map.
        return np.zeros_like(depth_map, dtype=np.uint8)

    normalized = (depth_map - d_min) / (d_max - d_min)  # scale to 0-1
    return (normalized * 255).astype(np.uint8)