"""
generate_sample_depth.py
==========================
A CPU-only, NO-DOWNLOAD, NO-TORCH-REQUIRED fake depth estimator.

Why this file exists:
    The real depth model (Depth Anything V2, via depth_model.py) needs
    torch + transformers installed and downloads ~100-400MB of weights on
    first run. That's fine on your laptop, but it means you can't quickly
    sanity-check the REST of the pipeline (point cloud generation, the
    FastAPI server, the Three.js viewer) without waiting on that setup.

    This module fakes a depth map using classical image processing
    (grayscale intensity + a bit of blur) so you can verify the full
    upload -> depth -> point cloud -> 3D viewer pipeline works end-to-end
    in seconds, before plugging in the real model.

    IMPORTANT: this fake depth is NOT a real 3D estimate — it just treats
    "brighter pixels" as "closer" as a crude placeholder. Swap to
    depth_model.DepthEstimator for actual results. This file is a
    development/testing tool, not part of the final demo.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def fake_depth_from_brightness(image: Image.Image) -> np.ndarray:
    """
    Produces a placeholder "depth map" using only PIL/numpy (no ML model,
    no GPU, no torch) so the rest of the pipeline can be tested immediately.

    Approach: convert to grayscale, blur slightly to remove noise/detail,
    and treat brightness as a rough proxy for depth. This is obviously not
    a real depth estimate — it's purely so we can move data through the
    pipeline before the real model is wired up.
    """
    gray = image.convert("L")  # grayscale
    # A blur smooths out fine detail/noise so the fake "depth" looks like
    # gentle rolling terrain instead of noisy static - just for a nicer
    # placeholder visual while testing.
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=4))
    depth_map = np.array(blurred).astype(np.float32)
    return depth_map


if __name__ == "__main__":
    # Quick manual test: generate a synthetic test image and fake-depth it,
    # so you can confirm this file works before wiring it into the server.
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Build a simple synthetic "terrain-like" test image: a radial gradient
    # (bright center fading to dark edges) plus some noise, so it has some
    # visual structure to turn into a point cloud.
    size = 256
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size / 2, size / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    gradient = 255 - (dist / dist.max() * 255)
    noise = np.random.default_rng(0).normal(0, 10, size=(size, size))
    synthetic = np.clip(gradient + noise, 0, 255).astype(np.uint8)

    # Make it a 3-channel "image" (fake RGB from the grayscale gradient)
    # so it round-trips correctly through the same code path as a real photo.
    rgb = np.stack([synthetic] * 3, axis=-1)
    test_image = Image.fromarray(rgb, mode="RGB")
    test_image.save(os.path.join(out_dir, "synthetic_test_input.png"))

    depth = fake_depth_from_brightness(test_image)
    depth_img = Image.fromarray(depth.astype(np.uint8))
    depth_img.save(os.path.join(out_dir, "synthetic_test_fake_depth.png"))

    print(f"Saved test input and fake depth map to: {out_dir}")
    print(f"Depth map shape: {depth.shape}, min={depth.min():.1f}, max={depth.max():.1f}")