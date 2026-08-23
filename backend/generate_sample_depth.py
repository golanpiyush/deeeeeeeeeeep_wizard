"""
generate_sample_depth.py
==========================
A CPU-only, NO-DOWNLOAD, NO-TORCH-REQUIRED fake depth estimator.

Why this file exists:
    The real depth model (Depth Anything V2, via depth_model.py) needs
    torch + transformers installed and downloads model weights on first
    run. That's fine on your laptop, but it means you can't quickly
    sanity-check the REST of the pipeline (point cloud generation, the
    FastAPI server, the Three.js viewer, the live terminal view) without
    waiting on that setup.

    This module fakes a depth map using classical image processing
    (grayscale intensity + a bit of blur) so you can verify the full
    upload -> depth -> point cloud -> 3D viewer pipeline works end-to-end
    in seconds, before plugging in the real model.

    IMPORTANT: this fake depth is NOT a real 3D estimate — it just treats
    "brighter pixels" as "closer" as a crude placeholder. Swap to
    depth_model.DepthEstimator for actual results. This file is a
    development/testing tool, not part of the final demo.

V2 CHANGES:
    - Now accepts a ProgressReporter so the live terminal view narrates
      fake-depth mode too (useful for testing the terminal UI itself
      without waiting on the real model to load).
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageFilter

from progress_events import ProgressReporter, NULL_REPORTER


def fake_depth_from_brightness(image: Image.Image, reporter: ProgressReporter = NULL_REPORTER) -> np.ndarray:
    """
    Produces a placeholder "depth map" using only PIL/numpy (no ML model,
    no GPU, no torch) so the rest of the pipeline can be tested immediately.

    Approach: convert to grayscale, blur slightly to remove noise/detail,
    and treat brightness as a rough proxy for depth. This is obviously not
    a real depth estimate — it's purely so we can move data through the
    pipeline before the real model is wired up.
    """
    w, h = image.size
    reporter.emit(
        "inference_start",
        f"[FAKE DEPTH MODE] Estimating depth for {w}x{h} image via brightness proxy...",
        width=w, height=h, total_pixels=w * h, device="cpu (no model)",
    )
    t0 = time.time()

    gray = image.convert("L")  # grayscale
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=4))
    depth_map = np.array(blurred).astype(np.float32)

    elapsed = time.time() - t0
    reporter.emit(
        "inference_done",
        f"[FAKE DEPTH MODE] Depth proxy computed in {elapsed:.3f}s "
        f"(range {depth_map.min():.0f} - {depth_map.max():.0f}).",
        inference_seconds=round(elapsed, 4),
        depth_min=float(depth_map.min()),
        depth_max=float(depth_map.max()),
    )
    return depth_map


if __name__ == "__main__":
    # Quick manual test: generate a synthetic test image and fake-depth it,
    # so you can confirm this file works before wiring it into the server.
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    size = 256
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size / 2, size / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    gradient = 255 - (dist / dist.max() * 255)
    noise = np.random.default_rng(0).normal(0, 10, size=(size, size))
    synthetic = np.clip(gradient + noise, 0, 255).astype(np.uint8)

    rgb = np.stack([synthetic] * 3, axis=-1)
    test_image = Image.fromarray(rgb, mode="RGB")
    test_image.save(os.path.join(out_dir, "synthetic_test_input.png"))

    depth = fake_depth_from_brightness(test_image)
    depth_img = Image.fromarray(depth.astype(np.uint8))
    depth_img.save(os.path.join(out_dir, "synthetic_test_fake_depth.png"))

    print(f"Saved test input and fake depth map to: {out_dir}")
    print(f"Depth map shape: {depth.shape}, min={depth.min():.1f}, max={depth.max():.1f}")