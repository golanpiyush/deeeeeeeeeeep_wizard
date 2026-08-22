"""
pointcloud.py
==============
Turns (original color image + depth map) into a 3D point cloud that the
browser can render with Three.js.

The core idea, in plain terms:
  - Every pixel in the image has an (x, y) position on the flat photo.
  - The depth model tells us how "deep" (far/near) that pixel should be —
    call that z.
  - So we place a point at 3D coordinates (x, y, z), colored with that
    pixel's original RGB color.
  - Do this for every pixel (or a downsampled subset, for performance) and
    you get a "3D photo" — a point cloud — that can be rotated and flown
    through in a 3D viewer.

We downsample heavily (see `max_points`) because:
  - A single photo can have millions of pixels -> millions of points is too
    much data to send over HTTP and too much for a browser to render smoothly.
  - A few tens of thousands of points already looks great and stays responsive
    even on modest hardware (important since judges' laptops vary).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def image_and_depth_to_pointcloud(
    image: Image.Image,
    depth_map: np.ndarray,
    max_points: int = 60_000,
    depth_scale: float = 40.0,
    invert_depth: bool = False,
) -> dict:
    """
    Build a point cloud from a color image and its matching depth map.

    Args:
        image: original PIL image (RGB), same size the depth map was computed on.
        depth_map: 2D numpy array of relative depth values, shape (H, W).
        max_points: cap on how many points to emit (downsamples if the image
            has more pixels than this, to keep the payload small and the
            browser fast).
        depth_scale: multiplier controlling how "deep" the 3D effect looks.
            Larger = more dramatic depth exaggeration. Tune this per-image;
            there's no single correct value since depth is relative, not
            metric.
        invert_depth: some depth conventions treat higher values as "closer"
            and others as "farther" — flip this if your scene looks
            inside-out (like a photo negative in 3D) when first rendered.

    Returns:
        A JSON-serializable dict:
            {
                "points": [[x, y, z], ...],       # 3D positions
                "colors": [[r, g, b], ...],        # 0-255 ints, same order as points
                "count": int
            }
        Ready to be sent straight to the frontend as JSON and consumed by
        Three.js's BufferGeometry.
    """
    image_rgb = np.array(image.convert("RGB"))  # shape (H, W, 3), uint8
    h, w = depth_map.shape

    # Sanity check: the depth map and image must line up pixel-for-pixel.
    if image_rgb.shape[:2] != (h, w):
        raise ValueError(
            f"Image size {image_rgb.shape[:2]} does not match depth map size {(h, w)}. "
            "Make sure the depth map was computed on this exact image."
        )

    # --- Normalize depth to a sane 0-1 range before scaling ---
    d = depth_map.astype(np.float32)
    d_min, d_max = d.min(), d.max()
    if d_max - d_min < 1e-6:
        d_norm = np.zeros_like(d)
    else:
        d_norm = (d - d_min) / (d_max - d_min)

    if invert_depth:
        d_norm = 1.0 - d_norm

    # --- Build pixel coordinate grid ---
    # We center x/y around 0 so the point cloud is centered at the origin,
    # which makes camera framing in Three.js much easier.
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs.astype(np.float32) - (w / 2.0)
    ys = (h / 2.0) - ys.astype(np.float32)  # flip y so "up" in image = up in 3D

    zs = d_norm * depth_scale

    # --- Flatten everything into (N, 3) points and (N, 3) colors ---
    points = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    colors = image_rgb.reshape(-1, 3)

    total_pixels = points.shape[0]

    # --- Downsample if there are more pixels than max_points ---
    if total_pixels > max_points:
        # Random uniform sampling keeps the overall shape/density of the
        # scene intact while cutting point count down to something the
        # browser can render at interactive frame rates.
        rng = np.random.default_rng(seed=42)  # fixed seed = reproducible demo
        idx = rng.choice(total_pixels, size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    return {
        "points": points.round(3).tolist(),
        "colors": colors.tolist(),
        "count": int(points.shape[0]),
    }