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

V2 CHANGES:
  - The actual math is UNCHANGED (same coordinate system, same downsampling
    strategy, same output shape) — this still produces identical point
    clouds to V1.
  - Point placement is now done in CHUNKS (not literally per-pixel — that
    would be tens of thousands of WebSocket messages and would actually be
    slower and would flood the frontend) so a ProgressReporter can emit
    real, meaningful progress events as the cloud is built. Chunk count is
    tuned to look genuinely "live" without spamming the connection.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from progress_events import ProgressReporter, NULL_REPORTER


pts = 150_000 # 60_000 view points

def image_and_depth_to_pointcloud(
    image: Image.Image,
    depth_map: np.ndarray,
    max_points: int = pts, 
    depth_scale: float = 40.0,
    invert_depth: bool = False,
    reporter: ProgressReporter = NULL_REPORTER,
    progress_chunks: int = 24,
    min_chunk_seconds: float = 0.05,
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
        invert_depth: flip depth direction if a scene looks inside-out.
        reporter: ProgressReporter to emit live status to (no-op if omitted).
        progress_chunks: how many progress events to emit while building
            the point cloud. Higher = smoother-looking live view, but more
            messages sent. 24 gives a nice steady terminal scroll without
            being spammy.
        min_chunk_seconds: real (wall-clock) pause inserted between each
            chunk event, in seconds. Without this, chunking is pure array
            slicing and all events fire within milliseconds — nothing
            visibly animates in the browser even though many events were
            sent. Default 0.05s (50ms) across ~24 chunks adds about 1.2s
            total to a request — a worthwhile trade for it actually
            looking live. Set to 0 to disable (fires instantly, old
            behavior).

    Returns:
        A JSON-serializable dict:
            {
                "points": [[x, y, z], ...],
                "colors": [[r, g, b], ...],
                "count": int
            }
    """
    t_start = time.time()
    reporter.emit("pointcloud_start", "Building 3D point cloud from depth map...")

    image_rgb = np.array(image.convert("RGB"))  # shape (H, W, 3), uint8
    h, w = depth_map.shape

    if image_rgb.shape[:2] != (h, w):
        raise ValueError(
            f"Image size {image_rgb.shape[:2]} does not match depth map size {(h, w)}. "
            "Make sure the depth map was computed on this exact image."
        )

    # --- Normalize depth to a sane 0-1 range before scaling ---
    reporter.emit("pointcloud_step", f"Normalizing depth values across {h}x{w} = {h*w:,} pixels...")
    d = depth_map.astype(np.float32)
    d_min, d_max = d.min(), d.max()
    if d_max - d_min < 1e-6:
        d_norm = np.zeros_like(d)
    else:
        d_norm = (d - d_min) / (d_max - d_min)

    if invert_depth:
        d_norm = 1.0 - d_norm

    # --- Build pixel coordinate grid ---
    reporter.emit("pointcloud_step", "Mapping pixel grid to 3D coordinate space...")
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs.astype(np.float32) - (w / 2.0)
    ys = (h / 2.0) - ys.astype(np.float32)  # flip y so "up" in image = up in 3D

    zs = d_norm * depth_scale

    points = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    colors = image_rgb.reshape(-1, 3)

    total_pixels = points.shape[0]

    # --- Downsample if there are more pixels than max_points ---
    if total_pixels > max_points:
        reporter.emit(
            "pointcloud_step",
            f"Downsampling {total_pixels:,} pixels -> {max_points:,} points "
            f"for smooth browser rendering...",
            total_pixels=total_pixels, max_points=max_points,
        )
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(total_pixels, size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    n = points.shape[0]

    # --- Emit chunked "live placement" progress events ---
    # IMPORTANT: this actually sends each chunk's real point/color data
    # (not just a log message) so the frontend can draw the point cloud
    # INCREMENTALLY as it builds — that's what makes this genuinely live,
    # rather than a text log that scrolls while the 3D view stays blank
    # until the very end.
    #
    # A tiny real sleep is inserted between chunks. Without it, this loop
    # is pure numpy array slicing and finishes in ~10 milliseconds total —
    # every event fires in the same instant and the browser has no chance
    # to paint between them, so nothing visibly animates even though the
    # log has many lines. min_chunk_seconds gives each chunk a real,
    # visible moment on screen. Tuned low enough to still feel fast for a
    # demo, high enough to actually see points appearing over time.
    chunk_size = max(1, n // progress_chunks)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk_points = points[i:end]
        chunk_colors = colors[i:end]
        sample_point = points[i]
        sample_color = colors[i]
        percent = round((end / n) * 100, 1)
        reporter.emit(
            "pointcloud_chunk",
            f"Placed points {i:,}-{end:,} of {n:,} "
            f"({percent}%) | sample @ ({sample_point[0]:.1f}, {sample_point[1]:.1f}, "
            f"{sample_point[2]:.1f}) rgb({sample_color[0]},{sample_color[1]},{sample_color[2]})",
            progress_percent=percent,
            points_done=end,
            points_total=n,
            sample_point=[round(float(c), 2) for c in sample_point],
            sample_color=[int(c) for c in sample_color],
            # The actual chunk data the frontend uses to draw incrementally:
            chunk_points=chunk_points.round(3).tolist(),
            chunk_colors=chunk_colors.tolist(),
        )
        if min_chunk_seconds > 0:
            time.sleep(min_chunk_seconds)

    elapsed = time.time() - t_start
    reporter.emit(
        "pointcloud_done",
        f"Point cloud complete: {n:,} points in {elapsed:.2f}s.",
        count=n, elapsed_seconds=round(elapsed, 3),
    )

    return {
        "points": points.round(3).tolist(),
        "colors": colors.tolist(),
        "count": int(n),
    }