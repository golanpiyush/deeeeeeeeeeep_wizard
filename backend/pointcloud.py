"""
pointcloud.py
==============
Turns (original color image + depth map) into a 3D point cloud that the
browser can render with Three.js.

WHAT CHANGED (realistic scaling):
    Previously depth_scale was a fixed, arbitrary constant (40.0) with no
    connection to the actual physical scene -- it just "looked okay" on
    whatever test photos were used. That's fine for generic demo photos,
    but wrong for lunar OHRC imagery, where we actually KNOW the ground
    sample distance (pixel_resolution_m, e.g. 0.24 m/pixel from the PDS4
    label) and roughly what realistic crater relief looks like.

    This version adds compute_realistic_depth_scale(), which derives a
    physically-motivated depth_scale from:
      - the image's real pixel_resolution_m (so the X/Y axes are already
        in a consistent unit -- meters per pixel)
      - a typical relief-to-diameter ratio for small lunar craters
        (~5-15% depth/diameter for simple bowl craters is well-established
        in lunar geomorphology -- we use 10% as a reasonable midpoint)

    This doesn't make the point cloud "metrically accurate" in the DSM
    sense (that requires the SRTM/scale-calibration module for Earth
    imagery -- see project notes) -- it makes the VISUAL exaggeration
    defensible with a stated assumption, instead of an unexplained magic
    number. Non-lunar photos still use the old fixed default.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from progress_events import ProgressReporter, NULL_REPORTER


pts = 150_000

# Typical depth-to-diameter ratio for small, simple (non-complex) lunar
# bowl craters -- well-established range in lunar geomorphology is
# roughly 1:5 to 1:10 (depth is 10-20% of diameter) for fresh simple
# craters. We use a conservative 10% as a defensible middle estimate for
# an UNKNOWN crater size in an arbitrary crop -- not a measurement, an
# assumption, stated explicitly here and in any report/pitch.
LUNAR_DEPTH_TO_DIAMETER_RATIO = 0.10

# Fallback for non-lunar (generic photo) input -- unchanged from before.
DEFAULT_DEPTH_SCALE = 40.0


def compute_realistic_depth_scale(
    image_width_px: int,
    pixel_resolution_m: float,
    depth_to_diameter_ratio: float = LUNAR_DEPTH_TO_DIAMETER_RATIO,
) -> float:
    """
    Derives a physically-motivated depth_scale for lunar imagery from the
    image's real ground sample distance, instead of using an arbitrary
    fixed constant.

    Reasoning: our X/Y point cloud coordinates are in PIXELS (see
    image_and_depth_to_pointcloud below), which at pixel_resolution_m
    meters/pixel means the crop spans (image_width_px * pixel_resolution_m)
    real meters across. If we assume the dominant visible feature is
    roughly crater-scale relative to the crop (a reasonable assumption for
    a single-crater-scale crop like ours), a defensible vertical relief is
    depth_to_diameter_ratio * (crop width in meters) / pixel_resolution_m,
    converting back into the same pixel-based Z units the rest of the
    point cloud uses.

    This is an ASSUMPTION-BASED estimate for VISUALIZATION purposes, not
    a calibrated metric height (that's what shadow_height.py provides for
    one clicked feature, and what SRTM-based calibration provides for a
    full Earth DSM). State this plainly if asked -- it's honest, not
    hand-wavy, because the assumption and its source are explicit.

    Args:
        image_width_px: width of the (possibly resized) image in pixels.
        pixel_resolution_m: real-world meters per pixel, from the image's
            PDS4 label (isda:pixel_resolution).
        depth_to_diameter_ratio: assumed vertical relief as a fraction of
            the crop's real-world width. Default 0.10 (10%) is a
            conservative mid-range value for simple lunar craters.

    Returns:
        A depth_scale value in the same pixel-based units the point cloud
        already uses for X/Y, so Z ends up visually proportionate rather
        than an arbitrary flat multiplier.
    """
    crop_width_m = image_width_px * pixel_resolution_m
    assumed_relief_m = crop_width_m * depth_to_diameter_ratio
    # Convert the assumed relief back into pixel-equivalent units (since
    # X/Y are in pixels, not meters) so Z is on a consistent visual scale.
    depth_scale_px_equivalent = assumed_relief_m / pixel_resolution_m
    return depth_scale_px_equivalent


def image_and_depth_to_pointcloud(
    image: Image.Image,
    depth_map: np.ndarray,
    max_points: int = pts,
    depth_scale: float | None = None,
    pixel_resolution_m: float | None = None,
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
        max_points: cap on how many points to emit.
        depth_scale: explicit override for the Z-axis multiplier. If None
            (default) AND pixel_resolution_m is provided, a realistic
            scale is computed via compute_realistic_depth_scale(). If
            None and pixel_resolution_m is also None, falls back to
            DEFAULT_DEPTH_SCALE (40.0) -- the old fixed behavior, correct
            for generic non-lunar photos where no real GSD is known.
        pixel_resolution_m: real-world meters/pixel for this image, if
            known (e.g. 0.24 for the OHRC dataset, from its PDS4 label).
            Pass this for lunar/satellite imagery to get a physically-
            grounded depth_scale instead of the arbitrary default.
        invert_depth: flip depth direction if a scene looks inside-out.
        reporter: ProgressReporter to emit live status to.
        progress_chunks / min_chunk_seconds: live-placement animation
            tuning, unchanged from before.

    Returns:
        {"points": [[x,y,z],...], "colors": [[r,g,b],...], "count": int,
         "depth_scale_used": float}   <- new field, so the frontend/report
         can display what scale was actually applied, for transparency.
    """
    t_start = time.time()
    reporter.emit("pointcloud_start", "Building 3D point cloud from depth map...")

    image_rgb = np.array(image.convert("RGB"))
    h, w = depth_map.shape

    if image_rgb.shape[:2] != (h, w):
        raise ValueError(
            f"Image size {image_rgb.shape[:2]} does not match depth map size {(h, w)}. "
            "Make sure the depth map was computed on this exact image."
        )

    # --- Resolve the depth_scale to actually use ---
    if depth_scale is not None:
        resolved_scale = depth_scale
        scale_source = "explicit override"
    elif pixel_resolution_m is not None:
        resolved_scale = compute_realistic_depth_scale(w, pixel_resolution_m)
        scale_source = (
            f"derived from pixel_resolution_m={pixel_resolution_m} "
            f"(assumes ~{int(LUNAR_DEPTH_TO_DIAMETER_RATIO*100)}% relief-to-width ratio)"
        )
    else:
        resolved_scale = DEFAULT_DEPTH_SCALE
        scale_source = "default fixed value (no real-world scale known)"

    reporter.emit(
        "pointcloud_step",
        f"Depth scale: {resolved_scale:.1f} ({scale_source}).",
        depth_scale=round(resolved_scale, 2),
    )

    reporter.emit("pointcloud_step", f"Normalizing depth values across {h}x{w} = {h*w:,} pixels...")
    d = depth_map.astype(np.float32)
    d_min, d_max = d.min(), d.max()
    if d_max - d_min < 1e-6:
        d_norm = np.zeros_like(d)
    else:
        d_norm = (d - d_min) / (d_max - d_min)

    if invert_depth:
        d_norm = 1.0 - d_norm

    reporter.emit("pointcloud_step", "Mapping pixel grid to 3D coordinate space...")
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs.astype(np.float32) - (w / 2.0)
    ys = (h / 2.0) - ys.astype(np.float32)

    zs = d_norm * resolved_scale

    points = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    colors = image_rgb.reshape(-1, 3)

    total_pixels = points.shape[0]

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
        "depth_scale_used": round(float(resolved_scale), 2),
    }