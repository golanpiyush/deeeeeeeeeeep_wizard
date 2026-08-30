"""
lunar_preprocess.py
=====================
THE FIX for the smooth-slope failure mode: Depth Anything V2, run on a
raw OHRC crop, reads the sun-angle lighting gradient as "depth" instead
of picking up on actual crater shadows.

Two classical steps, run before the image reaches the AI model:
  1. GRADIENT FLATTENING -- removes the broad lighting trend via a
     high-pass filter (heavy blur, then subtract), preserving fine
     crater-shadow detail.
  2. CLAHE -- boosts LOCAL contrast per-tile so every crater's shadow is
     pronounced regardless of where it sits in the frame.

Every step emits a live ProgressReporter event so this is visible in the
terminal, not a hidden side-effect.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image

from progress_events import ProgressReporter, NULL_REPORTER


def _flatten_global_gradient(gray: np.ndarray, blur_fraction: float = 0.15) -> np.ndarray:
    h, w = gray.shape
    kernel_size = int(min(h, w) * blur_fraction)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, 3)

    broad_trend = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    gray_f = gray.astype(np.float32)
    trend_f = broad_trend.astype(np.float32)
    detail = gray_f - trend_f

    flattened = np.clip(detail + 128.0, 0, 255).astype(np.uint8)
    return flattened


def _apply_clahe(gray: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def looks_like_lunar_grayscale(image: Image.Image, std_threshold: float = 2.0) -> bool:
    """
    Auto-detector: R/G/B channels nearly identical everywhere means this
    is a grayscale image saved as RGB (like our OHRC crops) rather than
    a real color photo.
    """
    arr = np.array(image.convert("RGB")).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    channel_diff_std = float(np.std(r - g) + np.std(g - b))
    return channel_diff_std < std_threshold


def preprocess_lunar_image(
    image: Image.Image,
    blur_fraction: float = 0.15,
    clip_limit: float = 3.0,
    tile_size: int = 8,
    reporter: ProgressReporter = NULL_REPORTER,
) -> Image.Image:
    """
    Full lunar preprocessing pipeline: flatten sun-angle gradient, then
    enhance local crater contrast. Returns a new RGB PIL Image ready to
    feed into DepthEstimator.predict().
    """
    t0 = time.time()
    w, h = image.size

    reporter.emit(
        "lunar_preprocess_start",
        f"Lunar image detected -- applying sun-angle gradient correction ({w}x{h})...",
        width=w, height=h,
    )

    gray = np.array(image.convert("L"))

    reporter.emit(
        "lunar_preprocess_step",
        "Flattening sun-angle lighting gradient (removes false 'slope' signal)...",
    )
    flattened = _flatten_global_gradient(gray, blur_fraction=blur_fraction)

    reporter.emit(
        "lunar_preprocess_step",
        "Enhancing local crater shadow contrast (CLAHE)...",
    )
    enhanced = _apply_clahe(flattened, clip_limit=clip_limit, tile_size=tile_size)

    rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)
    result = Image.fromarray(rgb, mode="RGB")

    elapsed = time.time() - t0
    reporter.emit(
        "lunar_preprocess_done",
        f"Lunar preprocessing complete in {elapsed:.2f}s -- crater shadow detail "
        f"now exposed uniformly across the frame.",
        elapsed_seconds=round(elapsed, 3),
    )

    return result