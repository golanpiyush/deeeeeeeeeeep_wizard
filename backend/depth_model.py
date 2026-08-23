"""
depth_model.py
================
Wraps the Depth Anything V2 model so the rest of the app doesn't need to
know anything about Hugging Face / PyTorch internals.

Depth Anything V2 is a pretrained monocular depth estimation model — it was
trained on millions of images to predict "how far away is each pixel" from
a single 2D photo, with no need for stereo cameras or LIDAR. We use it
as-is (no training needed) via the `transformers` library's pipeline API.

V2 CHANGES:
    - Model size (small/base/large) now comes from model_switch.py instead
      of being hardcoded — see that file to change it.
    - Every meaningful step (loading, inference start/end) now emits a
      ProgressReporter event, so the live terminal view in the frontend
      can narrate exactly what's happening, in real time.

Usage:
    from depth_model import DepthEstimator
    from progress_events import ProgressReporter

    estimator = DepthEstimator()               # size comes from model_switch.py
    depth_map = estimator.predict(pil_image, reporter=ProgressReporter(cb))
"""

from __future__ import annotations

import os
import time

import numpy as np
from PIL import Image

from model_switch import get_active_model
from progress_events import ProgressReporter, NULL_REPORTER


class DepthEstimator:
    """
    Thin wrapper around a Depth Anything V2 checkpoint.

    Loading the model is slow (a few seconds to a minute, plus a one-time
    download), so this class is designed to be instantiated ONCE when the
    server starts, then reused for every request — not re-created per image.
    """

    def __init__(self, model_size: str | None = None, device: str | None = None,
                 reporter: ProgressReporter = NULL_REPORTER):
        # If no explicit size is passed, use whatever model_switch.py says
        # is active — this is the normal path for server.py.
        if model_size is None:
            model_size, model_id, info = get_active_model()
        else:
            from model_switch import MODEL_IDS, MODEL_INFO
            if model_size not in MODEL_IDS:
                raise ValueError(f"model_size must be one of {list(MODEL_IDS)}, got {model_size!r}")
            model_id, info = MODEL_IDS[model_size], MODEL_INFO[model_size]

        # Import torch/transformers lazily so this module can be imported
        # (e.g. for type-checking or by generate_sample_depth.py's CPU
        # fallback path) even in environments where torch isn't installed.
        reporter.emit("model_load", f"Importing torch + transformers...")
        import torch
        from transformers import pipeline

        if device is None:
            cuda_available = torch.cuda.is_available()
            force_gpu = os.environ.get("DEPTHWIZARD_FORCE_GPU", "0") == "1"
            if force_gpu and not cuda_available:
                raise RuntimeError(
                    "DEPTHWIZARD_FORCE_GPU=1 is set, but torch.cuda.is_available() "
                    "returned False — PyTorch cannot see a GPU. This usually means "
                    "the CPU-only build of torch got installed instead of the CUDA "
                    "build. Fix on the machine running this:\n"
                    "    pip uninstall torch torchvision -y\n"
                    "    pip install torch==2.7.1 torchvision==0.22.1 "
                    "--index-url https://download.pytorch.org/whl/cu128\n"
                    "Then confirm with: python -c \"import torch; "
                    "print(torch.__version__, torch.cuda.is_available())\""
                )
            device = "cuda" if cuda_available else "cpu"
            if not cuda_available:
                reporter.emit(
                    "gpu_warning",
                    "No CUDA GPU detected by torch — falling back to CPU "
                    "(this will be much slower, especially for 'large'). "
                    "If you have an NVIDIA GPU, your torch install is probably "
                    "CPU-only; see README for the fix.",
                )
            else:
                gpu_name = torch.cuda.get_device_name(0)
                reporter.emit(
                    "gpu_detected",
                    f"CUDA GPU detected: {gpu_name} — using GPU acceleration.",
                    gpu_name=gpu_name,
                )

        self.device = device
        self.model_size = model_size
        self.model_id = model_id

        reporter.emit(
            "model_load",
            f"Loading Depth Anything V2 ({model_size}) on {device}...",
            model_size=model_size,
            model_id=model_id,
            device=device,
            download_estimate=info["download_mb"],
        )
        t0 = time.time()
        self._pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if device == "cuda" else -1,  # 0 = first GPU, -1 = CPU
        )
        load_seconds = time.time() - t0
        reporter.emit(
            "model_ready",
            f"Model loaded in {load_seconds:.1f}s — ready for inference.",
            load_seconds=round(load_seconds, 2),
        )

    def predict(self, image: Image.Image, reporter: ProgressReporter = NULL_REPORTER) -> np.ndarray:
        """
        Run depth estimation on a single PIL image.

        Returns:
            A 2D numpy float32 array, same (H, W) as the input image, where
            each value represents the model's relative depth estimate at
            that pixel (larger = generally closer for Depth Anything V2 —
            this is a RELATIVE depth, not metric/absolute distance, which
            is expected and fine for our point-cloud visualization use case).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        reporter.emit(
            "inference_start",
            f"Running {self.model_size} model on {w}x{h} image ({w*h:,} pixels)...",
            width=w, height=h, total_pixels=w * h, device=self.device,
        )

        t0 = time.time()
        result = self._pipe(image)
        inference_seconds = time.time() - t0

        depth_tensor = result["predicted_depth"]
        depth_np = depth_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

        reporter.emit(
            "inference_done",
            f"Depth map computed in {inference_seconds:.2f}s "
            f"(range {depth_np.min():.2f} - {depth_np.max():.2f}).",
            inference_seconds=round(inference_seconds, 3),
            depth_min=float(depth_np.min()),
            depth_max=float(depth_np.max()),
        )

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
        return np.zeros_like(depth_map, dtype=np.uint8)

    normalized = (depth_map - d_min) / (d_max - d_min)
    return (normalized * 255).astype(np.uint8)