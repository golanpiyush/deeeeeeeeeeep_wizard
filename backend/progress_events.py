"""
progress_events.py
====================
A tiny pub/sub system so that deep, slow-running code (depth_model.py,
pointcloud.py) can announce "here's what I'm doing right now" without
needing to know anything about WebSockets, FastAPI, or the frontend.

Design:
    - Each in-flight request gets its own ProgressReporter instance.
    - The reporter just calls `.emit(event_dict)` whenever something
      interesting happens (loading model, running inference, placing
      point N of M, etc).
    - server.py hands each reporter a callback that pushes those events
      onto a per-connection asyncio.Queue, which the WebSocket endpoint
      drains and forwards to the browser in real time.

This keeps pointcloud.py and depth_model.py free of any web-framework
knowledge — they just call reporter.emit(...), same as calling print().
"""

from __future__ import annotations

import time
from typing import Callable, Optional


class ProgressReporter:
    """
    Pass one of these into depth estimation / point cloud generation calls.
    If `callback` is None, all emit() calls are silently no-ops — so the
    same code path works whether or not anyone is listening (e.g. calling
    these functions from a script or test with no WebSocket attached).
    """

    def __init__(self, callback: Optional[Callable[[dict], None]] = None):
        self._callback = callback
        self._start_time = time.time()

    def emit(self, event_type: str, message: str, **extra):
        """
        event_type: short machine-readable tag, e.g. "model_load",
                    "inference", "pointcloud_chunk", "done", "error".
        message:    human-readable line for the terminal view.
        extra:      any additional JSON-serializable fields (progress
                    percent, chunk index, tensor shape, timing, etc).
        """
        if self._callback is None:
            return
        event = {
            "type": event_type,
            "message": message,
            "elapsed_ms": int((time.time() - self._start_time) * 1000),
            **extra,
        }
        try:
            self._callback(event)
        except Exception:
            # A broken frontend connection should NEVER break the actual
            # depth/pointcloud computation — progress reporting is best-effort.
            pass


# A reporter that does nothing — used as a safe default anywhere a
# ProgressReporter is expected but the caller didn't provide one.
NULL_REPORTER = ProgressReporter(callback=None)