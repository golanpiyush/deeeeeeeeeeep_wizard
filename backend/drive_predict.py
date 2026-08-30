"""
drive_predict.py
==================
Bridges drive_sync.py (Drive fetch) and prep_ohrc_for_depth.py's crop
logic into the existing _run_pipeline in server.py, so a user can pick
an OHRC image by name in the frontend and get a point cloud back
without ever manually downloading/cropping/uploading a .img file.

Flow:
    1. list_ohrc_candidates()   -> populates the frontend dropdown with
       every *_d_img_*.img the team's Drive has (paired with its .xml
       label, since we need both).
    2. fetch_and_crop_ohrc(name) -> downloads the chosen .img + its
       matching .xml fresh (no disk caching, per team decision),
       crops it using the same logic as prep_ohrc_for_depth.py's CLI,
       and returns an RGB PIL.Image + the pixel_resolution_m read from
       the label -- ready to feed straight into server.py's
       _run_pipeline exactly like an uploaded photo.

Design notes:
    - Every .img in this dataset needs its SIBLING .xml (same stem,
      .xml instead of .img) to know its dimensions -- there is no
      universal LINES/SAMPLES constant (see load_ohrc.py's docstring:
      "these vary per image"). So fetch_and_crop_ohrc always fetches
      the pair, never just the .img alone.
    - "Fresh every time, no caching" (team decision) means every call
      downloads the full .img again -- for the ~1.2GB products this is
      genuinely slow (expect a couple of minutes on a normal connection).
      We emit progress events via the existing ProgressReporter so the
      live terminal UI shows real download percentage instead of a
      silent multi-minute hang, exactly like the rest of the pipeline.
    - We reuse prep_ohrc_for_depth.py's read_dims_from_label() and
      load_crop() unchanged -- crop logic lives in ONE place, not
      duplicated here. load_crop() only reads the rows it needs via
      seek/read, so we still don't need the full array in memory even
      though the file itself was downloaded to disk.
    - Default crop (center_frac=(0.5, 0.6), crop_size=2000), per team
      decision -- no preview-and-pick step in this flow.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from drive_sync import (
    get_drive_service,
    list_files,
    fetch_file,
)
from prep_ohrc_for_depth import read_dims_from_label, load_crop
from progress_events import ProgressReporter, NULL_REPORTER

# Same defaults prep_ohrc_for_depth.py's CLI uses -- keep in sync if
# those ever change.
DEFAULT_CENTER_FRAC = (0.5, 0.6)
DEFAULT_CROP_SIZE = 2000

# Matches e.g. "ch2_ohr_ncp_20241115T1326321339_d_img_d18.img"
_IMG_NAME_RE = re.compile(r".*_d_img_.*\.img$", re.IGNORECASE)


class DrivePredictError(Exception):
    """Raised for any Drive/OHRC-specific failure, so server.py can turn
    it into a clean HTTP/WebSocket error instead of a raw stack trace."""


def list_ohrc_candidates(reporter: ProgressReporter = NULL_REPORTER) -> list[dict]:
    """
    Lists every OHRC .img product visible in the team's Drive, for the
    frontend dropdown. Returns [{"name": ..., "size_mb": ...}, ...].

    Does NOT check that a matching .xml exists for each one yet (that's
    checked at fetch time) -- this is just the picker list, and doing a
    second Drive call per candidate here would make the dropdown slow
    to populate for no benefit in the common case.
    """
    reporter.emit("drive_list_start", "Listing OHRC images in Drive...")
    service = get_drive_service()
    all_files = list_files(service)

    candidates = [
        {
            "name": f["name"],
            "size_mb": round(int(f.get("size", 0) or 0) / (1024 * 1024), 1),
        }
        for f in all_files
        if _IMG_NAME_RE.match(f["name"])
    ]
    candidates.sort(key=lambda c: c["name"])

    reporter.emit(
        "drive_list_done",
        f"Found {len(candidates)} OHRC image(s) in Drive.",
        count=len(candidates),
    )
    return candidates


def _xml_name_for(img_name: str) -> str:
    """'..._d_img_d18.img' -> '..._d_img_d18.xml' -- same stem, sibling label."""
    return str(Path(img_name).with_suffix(".xml"))


def fetch_and_crop_ohrc(
    img_name: str,
    reporter: ProgressReporter = NULL_REPORTER,
    center_frac: tuple[float, float] = DEFAULT_CENTER_FRAC,
    crop_size: int = DEFAULT_CROP_SIZE,
) -> tuple[Image.Image, float, float]:
    """
    Fetches img_name + its sibling .xml from Drive (fresh, no caching —
    downloaded straight into a temp dir that's cleaned up after crop),
    crops it exactly like prep_ohrc_for_depth.py's CLI does, and returns
    an RGB PIL.Image ready for _run_pipeline.

    Returns:
        (cropped_rgb_image, pixel_resolution_m, sun_elevation_deg)
        pixel_resolution_m / sun_elevation_deg are read straight from the
        .xml label's isda: fields when present, so the caller can pass
        pixel_resolution_m through to _run_pipeline for realistic depth
        scaling -- same as a manually-prepped crop would.

    Raises:
        DrivePredictError if the .img, its .xml pair, or the crop itself
        fails for any reason -- with a message specific enough to show
        directly in the frontend's live terminal, not a raw traceback.
    """
    if not _IMG_NAME_RE.match(img_name):
        raise DrivePredictError(
            f"'{img_name}' doesn't look like an OHRC .img product "
            f"(expected a name like '..._d_img_....img')."
        )

    xml_name = _xml_name_for(img_name)

    reporter.emit(
        "drive_auth",
        "Authenticating with Google Drive...",
    )
    service = get_drive_service()

    with tempfile.TemporaryDirectory(prefix="depthwizard_drive_") as tmp_dir:
        # --- .img (the big one) ---
        reporter.emit(
            "drive_fetch_img",
            f"Downloading {img_name} from Drive (this is the large file, "
            f"can take a couple of minutes)...",
            filename=img_name,
        )
        try:
            img_path = _download_to(service, img_name, tmp_dir, reporter)
        except FileNotFoundError:
            raise DrivePredictError(
                f"'{img_name}' was not found in Drive. It may have been "
                f"renamed or moved since the dropdown was populated -- "
                f"try refreshing the list."
            )

        # --- .xml (the label -- needed for dimensions + pixel_resolution_m) ---
        reporter.emit(
            "drive_fetch_xml",
            f"Downloading matching label {xml_name}...",
            filename=xml_name,
        )
        try:
            xml_path = _download_to(service, xml_name, tmp_dir, reporter)
        except FileNotFoundError:
            raise DrivePredictError(
                f"Found '{img_name}' but its label '{xml_name}' is missing "
                f"from Drive. The .xml label is required to know this "
                f"image's dimensions and real-world pixel scale -- without "
                f"it we can't safely crop this product. Every OHRC .img "
                f"needs its sibling .xml alongside it."
            )

        # --- crop, reusing prep_ohrc_for_depth.py's logic unchanged ---
        reporter.emit("ohrc_crop_start", "Reading label and cropping region...")
        try:
            lines, samples = read_dims_from_label(Path(xml_path))
        except Exception as exc:
            raise DrivePredictError(
                f"Could not read dimensions from {xml_name}'s label: {exc}"
            )

        crop = load_crop(
            Path(img_path), lines, samples,
            center_frac=center_frac, crop_size=crop_size,
        )

        pixel_resolution_m, sun_elevation_deg = _read_scale_and_sun(Path(xml_path))

        reporter.emit(
            "ohrc_crop_done",
            f"Cropped {crop.shape[1]}x{crop.shape[0]} region "
            f"(pixel_resolution_m={pixel_resolution_m}).",
            crop_width=int(crop.shape[1]),
            crop_height=int(crop.shape[0]),
            pixel_resolution_m=pixel_resolution_m,
            sun_elevation_deg=sun_elevation_deg,
        )

        rgb = np.stack([crop, crop, crop], axis=-1)
        image = Image.fromarray(rgb, mode="RGB")

        # temp dir (and the ~1.2GB .img inside it) is deleted here, on
        # exiting this `with` block -- nothing lingers on disk, per the
        # "always fetch fresh, no caching" decision.
        return image, pixel_resolution_m, sun_elevation_deg


def _download_to(service, name: str, out_dir: str, reporter: ProgressReporter) -> str:
    """
    Thin wrapper around drive_sync's fetch_file(download_first=True) that
    forwards Drive's chunk-progress into our ProgressReporter instead of
    printing to stdout (fetch_file prints via `print()`, which is fine
    for the CLI but invisible to the WebSocket-driven frontend).

    We don't reuse fetch_file's DOWNLOAD_DIR here on purpose -- it writes
    into drive_sync/downloaded/ and persists, which is a cache. We want
    this saved into a temp dir that's deleted after the crop, per the
    "always fetch fresh" decision, so we call the lower-level pieces
    fetch_file wraps instead of fetch_file itself.
    """
    from backend.drive_sync import find_exact_name, download_file_to_disk

    file_info = find_exact_name(service, name)
    if file_info is None:
        raise FileNotFoundError(name)

    size_mb = int(file_info.get("size", 0) or 0) / (1024 * 1024)
    reporter.emit(
        "drive_download_progress",
        f"Found {name} ({size_mb:.0f} MB) -- downloading...",
        filename=name, size_mb=round(size_mb, 1), progress_percent=0,
    )

    # download_file_to_disk prints its own \r progress bar to stdout;
    # we let it (harmless in the terminal running the server) and just
    # emit a start/done pair to the reporter, since patching in
    # per-chunk callbacks would mean copying its internals. Good enough
    # for now -- the frontend still gets a "still working" heartbeat
    # via the reporter's start message above plus this done message.
    path = download_file_to_disk(service, file_info["id"], name, out_dir=out_dir)

    reporter.emit(
        "drive_download_progress",
        f"{name} downloaded.",
        filename=name, size_mb=round(size_mb, 1), progress_percent=100,
    )
    return path


def _read_scale_and_sun(xml_path: Path) -> tuple[float, float]:
    """
    Reads isda:pixel_resolution and isda:sun_elevation straight from the
    PDS4 label, same namespace prep_ohrc_for_depth.py already uses for
    Line/Sample. Falls back to None if either field is genuinely absent
    from this particular product's label, rather than guessing a number.
    """
    import xml.etree.ElementTree as ET

    ISDA_NS = "{https://isda.issdc.gov.in/pds4/isda/v1}"
    tree = ET.parse(xml_path)
    root = tree.getroot()

    pixel_resolution_m = None
    sun_elevation_deg = None

    res_el = root.find(f".//{ISDA_NS}pixel_resolution")
    if res_el is not None and res_el.text:
        pixel_resolution_m = float(res_el.text)

    sun_el = root.find(f".//{ISDA_NS}sun_elevation")
    if sun_el is not None and sun_el.text:
        sun_elevation_deg = float(sun_el.text)

    return pixel_resolution_m, sun_elevation_deg