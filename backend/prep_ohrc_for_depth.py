"""
prep_ohrc_for_depth.py
========================
Your existing pipeline (depth_model.py / server.py) expects a normal
RGB photo. An OHRC .IMG is a single huge (101074 x 12000, in this case)
8-bit grayscale strip -- too big to run through Depth Anything V2 directly,
and not RGB.

This script:
    1. Loads the raw .IMG using dimensions read from its .xml label
       (same approach as load_ohrc.py / catalog_ohrc.py -- never hardcode
       dimensions, they differ per image).
    2. Crops out ONE interesting region (default: a square from roughly
       the middle of the frame, where your earlier preview showed the
       illuminated crater slope + shadow edge -- adjust CROP_CENTER_FRAC
       below once you've looked at the preview PNG).
    3. Converts grayscale -> RGB (just replicates the single channel
       across R, G, B -- Depth Anything V2 accepts this fine even though
       it's not "real" color).
    4. Saves a PNG you can feed straight into your existing WebSocket
       pipeline exactly like any other test photo (drag it into the
       DepthWizard frontend, or POST it to /predict).

Usage:
    python prep_ohrc_for_depth.py <path_to.img> <path_to.xml> [--out crop.png] [--crop-size 2000]

The --crop-size is in PIXELS OF THE ORIGINAL IMAGE (0.24 m/pixel for
this dataset -- check your own image's label), not the output size.
A 2000x2000 crop at 0.24 m/pixel covers a 480m x 480m patch of lunar
surface, which is a reasonable single-crater-scale test case.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

PDS_NS = "{http://pds.nasa.gov/pds4/pds/v1}"


def read_dims_from_label(xml_path: Path) -> tuple[int, int]:
    """Returns (lines, samples) read from the PDS4 label -- never hardcoded."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    array_el = root.find(f".//{PDS_NS}File_Area_Observational/{PDS_NS}Array_2D_Image")
    lines = samples = None
    for axis in array_el.findall(f"{PDS_NS}Axis_Array"):
        name = axis.find(f"{PDS_NS}axis_name").text
        n = int(axis.find(f"{PDS_NS}elements").text)
        if name == "Line":
            lines = n
        elif name == "Sample":
            samples = n
    if lines is None or samples is None:
        raise ValueError(f"Could not read Line/Sample dimensions from {xml_path}")
    return lines, samples


def load_crop(img_path: Path, lines: int, samples: int,
              center_frac=(0.5, 0.6), crop_size: int = 2000) -> np.ndarray:
    """
    Reads only the needed rows from disk (not the whole 1.2GB file) using
    seek/read, and crops a crop_size x crop_size square centered at
    (center_frac[0] * samples, center_frac[1] * lines).

    Default center_frac=(0.5, 0.6) targets roughly the middle-lower area
    of the frame -- based on the earlier preview showing the illuminated
    slope in that region. Adjust after checking your own preview PNG:
    center_frac is (fraction across width, fraction down height), each 0-1.
    """
    cx = int(center_frac[0] * samples)
    cy = int(center_frac[1] * lines)
    half = crop_size // 2

    row_start = max(0, cy - half)
    row_end = min(lines, cy + half)
    col_start = max(0, cx - half)
    col_end = min(samples, cx + half)

    n_rows = row_end - row_start
    bytes_per_row = samples  # 1 byte/pixel, UnsignedByte

    with open(img_path, "rb") as f:
        f.seek(row_start * bytes_per_row)
        raw = f.read(n_rows * bytes_per_row)

    full_rows = np.frombuffer(raw, dtype=np.uint8).reshape((n_rows, samples))
    crop = full_rows[:, col_start:col_end]
    return crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path")
    parser.add_argument("xml_path")
    parser.add_argument("--out", default="lunar_crop.png")
    parser.add_argument("--crop-size", type=int, default=2000)
    parser.add_argument("--center-x-frac", type=float, default=0.5,
                         help="0-1, fraction across the image width")
    parser.add_argument("--center-y-frac", type=float, default=0.6,
                         help="0-1, fraction down the image height")
    args = parser.parse_args()

    img_path = Path(args.img_path)
    xml_path = Path(args.xml_path)

    lines, samples = read_dims_from_label(xml_path)
    print(f"[prep_ohrc] Full image: {lines} lines x {samples} samples")

    crop = load_crop(
        img_path, lines, samples,
        center_frac=(args.center_x_frac, args.center_y_frac),
        crop_size=args.crop_size,
    )
    print(f"[prep_ohrc] Cropped region: {crop.shape}")
    print(f"[prep_ohrc] Crop pixel stats: mean={crop.mean():.1f} std={crop.std():.1f} "
          f"min={crop.min()} max={crop.max()}")

    rgb = np.stack([crop, crop, crop], axis=-1)  # grayscale -> RGB
    Image.fromarray(rgb, mode="RGB").save(args.out)
    print(f"[prep_ohrc] Saved -> {args.out}")
    print("[prep_ohrc] Drag this PNG into the DepthWizard frontend like any other photo.")

    if crop.mean() < 5:
        print("[prep_ohrc] WARNING: crop is mostly black -- try a different "
              "--center-x-frac / --center-y-frac (check the full preview PNG "
              "from load_ohrc.py / catalog_ohrc.py to pick a better spot).")


if __name__ == "__main__":
    main()