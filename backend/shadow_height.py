"""
shadow_height.py
==================
Estimates real-world height (in meters) of lunar terrain features from a
single image, using classical shadow photogrammetry -- no ML, no metric
depth calibration needed. This is the scientific differentiator for the
lunar mode of DepthWizard: everything else in the app produces a visually
convincing but scientifically UNVALIDATED relative-depth point cloud;
this module produces one real, defensible, physically-grounded number.

The physics (genuinely simple, and that's a feature -- easy to defend to
judges):

    A vertical feature of height H, lit by the sun at elevation angle
    theta above the horizon, casts a shadow of length L across flat
    ground, where:

        L = H / tan(theta)   ->   H = L * tan(theta)

    We measure L directly from the image: the user (or an automated
    edge-detector, later) clicks two points -- one at the top of the
    feature (e.g. a crater rim crest), one at the tip of its shadow.
    The pixel distance between those two points, multiplied by the
    image's ground sample distance (GSD, meters/pixel), gives L in
    real meters. theta comes straight from the image's PDS4 label
    (sun_elevation).

Known limitations (be upfront about these -- it's what makes this
defensible rather than oversold):
    - Assumes the shadow falls on locally flat ground. Sloped terrain
      near the shadow introduces error this model doesn't correct for.
    - Assumes the two clicked points are genuinely "top of feature" and
      "tip of shadow" -- user click precision directly limits accuracy.
      We propagate a configurable pixel-uncertainty into the final
      answer rather than pretending the measurement is exact.
    - Breaks down as sun_elevation approaches 0 deg: tan(theta) -> 0,
      so tiny shadow-length errors blow up into huge height errors.
      We flag this explicitly rather than silently returning a
      falsely-precise number.
    - This gives you the height of ONE feature you pick, not a full
      terrain reconstruction. That's an intentional, honest scope --
      full DEM reconstruction is a much larger problem (see project notes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Below this solar elevation angle, tan(theta) amplifies pixel-measurement
# error so much that the result is not meaningfully trustworthy. Flag it
# rather than returning a falsely precise number.
UNSTABLE_SUN_ELEVATION_DEG = 2.0

# Default assumed click imprecision, in pixels. A real user clicking on a
# screen is unlikely to land on the exact correct pixel -- this is a
# starting estimate, not a measured value. Expose it as a parameter so it
# can be tuned/justified in the report rather than hidden as a magic number.
DEFAULT_CLICK_UNCERTAINTY_PX = 3.0


@dataclass
class HeightEstimate:
    height_m: float
    height_uncertainty_m: float
    shadow_length_px: float
    shadow_length_m: float
    sun_elevation_deg: float
    pixel_resolution_m: float
    is_reliable: bool
    warning: str | None


def pixel_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two (x, y) pixel coordinates."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def estimate_height_from_shadow(
    feature_top_px: tuple[float, float],
    shadow_tip_px: tuple[float, float],
    sun_elevation_deg: float,
    pixel_resolution_m: float,
    click_uncertainty_px: float = DEFAULT_CLICK_UNCERTAINTY_PX,
) -> HeightEstimate:
    """
    Args:
        feature_top_px: (x, y) pixel coords of the top of the feature
            (e.g. crater rim crest), as clicked by the user on the
            DISPLAYED image (same pixel space the shadow_tip_px is in).
        shadow_tip_px: (x, y) pixel coords of the tip of the feature's
            cast shadow.
        sun_elevation_deg: sun elevation angle above the horizon, in
            degrees, from the image's PDS4 label (isda:sun_elevation).
            Must be > 0 for a meaningful (non-degenerate) result --
            negative or zero means the sun is at/below the horizon in
            this geometry, which this simple model can't handle.
        pixel_resolution_m: ground sample distance in meters/pixel,
            from the image's PDS4 label (isda:pixel_resolution).
        click_uncertainty_px: assumed +/- pixel imprecision in each
            click, used to propagate a real uncertainty range onto the
            final height rather than reporting a falsely exact number.

    Returns:
        HeightEstimate with the computed height, an uncertainty range,
        and an honesty flag (is_reliable / warning) for cases where the
        sun geometry makes this measurement inherently unstable.
    """
    shadow_length_px = pixel_distance(feature_top_px, shadow_tip_px)
    shadow_length_m = shadow_length_px * pixel_resolution_m

    theta_deg = sun_elevation_deg
    warning = None
    is_reliable = True

    if theta_deg <= 0:
        # Sun at or below horizon in this simple flat-ground model --
        # tan() is zero or negative, the formula is not physically
        # meaningful here. Don't return a fabricated number.
        return HeightEstimate(
            height_m=float("nan"),
            height_uncertainty_m=float("nan"),
            shadow_length_px=shadow_length_px,
            shadow_length_m=shadow_length_m,
            sun_elevation_deg=theta_deg,
            pixel_resolution_m=pixel_resolution_m,
            is_reliable=False,
            warning=(
                f"sun_elevation_deg={theta_deg:.3f} is at/below the horizon -- "
                "the flat-ground shadow model breaks down here. This image's "
                "illumination geometry is unsuitable for height measurement; "
                "pick an image with sun_elevation_deg > ~2 (see catalog.csv)."
            ),
        )

    theta_rad = math.radians(theta_deg)
    height_m = shadow_length_m * math.tan(theta_rad)

    if theta_deg < UNSTABLE_SUN_ELEVATION_DEG:
        is_reliable = False
        warning = (
            f"sun_elevation_deg={theta_deg:.3f} is very low (grazing light). "
            "tan(theta) is small, so this height estimate is highly sensitive "
            "to small errors in where you clicked -- treat this number as "
            "qualitative, not a precise measurement. Prefer an image with "
            "sun_elevation_deg > 2-5 deg for a defensible result."
        )

    # --- Propagate click-pixel uncertainty into a height uncertainty ---
    # If the true shadow length could be off by +/- click_uncertainty_px
    # in EACH click (so up to ~sqrt(2)*click_uncertainty_px combined),
    # see how much that swings the final height. This is a simple
    # first-order sensitivity estimate, not a rigorous statistical
    # propagation -- good enough to report an honest +/- range.
    combined_px_uncertainty = math.sqrt(2) * click_uncertainty_px
    shadow_length_m_high = (shadow_length_px + combined_px_uncertainty) * pixel_resolution_m
    shadow_length_m_low = max(0.0, (shadow_length_px - combined_px_uncertainty) * pixel_resolution_m)

    height_high = shadow_length_m_high * math.tan(theta_rad)
    height_low = shadow_length_m_low * math.tan(theta_rad)
    height_uncertainty_m = (height_high - height_low) / 2.0

    return HeightEstimate(
        height_m=round(height_m, 2),
        height_uncertainty_m=round(height_uncertainty_m, 2),
        shadow_length_px=round(shadow_length_px, 2),
        shadow_length_m=round(shadow_length_m, 2),
        sun_elevation_deg=theta_deg,
        pixel_resolution_m=pixel_resolution_m,
        is_reliable=is_reliable,
        warning=warning,
    )


if __name__ == "__main__":
    # Quick manual sanity check using made-up but plausible numbers.
    # Example: a 500m-tall crater rim casting a shadow across flat ground,
    # imaged at 0.24 m/pixel with the sun 10 deg above the horizon.
    #
    # Expected shadow length for a 500m rim at 10 deg sun elevation:
    #   L = H / tan(theta) = 500 / tan(10 deg) ~= 2835 m ~= 11,813 px at 0.24 m/px
    result = estimate_height_from_shadow(
        feature_top_px=(1000.0, 1000.0),
        shadow_tip_px=(1000.0 + 11813.0, 1000.0),  # 11,813 px east of the rim
        sun_elevation_deg=10.0,
        pixel_resolution_m=0.24,
    )
    print("Sanity check (expect height_m close to 500):")
    print(result)