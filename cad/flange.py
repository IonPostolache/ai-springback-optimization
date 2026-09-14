"""
Parametric sheet-metal flange geometry (build123d).

Geometry:
- Base leg: 30 mm
- Flange leg: 50 mm
- Thickness: 1.2 mm
- Width: 30 mm
- Bend angle: 90 deg
- Variable: r_bend = INNER bend radius

The profile is generated from a filleted centerline and a constant
thickness offset. This is the standard build123d pattern for a bent
sheet/bracket profile.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from build123d import (
    BuildLine,
    BuildPart,
    BuildSketch,
    FilletPolyline,
    Plane,
    Side,
    export_step,
    extrude,
    make_face,
    offset,
)


FLANGE_LENGTH_MM = 50.0
BASE_LENGTH_MM = 30.0
THICKNESS_MM = 1.2
WIDTH_MM = 30.0
BEND_ANGLE_DEG = 90.0


@dataclass(frozen=True)
class FlangeGeometry:
    """Result of generating a flange for a given bend radius."""

    r_bend: float
    volume_mm3: float
    is_valid: bool
    geometry_hash: str
    step_path: Path | None = None


def _geometry_hash(r_bend: float) -> str:
    payload = (
        f"r_bend={r_bend:.6f}|"
        f"length={FLANGE_LENGTH_MM}|"
        f"base={BASE_LENGTH_MM}|"
        f"t={THICKNESS_MM}|"
        f"w={WIDTH_MM}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def build_flange(
    r_bend: float,
    export_path: Path | None = None,
) -> FlangeGeometry:
    """
    Build a 90-degree bent sheet-metal flange.

    r_bend is the inner bend radius.

    The bend centerline is filleted with radius r_bend and then
    offset by the sheet thickness to create the constant-thickness
    cross-section.
    """

    if r_bend <= 0:
        raise ValueError("r_bend must be greater than zero")

    if r_bend >= BASE_LENGTH_MM:
        raise ValueError(
            f"r_bend must be smaller than BASE_LENGTH_MM ({BASE_LENGTH_MM} mm)"
        )

    with BuildPart() as part:

        # The sketch is the cross-section of the sheet.
        # BuildLine uses its local 2D coordinates; BuildSketch places
        # the resulting profile in the XZ plane.
        with BuildSketch(Plane.XZ):

            with BuildLine() as profile:

                # Centerline of the sheet:
                #
                #       |
                #       |
                #       |
                #       |
                #       |
                #       )
                # ----------------
                #
                # The corner is replaced by a circular fillet.
                FilletPolyline(
                    (0, 0),
                    (BASE_LENGTH_MM, 0),
                    (BASE_LENGTH_MM, FLANGE_LENGTH_MM),
                    radius=r_bend,
                )

                # Create the second boundary at the sheet thickness.
                #
                # For this orientation Side.LEFT puts the material on
                # the inside of the L-shaped path.
                offset(
                    amount=THICKNESS_MM,
                    side=Side.LEFT,
                )

            # The original and offset curves form a closed profile.
            make_face()

        # Extrude across the bend width.
        extrude(amount=WIDTH_MM)

    is_valid = (
        bool(part.part.is_valid)
        if hasattr(part.part, "is_valid")
        else True
    )

    volume = part.part.volume
    geom_hash = _geometry_hash(r_bend)

    step_path = None

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_step(part.part, str(export_path))
        step_path = export_path

    return FlangeGeometry(
        r_bend=r_bend,
        volume_mm3=volume,
        is_valid=is_valid and volume > 0,
        geometry_hash=geom_hash,
        step_path=step_path,
    )


if __name__ == "__main__":
    out_dir = Path("outputs/cad")

    for r in [4.0, 5.0, 5.6, 8.0]:
        result = build_flange(
            r,
            export_path=out_dir / f"flange_r{r:.1f}.step",
        )

        print(
            f"r={r:.2f}mm  "
            f"volume={result.volume_mm3:.1f}mm^3  "
            f"valid={result.is_valid}  "
            f"hash={result.geometry_hash}  "
            f"-> {result.step_path}"
        )