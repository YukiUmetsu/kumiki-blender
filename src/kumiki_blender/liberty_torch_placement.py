"""Pure placement and selection contracts for Liberty's detailed torch.

The historical procedural figure remains the coordinate source for its
gripping fingers. Detailed torch meshes are scaled to the existing handle
diameter, gallery width and flame height without moving the sculpted hand.
"""

from __future__ import annotations

from dataclasses import dataclass

from .liberty import LibertyStudy
from .shots import QualityShot


@dataclass(frozen=True)
class DetailedTorchPlacement:
    origin: tuple[float, float, float]
    radial_scale: float
    height_scale: float


def replace_legacy_torch_part(name: str) -> bool:
    return name.startswith("liberty-torch-") and not name.startswith("liberty-torch-grip-")


def detailed_torch_placement(study: LibertyStudy) -> DetailedTorchPlacement:
    try:
        handle = next(part for part in study.instances if part.name == "liberty-torch-handle")
        flames = [part for part in study.instances if part.name.startswith("liberty-torch-flame-")]
        if not flames:
            raise ValueError("No existing flame geometry")
    except (StopIteration, ValueError) as error:
        raise ValueError("A detailed Liberty torch needs the legacy handle and flame anchors") from error

    vertices = handle.mesh.vertices
    low = [min(p[i] for p in vertices) for i in range(3)]
    high = [max(p[i] for p in vertices) for i in range(3)]
    flame_top = max(p[2] for part in flames for p in part.mesh.vertices)
    return DetailedTorchPlacement(
        origin=((low[0]+high[0])/2, (low[1]+high[1])/2, low[2]),
        radial_scale=max(high[0]-low[0], high[1]-low[1]) / (2 * .115),
        height_scale=(flame_top-low[2]) / 1.86,
    )


def detailed_torch_shots(study: LibertyStudy) -> tuple[QualityShot, ...]:
    """Three inspectable views for the flame, cast gallery and hand contact."""
    fit = detailed_torch_placement(study)
    x, y, z = fit.origin
    unit = dict(study.parameters)["scale"]
    focus = (x, y, z+1.00*fit.height_scale)
    return (
        QualityShot("torch-detail-front", (x, y-6.5*unit, z+1.0*fit.height_scale),
                    focus, 3.7*unit),
        QualityShot("torch-detail-profile", (x+6.5*unit, y, z+1.0*fit.height_scale),
                    focus, 3.7*unit),
        QualityShot("torch-detail-low", (x-2.6*unit, y-5.7*unit, z-.12*unit),
                    focus, 3.8*unit),
        QualityShot("torch-flame-front", (x, y-4.5*unit, z+1.56*fit.height_scale),
                    (x, y, z+1.56*fit.height_scale), 2.4*unit),
        QualityShot("torch-grip-front", (x, y-4.5*unit, z+.43*fit.height_scale),
                    (x, y, z+.43*fit.height_scale), 2.6*unit),
        QualityShot("torch-grip-profile", (x+4.5*unit, y, z+.43*fit.height_scale),
                    (x, y, z+.43*fit.height_scale), 2.6*unit),
    )
