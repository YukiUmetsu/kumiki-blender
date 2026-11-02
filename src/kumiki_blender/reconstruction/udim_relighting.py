"""Private, reproducible multi-angle / opposing-light material inspection.

The same UVs and materials are held fixed under different light directions.
Persistent dark patches in both lighting conditions can reveal baked shadows;
these diagnostic views do not establish calibrated photometric accuracy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from kumiki_blender.blender_output import aim_at

from .blender_udim import _validated_package, import_finished_udim_package
from .dense import _sha256
from .references import ReferenceError


_AZIMUTHS_DEG = (0, 55, 110)
_LIGHTS = ("key-left", "key-right")


def validate_udim_relighting_request(package: str | Path, output: str | Path,
                                     resolution: int, samples: int) -> tuple[Path, Path]:
    """Preflight on host or in Blender, before changing the filesystem."""
    source, _ = _validated_package(package)
    target = Path(output).resolve()
    if (Path(output).is_symlink() or target.exists() or target == source
            or target.is_relative_to(source) or source.is_relative_to(target)):
        raise ReferenceError("Relighting review needs a new directory outside the UDIM package")
    if type(resolution) is not int or not 96 <= resolution <= 2048:
        raise ReferenceError("Relighting resolution must be 96–2048")
    if type(samples) is not int or not 1 <= samples <= 256:
        raise ReferenceError("Relighting samples must be 1–256")
    return source, target


def build_udim_relighting_review(bpy: Any, package: str | Path, output: str | Path,
                                 *, resolution: int = 512, samples: int = 24,
                                 render_previews: bool = True) -> dict[str, Any]:
    """Write an editable scene and a 3-angle x 2-light comparison gallery."""
    source, target = validate_udim_relighting_request(package, output, resolution, samples)
    if type(render_previews) is not bool:
        raise ReferenceError("render_previews must be a boolean")
    if bpy.app.background and bpy.data.filepath:
        raise ReferenceError("Run Blender relighting from a fresh factory-startup scene")
    objects = import_finished_udim_package(bpy, source)
    from mathutils import Vector  # Blender-only runtime

    corners = [obj.matrix_world @ Vector(corner) for obj in objects
               for corner in obj.bound_box]
    mins = [min(point[axis] for point in corners) for axis in range(3)]
    maxes = [max(point[axis] for point in corners) for axis in range(3)]
    center = tuple((lo + hi) * .5 for lo, hi in zip(mins, maxes))
    extent = max(high - low for low, high in zip(mins, maxes))
    if not math.isfinite(extent) or extent <= 1e-8:
        raise ReferenceError("UDIM mesh has invalid spatial bounds")

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.device = "CPU"  # Portable across Apple Silicon and CUDA machines.
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if hasattr(scene.view_settings, "view_transform"):
        scene.view_settings.view_transform = "AgX"
    world = bpy.data.worlds.new("Kumiki / consistent neutral environment")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.42, .42, .42, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .28
    scene.world = world

    camera_data = bpy.data.cameras.new("Kumiki / fixed comparison camera")
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = extent * 1.9
    scene.camera = camera

    lamp_data = bpy.data.lights.new("Kumiki / directional test light", "SUN")
    lamp = bpy.data.objects.new(lamp_data.name, lamp_data)
    scene.collection.objects.link(lamp)
    lamp.data.energy = 2.25
    lamp.data.angle = .14

    # Hide factory-startup lights and geometry without mutating imported data.
    for obj in scene.objects:
        if obj not in objects and obj not in (lamp, camera) and obj.type in {"LIGHT", "MESH"}:
            obj.hide_render = True

    target.mkdir(parents=True, exist_ok=False)
    comparison: list[dict[str, Any]] = []
    distance = extent * 3.2
    for label in _LIGHTS:
        direction = (-1 if label == "key-left" else 1)
        lamp.location = (center[0] + direction * extent * 2.0,
                         center[1] - extent * 1.8, center[2] + extent * 1.2)
        aim_at(lamp, center)
        for azimuth in _AZIMUTHS_DEG:
            theta = math.radians(azimuth)
            camera.location = (center[0] + distance * math.sin(theta),
                               center[1] - distance * math.cos(theta),
                               center[2] + distance * .18)
            aim_at(camera, center)
            filename = f"azimuth-{azimuth:03d}-{label}.png"
            if render_previews:
                scene.render.filepath = str(target / filename)
                bpy.ops.render.render(write_still=True)
                comparison.append({"azimuth_deg": azimuth, "lighting": label,
                                   "path": filename, "sha256": _sha256(target / filename)})

    scene.render.filepath = str(target / "preview.png")
    blend = target / "udim-material-review.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    if not blend.is_file():
        raise ReferenceError("Blender did not save editable UDIM material review")
    gallery = None
    if render_previews:
        gallery = "relighting-comparison.html"
        # Image filenames and labels are internally generated, never supplied
        # as HTML by a user-controlled manifest.
        cells = []
        for light in _LIGHTS:
            for angle in _AZIMUTHS_DEG:
                name = f"azimuth-{angle:03d}-{light}.png"
                cells.append(f'<figure><img src="{name}" alt="{light} / {angle} degrees">'
                             f'<figcaption>{light} · {angle}°</figcaption></figure>')
        html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<title>Kumiki UDIM relighting comparison</title>'
                '<style>body{margin:36px;background:#12151a;color:#e9edf2;'
                'font:15px system-ui,sans-serif}h1{font-size:24px}p{color:#abb5c0}'
                'main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}'
                'figure{margin:0;background:#21262d;padding:8px;border-radius:8px}'
                'img{width:100%;height:auto;display:block}figcaption{padding:8px}'
                '</style><h1>Material relighting comparison</h1>'
                '<p>Same mesh, camera positions and materials. Two opposing lights. '
                'Inspect dark regions that remain fixed when light direction changes.</p>'
                '<main>' + ''.join(cells) + '</main></html>')
        (target / gallery).write_text(html, encoding="utf-8")

    report = {
        "schema_version": 1,
        "status": "private-UDIM-relighting-review / artistic-review-required",
        "distribution_allowed": False,
        "photo_fidelity_verified": False,
        "source_udim_package": str(source),
        "source_udim_report_sha256": _sha256(source / "udim-report.json"),
        "blend": {"path": blend.name, "sha256": _sha256(blend)},
        "gallery": gallery,
        "comparison": comparison,
        "azimuths_deg": list(_AZIMUTHS_DEG), "lighting": list(_LIGHTS),
        "resolution": resolution, "cycles_samples": samples,
        "notes": "Diagnostic views are not photo-matched; the original front axis may differ. "
                 "Measured roughness and intrinsic albedo require calibrated multi-light data. "
                 "Image paths inside the .blend depend on the source UDIM package staying available.",
    }
    (target / "material-review-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                         encoding="utf-8")
    return report
