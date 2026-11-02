"""Private, reproducible multi-angle / opposing-light material inspection.

The same UVs and materials are held fixed under different light directions.
Persistent dark patches in both lighting conditions can reveal baked shadows;
these diagnostic views do not establish calibrated photometric accuracy.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from kumiki_blender.blender_output import aim_at

from .blender_udim import _validated_package, import_finished_udim_package
from .dense import _sha256
from .references import ReferenceError


_AZIMUTHS_DEG = (0, 55, 110)
_LIGHTS = ("key-left", "key-right")


def _matched_photo_entries(package: Path, count: int) -> list[dict[str, Any]]:
    """Reuse validated sparse COLMAP camera evidence without fabricating poses."""
    if not count:
        return []
    from .blender_dense import _matched_scan_cameras
    from .dense import _json

    record = _json(package / "udim-report.json")
    finish = Path(record["source_texture_finish_revision"])
    finish_report_path = finish / "texture-finish-report.json"
    if (finish.is_symlink() or not finish.is_dir()
            or finish_report_path.is_symlink() or not finish_report_path.is_file()
            or _sha256(finish_report_path)
            != record.get("source_texture_finish_report_sha256")):
        raise ReferenceError("UDIM source texture finishing evidence changed")
    finish_report = _json(finish_report_path)
    dense = Path(finish_report["source_dense_revision"])
    if (dense.is_symlink() or not dense.is_dir()
            or (dense / "dense-report.json").is_symlink()):
        raise ReferenceError("UDIM source dense reconstruction is missing")
    cameras = _matched_scan_cameras(_json(dense / "dense-report.json"), count)
    if any((Path(item["source_photo_path"]).stat().st_size > 512 * 1024 * 1024
            or Path(item["source_photo_path"]).suffix.lower()
            not in {".jpg", ".jpeg", ".png", ".webp"})
           for item in cameras):
        raise ReferenceError("Photo-matched review image is oversized or unsupported")
    return cameras


def validate_udim_relighting_request(package: str | Path, output: str | Path,
                                     resolution: int, samples: int,
                                     matched_views: int = 0,
                                     matched_resolution_percent: int = 25
                                     ) -> tuple[Path, Path, list[dict[str, Any]]]:
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
    if type(matched_views) is not int or not 0 <= matched_views <= 6:
        raise ReferenceError("Relighting matched_views must be 0–6")
    if (type(matched_resolution_percent) is not int
            or not 10 <= matched_resolution_percent <= 100):
        raise ReferenceError("Matched photo render resolution percent must be 10–100")
    cameras = _matched_photo_entries(source, matched_views)
    return source, target, cameras


def build_udim_relighting_review(bpy: Any, package: str | Path, output: str | Path,
                                 *, resolution: int = 512, samples: int = 24,
                                 render_previews: bool = True,
                                 matched_views: int = 0,
                                 matched_resolution_percent: int = 25) -> dict[str, Any]:
    """Write an editable scene and a 3-angle x 2-light comparison gallery."""
    source, target, photo_entries = validate_udim_relighting_request(
        package, output, resolution, samples, matched_views, matched_resolution_percent)
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
    photo_comparison: list[dict[str, Any]] = []
    distance = extent * 3.2

    def aim_light(label: str) -> None:
        direction = (-1 if label == "key-left" else 1)
        lamp.location = (center[0] + direction * extent * 2.0,
                         center[1] - extent * 1.8, center[2] + extent * 1.2)
        aim_at(lamp, center)

    for label in _LIGHTS:
        aim_light(label)
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

    # Optional exact photo camera poses are stronger evidence than a generic
    # side/front guess; the lighting still deliberately differs from the shot.
    if photo_entries:
        from mathutils import Matrix

        for index, entry in enumerate(photo_entries, start=1):
            intr = entry["intrinsics"]
            calibrated = bpy.data.cameras.new(f"Kumiki / photo-matched {index:02d}")
            calibrated.type = "PERSP"
            calibrated.sensor_fit = "HORIZONTAL"
            calibrated.sensor_width = 36.0
            calibrated.lens = intr["fx_px"] * 36.0 / intr["width"]
            calibrated.shift_x = (intr["width"] / 2 - intr["cx_px"]) / intr["width"]
            calibrated.shift_y = (intr["cy_px"] - intr["height"] / 2) / intr["width"]
            camera_obj = bpy.data.objects.new(calibrated.name, calibrated)
            scene.collection.objects.link(camera_obj)
            orientation, location = entry["rotation_3x3"], entry["location_colmap"]
            camera_obj.matrix_world = Matrix((
                (*orientation[0], location[0]), (*orientation[1], location[1]),
                (*orientation[2], location[2]), (0, 0, 0, 1),
            ))
            source_photo = Path(entry["source_photo_path"])
            filename = f"registered-photo-{index:02d}{source_photo.suffix.lower()}"
            if render_previews:
                shutil.copyfile(source_photo, target / filename)
            record: dict[str, Any] = {
                "reference_photo": filename if render_previews else None,
                "reference_sha256": entry["source_photo_sha256"],
                "source_photo_path": entry["source_photo_path"],
                "source_image_id": entry["image_id"],
                "rendered_lights": [],
            }
            for light in _LIGHTS:
                aim_light(light)
                scene.camera = camera_obj
                scene.render.resolution_x = intr["width"]
                scene.render.resolution_y = intr["height"]
                scene.render.resolution_percentage = matched_resolution_percent
                name = f"matched-{index:02d}-{light}.png"
                if render_previews:
                    scene.render.filepath = str(target / name)
                    bpy.ops.render.render(write_still=True)
                    record["rendered_lights"].append({
                        "lighting": light, "path": name, "sha256": _sha256(target / name)})
            photo_comparison.append(record)

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

    matched_gallery = None
    if render_previews and photo_comparison:
        matched_gallery = "photo-matched-comparison.html"
        rows = []
        for record in photo_comparison:
            files = [record["reference_photo"]] + [item["path"] for item in record["rendered_lights"]]
            labels = ["Registered photograph", "Key left", "Key right"]
            rows.extend(f'<figure><img src="{path}" alt="{label}">'
                        f'<figcaption>{label}</figcaption></figure>'
                        for path, label in zip(files, labels))
        html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<title>Kumiki photo-matched material comparison</title>'
                '<style>body{margin:36px;background:#12151a;color:#e9edf2;'
                'font:15px system-ui}main{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}'
                'figure{margin:0;background:#21262d;padding:8px}img{width:100%;height:auto}'
                'figcaption{padding:8px}</style><h1>Calibrated camera comparison</h1>'
                '<p>Camera positions match registered photos. The studio lighting '
                'does not reproduce original sunlight, so color differences are expected.</p>'
                '<main>' + ''.join(rows) + '</main></html>')
        (target / matched_gallery).write_text(html, encoding="utf-8")

    report = {
        "schema_version": 1,
        "status": "private-UDIM-relighting-review / artistic-review-required",
        "distribution_allowed": False,
        "photo_fidelity_verified": False,
        "source_udim_package": str(source),
        "source_udim_report_sha256": _sha256(source / "udim-report.json"),
        "blend": {"path": blend.name, "sha256": _sha256(blend)},
        "gallery": gallery,
        "matched_gallery": matched_gallery,
        "comparison": comparison,
        "photo_matched_comparison": photo_comparison,
        "azimuths_deg": list(_AZIMUTHS_DEG), "lighting": list(_LIGHTS),
        "resolution": resolution, "cycles_samples": samples,
        "matched_resolution_percent": matched_resolution_percent,
        "notes": "Diagnostic views are not photo-matched; the original front axis may differ. "
                 "Optional COLMAP camera views are photo-calibrated but studio lighting is not. "
                 "Measured roughness and intrinsic albedo require calibrated multi-light data. "
                 "Image paths inside the .blend depend on the source UDIM package staying available.",
    }
    (target / "material-review-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                         encoding="utf-8")
    return report
