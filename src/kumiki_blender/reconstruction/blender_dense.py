"""Import a reviewed dense PLY into a private editable Blender QA scene."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from kumiki_blender.blender_import import imported_bounds
from kumiki_blender.blender_output import aim_at
from kumiki_blender.glb_audit import audit_glb

from .dense import _json, _ply_counts, _sha256
from .references import ReferenceError


def _textured_obj_plan(root: Path, report: dict[str, Any],
                       max_mesh_faces: int) -> dict[str, Any]:
    """Pin OBJ/MTL/atlases before Blender follows any relative file references."""
    obj_record = report.get("textured_obj")
    obj_name = (obj_record.get("path") if isinstance(obj_record, dict) else
                "workspace/scene_dense_mesh_refined_texture.obj")
    if (not isinstance(obj_name, str) or not obj_name.endswith(".obj")
            or Path(obj_name).is_absolute() or ".." in Path(obj_name).parts):
        raise ReferenceError("Dense textured OBJ path is unsafe")
    obj = root / obj_name
    if obj.is_symlink() or not obj.is_file() or not obj.resolve().is_relative_to(root):
        raise ReferenceError("Dense textured OBJ is missing or linked")
    with obj.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if isinstance(obj_record, dict) and obj_record.get("sha256") != digest:
        raise ReferenceError("Textured OBJ changed since dense reconstruction")
    atlases = report.get("texture_atlases")
    if atlases is None:
        atlases = [{"path": filename} for filename in report.get("texture_images", [])]
    if not isinstance(atlases, list) or not atlases:
        raise ReferenceError("Dense textured review needs a recorded private texture atlas")
    verified_atlases = []
    for entry in atlases:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ReferenceError("Invalid recorded texture atlas")
        relative = Path(entry["path"])
        if (relative.is_absolute() or ".." in relative.parts
                or relative.suffix.lower() not in (".png", ".jpg", ".jpeg")):
            raise ReferenceError("Unsafe texture atlas path")
        path = root / relative
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
                or path.parent != obj.parent):
            raise ReferenceError("Texture atlas missing or outside OBJ folder")
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if entry.get("sha256") is not None and entry["sha256"] != sha:
            raise ReferenceError("Texture atlas changed since dense reconstruction")
        verified_atlases.append({"path": str(path), "sha256": sha})
    # OpenMVS always writes a local MTL referenced by the OBJ. Scan the whole
    # OBJ, not just its header, so a later malicious mtllib cannot cause
    # Blender to load an unpinned external file. Count faces before import.
    directives = []
    face_count = 0
    with obj.open("rb") as stream:
        for line in stream:
            if line.startswith(b"mtllib "):
                directives.append(line[7:].decode("utf-8", errors="replace").strip())
            elif line.startswith(b"f "):
                face_count += 1
                if face_count > max_mesh_faces:
                    raise ReferenceError("Textured OBJ exceeds Blender review's face budget")
    if face_count == 0:
        raise ReferenceError("Textured OBJ contains no faces")
    if len(directives) != 1 or Path(directives[0]).name != directives[0]:
        raise ReferenceError("OBJ must reference exactly one local MTL file")
    material = obj.parent / directives[0]
    if material.is_symlink() or not material.is_file() or material.stat().st_size > 65_536:
        raise ReferenceError("OBJ MTL file missing, linked or oversized")
    mtl_lines = material.read_text(encoding="utf-8").splitlines()
    # OBJ material loaders accept additional path-bearing directives such as
    # bump, disp, norm and refl. Reject unknown MTL operations completely;
    # checking only map_* would still let Blender load unpinned local files.
    safe_mtl_directives = {
        "newmtl", "Ka", "Kd", "Ks", "Ke", "Ns", "Ni", "d", "Tr",
        "illum", "Tf", "Pr", "Pm", "Ps", "Pc", "Pcr", "aniso", "anisor",
        "map_Kd",
    }
    if any(line.strip() and not line.lstrip().startswith("#")
           and line.split(maxsplit=1)[0] not in safe_mtl_directives
           for line in mtl_lines):
        raise ReferenceError("OBJ MTL contains unsupported or unpinned resource directives")
    texture_names = [line.split(maxsplit=1)[1].strip() for line in mtl_lines
                     if line.startswith("map_Kd ")]
    if (not texture_names or any(name not in {Path(item["path"]).name
                                              for item in verified_atlases}
                                 for name in texture_names)
            or any(line.startswith("map_") and not line.startswith("map_Kd ")
                   for line in mtl_lines)):
        raise ReferenceError("OBJ MTL must reference only pinned diffuse atlases")
    return {"source_obj": str(obj), "source_obj_sha256": digest,
            "source_mtl": str(material), "texture_atlases": verified_atlases,
            "textured_faces": face_count}


def _png_metadata(path: Path) -> tuple[tuple[int, int], int, int]:
    """Read PNG IHDR without Pillow (Blender's Python has no QA extras)."""
    with path.open("rb") as stream:
        header = stream.read(26)
    if (len(header) != 26 or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[8:16] != b"\x00\x00\x00\rIHDR"):
        raise ReferenceError("Finished surface image is not a valid PNG header")
    dimensions = (int.from_bytes(header[16:20], "big"),
                  int.from_bytes(header[20:24], "big"))
    if not all(0 < size <= 16384 for size in dimensions):
        raise ReferenceError("Finished surface PNG dimensions are invalid")
    return dimensions, header[24], header[25]


def _finished_texture_plan(root: Path, dense_report: dict[str, Any],
                           texture_finish: str | Path, max_faces: int) -> dict[str, Any]:
    """Pin a non-destructive corrected atlas without trusting its filesystem links."""
    original = _textured_obj_plan(root, dense_report, max_faces)
    finish = Path(texture_finish).resolve()
    if Path(texture_finish).is_symlink() or not finish.is_dir():
        raise ReferenceError("Finished texture revision is missing or linked")
    record = _json(finish / "texture-finish-report.json")
    if (record.get("status") != "candidate / corrected-atlas-artistic-review-required"
            or record.get("source_dense_revision") != str(root)
            or record.get("source_dense_report_sha256") != _sha256(root / "dense-report.json")
            or record.get("source_obj_sha256") != original["source_obj_sha256"]
            or record.get("source_mtl_sha256") != _sha256(Path(original["source_mtl"]))):
        raise ReferenceError("Finished texture source or provenance differs from original dense mesh")
    original_atlases = {Path(item["path"]).name: item["sha256"]
                        for item in original["texture_atlases"]}
    expected = record.get("source_texture_atlases")
    if (not isinstance(expected, list) or
            {item.get("name"): item.get("sha256") for item in expected if isinstance(item, dict)}
            != original_atlases):
        raise ReferenceError("Finished texture original atlas provenance has changed")
    obj_record, mtl_record = record.get("textured_obj"), record.get("mtl")
    if (not isinstance(obj_record, dict) or not isinstance(mtl_record, dict)
            or obj_record.get("path") != Path(original["source_obj"]).name
            or mtl_record.get("path") != Path(original["source_mtl"]).name):
        raise ReferenceError("Finished texture OBJ/MTL does not match the pinned original")
    finished = _textured_obj_plan(finish, record, max_faces)
    if (finished["source_obj_sha256"] != original["source_obj_sha256"]
            or _sha256(Path(finished["source_mtl"])) != mtl_record.get("sha256")
            or mtl_record["sha256"] != record["source_mtl_sha256"]
            or finished["textured_faces"] != original["textured_faces"]):
        raise ReferenceError("Finished texture geometry or MTL changed since original")
    surface_entries = record.get("surface_maps", [])
    if not isinstance(surface_entries, list):
        raise ReferenceError("Finished surface maps must be a list")
    if record.get("surface_recipe_sha256") is not None and not surface_entries:
        raise ReferenceError("Finished surface maps missing despite a recorded recipe")
    if surface_entries:
        names = [item.get("atlas") for item in surface_entries if isinstance(item, dict)]
        if (len(names) != len(surface_entries) or len(names) != len(original_atlases)
                or set(names) != set(original_atlases)):
            raise ReferenceError("Finished surface maps need exactly one entry per atlas; duplicate/missing records")
    verified_surface = []
    for entry in surface_entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("atlas"), str):
            raise ReferenceError("Invalid finished surface map record")
        name = entry["atlas"]
        maps: dict[str, Any] = {}
        for channel in ("roughness", "height"):
            asset = entry.get(channel)
            if asset is None:
                if channel == "roughness":
                    raise ReferenceError("Finished material surface requires roughness map")
                maps[channel] = None
                continue
            expected_name = name + (".roughness.png" if channel == "roughness" else ".height.png")
            if (not isinstance(asset, dict) or asset.get("path") != expected_name
                    or not isinstance(asset.get("sha256"), str)):
                raise ReferenceError("Invalid finished surface map path/hash")
            path = finish / expected_name
            if (path.is_symlink() or not path.is_file()
                    or path.stat().st_size > 512 * 1024 * 1024
                    or _sha256(path) != asset["sha256"]):
                raise ReferenceError(f"Surface map changed after finishing: {expected_name}")
            dimensions, depth, color_type = _png_metadata(path)
            if (depth != (8 if channel == "roughness" else 16) or color_type != 0):
                raise ReferenceError("Finished surface map requires grayscale PNG at expected bit depth")
            maps[channel] = {"path": str(path), "sha256": asset["sha256"],
                             "dimensions": dimensions}
        scale = entry.get("displacement_scale_m")
        if ((maps["height"] is None) != (scale is None)
                or (scale is not None and (type(scale) not in (float, int)
                                             or not math.isfinite(scale) or not .00001 <= scale <= .5))):
            raise ReferenceError("Finished height displacement scale is invalid")
        # Match auxiliary PNG headers to pinned albedo metadata without
        # requiring optional Pillow inside Blender's bundled Python.
        atlas_records = [item for item in record.get("texture_atlases", [])
                         if isinstance(item, dict) and item.get("path") == name]
        if len(atlas_records) != 1:
            raise ReferenceError("Finished surface albedo metadata is missing or duplicate")
        reported_dims = atlas_records[0].get("dimensions")
        if (not isinstance(reported_dims, list) or len(reported_dims) != 2
                or any(type(value) is not int or not 0 < value <= 16384
                       for value in reported_dims)):
            raise ReferenceError("Finished surface albedo dimensions are invalid")
        source_size = tuple(reported_dims)
        if name.lower().endswith(".png") and _png_metadata(finish / name)[0] != source_size:
            raise ReferenceError("Finished albedo dimensions differ from its recorded metadata")
        for map_record in maps.values():
            if map_record is not None and map_record["dimensions"] != source_size:
                raise ReferenceError("Finished surface map dimensions differ from albedo")
        verified_surface.append({"atlas": name,
                                 **{key: ({"path": value["path"], "sha256": value["sha256"]}
                                     if value is not None else None) for key, value in maps.items()},
                                 "displacement_scale_m": scale})
    return {**finished, "texture_finish_revision": str(finish),
            "texture_finish_settings": record.get("settings"),
            "surface_maps": verified_surface,
            "texture_quality_verified": False}


def _attach_finished_surface_nodes(bpy: Any, material: Any,
                                   entry: dict[str, Any]) -> None:
    """Use hash-checked PBR maps. True height affects Blender, not the GLB."""
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if shader is None:
        raise ReferenceError("Finished material has no Principled BSDF")

    def texture(channel: str) -> Any:
        source = entry[channel]
        image = bpy.data.images.load(source["path"], check_existing=False)
        image.colorspace_settings.name = "Non-Color"
        image.pack()
        if image.packed_file is None:
            raise ReferenceError(f"Cannot pack finished {channel} map")
        node = nodes.new("ShaderNodeTexImage")
        node.label = f"Kumiki / finished {channel} / {entry['atlas']}"
        node.image = image
        return node

    rough = texture("roughness")
    links.new(rough.outputs["Color"], shader.inputs["Roughness"])
    if entry["height"] is not None:
        height = texture("height")
        displacement = nodes.new("ShaderNodeDisplacement")
        displacement.inputs["Scale"].default_value = entry["displacement_scale_m"]
        displacement.inputs["Midlevel"].default_value = 0.5
        links.new(height.outputs["Color"], displacement.inputs["Height"])
        output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
        if output is None:
            raise ReferenceError("Finished material has no Material Output node")
        links.new(displacement.outputs["Displacement"], output.inputs["Displacement"])
        # Blender 4.x/5.x material property names differ. Do not invent a
        # universal setting: leave a clear Displacement node for Cycles artists.
        if hasattr(material, "displacement_method"):
            material.displacement_method = "BOTH"


def _matched_scan_cameras(report: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Camera poses and the dense PLY share COLMAP coordinates: no sculpt anchors."""
    from .cameras import (MAPANYTHING_MAX_FOCAL_ASYMMETRY, _matmul, _matvec, _quat_matrix,
                          _read_intrinsics, _read_poses)

    source_value = report.get("source_revision")
    if not isinstance(source_value, str):
        raise ReferenceError("Matched views need the source sparse reconstruction revision")
    source = Path(source_value).resolve()
    if (Path(source_value).is_symlink() or not source.is_dir()
            or (source / "undistorted").is_symlink()):
        raise ReferenceError("Matched-view sparse revision is missing or linked")
    sparse_report = _json(source / "reconstruction-report.json")
    if not str(sparse_report.get("status", "")).startswith("candidate"):
        raise ReferenceError("Matched-view sparse reconstruction is not a candidate")
    model = source / "undistorted/text-model"
    photo_root = source / "undistorted/images"
    if model.is_symlink() or not model.is_dir() or photo_root.is_symlink():
        raise ReferenceError("Rectified camera model or images are missing or linked")
    index = _json(source / "undistorted-reference-index.json")
    entries = {entry["filename"]: entry for entry in index.get("images", [])
               if isinstance(entry, dict) and isinstance(entry.get("filename"), str)}
    poses = _read_poses(model / "images.txt")
    mapanything = sparse_report.get("engine") == "mapanything"
    intrinsics = _read_intrinsics(
        model / "cameras.txt", require_square_pixels=not mapanything,
        max_focal_asymmetry=MAPANYTHING_MAX_FOCAL_ASYMMETRY if mapanything else None)
    if (len(poses) < count or len(entries) != len(index.get("images", []))
            or len(poses) != sparse_report.get("registered_images")):
        raise ReferenceError("Matched-view image and camera counts differ from sparse report")
    # Even spacing through COLMAP image IDs, with no subjective view labels.
    chosen = [poses[i * len(poses) // count] for i in range(count)]
    matched = []
    for image_id, camera_id, filename, quaternion, tvec in chosen:
        entry = entries.get(filename)
        intr = intrinsics.get(camera_id)
        photo = photo_root / filename
        if (entry is None or intr is None or photo.is_symlink() or not photo.is_file()
                or not photo.resolve().is_relative_to(source)
                or entry.get("dimensions") != [intr["width"], intr["height"]]):
            raise ReferenceError(f"Rectified camera/photo mismatch for {filename}")
        with photo.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry.get("sha256"):
            raise ReferenceError(f"Rectified photo changed since reconstruction: {filename}")
        world_to_camera = _quat_matrix(quaternion)
        camera_to_world = [[world_to_camera[j][i] for j in range(3)] for i in range(3)]
        center = _matvec(camera_to_world, tuple(-value for value in tvec))
        orientation = _matmul(camera_to_world, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        matched.append({"image_id": image_id, "filename": filename,
                        "source_photo_path": str(photo), "source_photo_sha256": digest,
                        "intrinsics": intr, "location_colmap": center,
                        "rotation_3x3": orientation})
    return matched


def plan_dense_blender_review(dense_revision: str | Path, output: str | Path,
                              *, max_mesh_faces: int = 5_000_000,
                              matched_views: int = 0,
                              matched_resolution_percent: int = 100,
                              textured: bool = False,
                              texture_finish: str | Path | None = None) -> dict[str, Any]:
    """Check a dense mesh has not changed since its reconstruction report."""
    root = Path(dense_revision).resolve()
    target = Path(output).resolve()
    if Path(dense_revision).is_symlink() or not root.is_dir():
        raise ReferenceError("Dense reconstruction revision is missing or linked")
    if target.exists() or Path(output).is_symlink() or target.is_relative_to(root):
        raise ReferenceError("Blender review needs an unused output outside dense revision")
    if type(max_mesh_faces) is not int or not 100 <= max_mesh_faces <= 10_000_000:
        raise ReferenceError("max_mesh_faces must be 100–10,000,000")
    if type(matched_views) is not int or not 0 <= matched_views <= 12:
        raise ReferenceError("matched_views must be 0–12")
    if (type(matched_resolution_percent) is not int
            or not 10 <= matched_resolution_percent <= 100):
        raise ReferenceError("matched_resolution_percent must be 10–100")
    if type(textured) is not bool:
        raise ReferenceError("textured must be a boolean")
    if texture_finish is not None and not textured:
        raise ReferenceError("A finished texture revision requires --textured")
    report = _json(root / "dense-report.json")
    mesh = report.get("mesh")
    if (not isinstance(report.get("status"), str)
            or not report["status"].startswith("dense-mesh-candidate")
            or not isinstance(mesh, dict)):
        raise ReferenceError("Dense revision is not a candidate mesh")
    filename = mesh.get("ply")
    expected = mesh.get("sha256")
    if (not isinstance(filename, str) or not filename.endswith(".ply")
            or not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)):
        raise ReferenceError("Dense mesh report lacks a safe PLY path and SHA-256")
    relative = Path(filename)
    if relative.is_absolute() or any(part in (".", "..") for part in relative.parts):
        raise ReferenceError("Dense mesh path is not a safe relative path")
    path = root / relative
    if (path.is_symlink() or not path.resolve().is_relative_to(root)
            or not path.is_file()):
        raise ReferenceError("Dense PLY file is missing or linked outside the revision")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ReferenceError("Dense mesh PLY changed after reconstruction")
    counts = _ply_counts(path)
    if counts["faces"] != mesh.get("faces") or counts["vertices"] != mesh.get("vertices"):
        raise ReferenceError("Dense mesh header no longer matches audited geometry")
    if counts["faces"] > max_mesh_faces:
        raise ReferenceError("Dense mesh exceeds Blender review's face budget; split the scan")
    cameras = _matched_scan_cameras(report, matched_views) if matched_views else []
    texture_plan = (_finished_texture_plan(root, report, texture_finish, max_mesh_faces)
                    if textured and texture_finish is not None else
                    _textured_obj_plan(root, report, max_mesh_faces) if textured else {})
    return {"source_revision": str(root), "source_ply": str(path), "source_sha256": actual,
            "output": str(target), "faces": counts["faces"], "vertices": counts["vertices"],
            "matched_cameras": cameras,
            "matched_resolution_percent": matched_resolution_percent,
            "textured": textured, **texture_plan,
            "status": "preflight-only / Blender-review-required",
            "distribution_allowed": False}


def _preview_axes(bpy: Any, obj: Any, target: Path) -> list[str]:
    scene = bpy.context.scene
    bounds_min, bounds_max = imported_bounds([obj])
    center = tuple((bounds_min[i]+bounds_max[i])/2 for i in range(3))
    size = max(bounds_max[i]-bounds_min[i] for i in range(3))
    if not math.isfinite(size) or size <= 0:
        raise ReferenceError("Dense mesh has invalid or degenerate bounds")
    camera_data = bpy.data.cameras.new("Private dense QA / orthographic views")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(0.5, size*1.75)
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    world = bpy.data.worlds.new("Private dense QA / neutral world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.35, 0.35, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
    scene.world = world
    light_data = bpy.data.lights.new("Dense QA / raking key", "AREA")
    light_data.energy = 900.0
    light_data.shape = "DISK"
    light_data.size = max(size*1.3, 0.2)
    light = bpy.data.objects.new(light_data.name, light_data)
    scene.collection.objects.link(light)
    light.location = (center[0]+size, center[1]-size, center[2]+size*1.5)
    aim_at(light, center)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.render.resolution_x = 320
    scene.render.resolution_y = 320
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    views = [("axis-negative-y", (0, -1.8, 0.4)),
             ("axis-negative-x", (-1.8, 0, 0.4)),
             ("axis-three-quarter", (1.5, -1.5, 0.9))]
    paths = []
    for name, direction in views:
        camera.location = tuple(center[i]+size*direction[i] for i in range(3))
        aim_at(camera, center)
        filename = "renders/" + name + ".png"
        scene.render.filepath = str(target / filename)
        bpy.ops.render.render(write_still=True)
        paths.append(filename)
    return paths


def _preview_matched(bpy: Any, target: Path, entries: list[dict[str, Any]],
                     resolution_percent: int) -> list[str]:
    """Clay renders from rectified photo cameras, in the unscaled scan frame."""
    from mathutils import Matrix  # type: ignore[import-not-found]

    scene = bpy.context.scene
    scene.render.resolution_percentage = resolution_percent
    output = []
    for index, entry in enumerate(entries):
        intr = entry["intrinsics"]
        camera_data = bpy.data.cameras.new("Scan/photo registration / " + entry["filename"])
        camera_data.type = "PERSP"
        camera_data.sensor_fit = "HORIZONTAL"
        camera_data.sensor_width = 36.0
        camera_data.lens = 0.5 * (intr["fx_px"] + intr["fy_px"]) * 36.0 / intr["width"]
        camera_data.shift_x = (intr["width"] / 2 - intr["cx_px"]) / intr["width"]
        camera_data.shift_y = (intr["cy_px"] - intr["height"] / 2) / intr["width"]
        camera_data.show_background_images = True
        background = camera_data.background_images.new()
        background.image = bpy.data.images.load(entry["source_photo_path"], check_existing=True)
        background.alpha = 0.45
        camera = bpy.data.objects.new(camera_data.name, camera_data)
        scene.collection.objects.link(camera)
        rotation = entry["rotation_3x3"]
        location = entry["location_colmap"]
        camera.matrix_world = Matrix((
            (*rotation[0], location[0]), (*rotation[1], location[1]),
            (*rotation[2], location[2]), (0, 0, 0, 1),
        ))
        scene.camera = camera
        scene.render.resolution_x = intr["width"]
        scene.render.resolution_y = intr["height"]
        name = f"renders/photo-matched-{index + 1:02d}.png"
        scene.render.filepath = str(target / name)
        bpy.ops.render.render(write_still=True)
        output.append(name)
    return output


def build_dense_blender_review(dense_revision: str | Path, output: str | Path,
                               *, bpy_module: Any, render_previews: bool = True,
                               max_mesh_faces: int = 5_000_000,
                               matched_views: int = 0,
                               matched_resolution_percent: int = 100,
                               textured: bool = False,
                               texture_finish: str | Path | None = None) -> dict[str, Any]:
    """Save a private editable .blend and GLB; neither clears photo rights."""
    plan = plan_dense_blender_review(dense_revision, output,
                                     max_mesh_faces=max_mesh_faces,
                                     matched_views=matched_views,
                                     matched_resolution_percent=matched_resolution_percent,
                                     textured=textured, texture_finish=texture_finish)
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise ReferenceError("Dense Blender review requires Blender 4.2+")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = (bpy.ops.wm.obj_import(filepath=plan["source_obj"],
                                    forward_axis="Y", up_axis="Z") if textured else
              bpy.ops.wm.ply_import(filepath=plan["source_ply"], import_colors="SRGB"))
    if "FINISHED" not in result:
        raise ReferenceError("Blender could not import the private dense mesh")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(meshes) != 1 or not meshes[0].data.polygons:
        raise ReferenceError("Dense review expects exactly one nonempty mesh")
    obj = meshes[0]
    if not textured and (len(obj.data.polygons) != plan["faces"] or
                        len(obj.data.vertices) != plan["vertices"]):
        raise ReferenceError("Blender geometry differs from audited PLY counts")
    if textured and len(obj.data.polygons) != plan["textured_faces"]:
        raise ReferenceError("Blender textured OBJ geometry differs from audited face count")
    obj.name = "Private dense scan / unverified scale and axes"
    if textured:
        linked = {Path(item["path"]).resolve() for item in plan["texture_atlases"]}
        surface_by_name = {item["atlas"]: item for item in plan.get("surface_maps", [])}
        for material in obj.data.materials:
            if material is None or not material.use_nodes:
                raise ReferenceError("Textured OBJ lost its image-linked material")
            images = [node.image for node in material.node_tree.nodes
                      if node.type == "TEX_IMAGE" and node.image is not None]
            if not images or any(Path(bpy.path.abspath(img.filepath)).resolve() not in linked
                                 for img in images):
                raise ReferenceError("Textured OBJ references an unpinned image")
            for shader in material.node_tree.nodes:
                if shader.type == "BSDF_PRINCIPLED":
                    shader.inputs["Roughness"].default_value = 0.85
                    shader.inputs["Alpha"].default_value = 1.0
            if surface_by_name:
                # OBJ material groups may use shared atlases; the map is per
                # atlas so each group receives consistent surface treatment.
                image_names = {Path(bpy.path.abspath(img.filepath)).name for img in images}
                if len(image_names) != 1 or next(iter(image_names)) not in surface_by_name:
                    raise ReferenceError("Cannot uniquely associate finished material and atlas")
                _attach_finished_surface_nodes(bpy, material,
                                               surface_by_name[next(iter(image_names))])
        bpy.ops.file.pack_all()  # Private portable .blend; GLB embeds the atlas separately.
    else:
        material = bpy.data.materials.new("Dense statue / neutral review material")
        material.use_nodes = True
        shader = material.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (0.53, 0.49, 0.44, 1.0)
        shader.inputs["Roughness"].default_value = 0.92
        if obj.data.color_attributes:
            color = obj.data.color_attributes.active_color
            if color is not None:
                attribute = material.node_tree.nodes.new("ShaderNodeAttribute")
                attribute.attribute_name = color.name
                material.node_tree.links.new(attribute.outputs["Color"], shader.inputs["Base Color"])
        obj.data.materials.clear()
        obj.data.materials.append(material)
    target = Path(plan["output"])
    target.mkdir(parents=True, exist_ok=False)
    record = {**plan, "status": "review-required",
              "historical_fidelity_verified": False, "previews": [],
              "photo_matched_previews": [],
              "texture_quality_verified": False if textured else None,
              "review_notes": ("Inspect black/empty atlas patches, seams and "
                               "foreground contamination at matched photo pixels"
                               if textured else "Neutral PLY geometry is untextured")}

    def checkpoint() -> None:
        (target / "blender-review-report.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")

    checkpoint()
    try:
        if render_previews:
            (target / "renders").mkdir()
            record["previews"] = _preview_axes(bpy, obj, target)
            if plan["matched_cameras"]:
                record["photo_matched_previews"] = _preview_matched(
                    bpy, target, plan["matched_cameras"], matched_resolution_percent)
        base_name = "private-textured-review" if textured else "private-dense-review"
        bpy.ops.wm.save_as_mainfile(filepath=str(target / (base_name + ".blend")))
        for item in bpy.context.selected_objects:
            item.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        result = bpy.ops.export_scene.gltf(filepath=str(target / (base_name + ".glb")),
                                           export_format="GLB", use_selection=True,
                                           export_animations=False)
        if "FINISHED" not in result:
            raise ReferenceError("Blender failed to export the private review GLB")
        record["glb_audit"] = audit_glb(target / (base_name + ".glb"))
        record["review_blend"] = base_name + ".blend"
        record["review_glb"] = base_name + ".glb"
        record["status"] = "private-Blender-mesh-candidate / artistic-and-rights-review-required"
    except (ReferenceError, RuntimeError, OSError, ValueError) as error:
        record.update(failed_stage="Blender-review", error=str(error))
    checkpoint()
    return record
