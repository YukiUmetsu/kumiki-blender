"""Native Blender smoke for multi-angle, frame-locked clay character review.

Uses only synthetic geometric test art; proves the review capture pipeline,
not finished Pip/Quill source-image fidelity or likeness.
"""
from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kumiki_blender.character_build.blender import run_worker  # noqa: E402
from kumiki_blender.character_build.character_review_plan import (  # noqa: E402
    confirm_review_sources, digest, make_character_review_plan,
)


def _png_chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
        ">I", zlib.crc32(kind + payload) & 0xffffffff)


def _fake_approved_reference(path):
    """A valid synthetic PNG; no external image tools within native Blender."""
    header = struct.pack(">2I5B", 64, 64, 8, 6, 0, 0, 0)
    scanline = b"\0" + b"\xee\xee\xee\xff" * 64
    png = (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header)
           + _png_chunk(b"IDAT", zlib.compress(scanline * 64))
           + _png_chunk(b"IEND", b""))
    path.write_bytes(png)


def _skinned(name, location, radius, group, rig):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=16, ring_count=8, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    weight = obj.vertex_groups.new(name=group)
    weight.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    modifier = obj.modifiers.new("Skin", "ARMATURE")
    modifier.object = rig
    return obj


def _fixture():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    arm = bpy.data.armatures.new("CharacterRig")
    rig = bpy.data.objects.new("CharacterRig", arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    root = arm.edit_bones.new("root")
    root.head = (0, 0, 0)
    root.tail = (0, 0, .25)
    wing = arm.edit_bones.new("wing")
    wing.head = (.06, 0, .12)
    wing.tail = (.18, 0, .13)
    wing.parent = root
    bpy.ops.object.mode_set(mode="OBJECT")

    _skinned("Skin", (0, 0, .18), .10, "root", rig)
    wing_obj = _skinned("Wing.L", (.145, -.016, .18), .045, "wing", rig)
    region = wing_obj.vertex_groups.new(name="REGION_WingLeft")
    region.add(list(range(len(wing_obj.data.vertices))), 1, "REPLACE")
    # A separate rigid accessory parented to the rig must remain visible
    # even without an Armature modifier, e.g. Pip's bag.
    bpy.ops.mesh.primitive_cube_add(size=.04, location=(-.08, 0, .16))
    bag = bpy.context.object
    bag.name = "Bag"
    matrix = bag.matrix_world.copy()
    bag.parent = rig
    bag.matrix_world = matrix

    bpy.context.scene.frame_set(1)
    pose = rig.pose.bones["wing"]
    pose.rotation_mode = "XYZ"
    pose.rotation_euler = (0, 0, 0)
    pose.keyframe_insert(data_path="rotation_euler", frame=1)
    bpy.context.scene.frame_set(12)
    pose.rotation_euler = (0, 0, math.radians(40))
    pose.keyframe_insert(data_path="rotation_euler", frame=12)
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()


def main():
    with tempfile.TemporaryDirectory(prefix="kumiki-character-review-") as tmp:
        folder = Path(tmp)
        _fixture()
        source = folder / "animated-character.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        frozen = digest(source)

        source_art = folder / "synthetic-approved-front.png"
        _fake_approved_reference(source_art)
        manifest = folder / "refs.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "views": {"front": str(source_art)},
            "regions": [{
                "id": "wing", "object": "Wing.L",
                "vertex_group": "REGION_WingLeft",
                "views": {"right": str(source_art)},
            }],
        }), encoding="utf-8")
        plan = make_character_review_plan(
            source, frames=[1, 12], views=["front", "right"],
            references=manifest, resolution=256)
        out = folder / "review-r01"
        report = run_worker(
            "character-review", {"plan": plan, "output": str(out)},
            blender=bpy.app.binary_path, blend_file=source, timeout_s=120)
        assert report["status"] == "source-visual-review-required"
        assert report["source_sha256"] == frozen
        assert report["camera_alignment"] == "unregistered-visual-review-only"
        assert report["render_style"] == "workbench-neutral-clay"
        assert report["frame_locked"]
        assert set(report["visible_skinned_parts"]) == {"Skin", "Wing.L", "Bag"}
        assert len(report["images"]) == 8

        by_key = {
            (row["scope"], row["view"], row["frame"]): row
            for row in report["images"]
        }
        front_1 = by_key[("full", "front", 1)]
        front_12 = by_key[("full", "front", 12)]
        assert front_1["camera"] == front_12["camera"], (
            "Review camera drifts or zooms between animated poses")
        wing_1 = by_key[("wing", "right", 1)]
        wing_12 = by_key[("wing", "right", 12)]
        assert wing_1["camera"] == wing_12["camera"]
        assert wing_1["camera"]["ortho_scale_m"] < front_1["camera"]["ortho_scale_m"]
        assert front_1["reference"]["sha256"] == digest(source_art)
        assert wing_1["reference"]["sha256"] == digest(source_art)
        assert (out / front_1["render"]).is_file()
        assert (out / front_12["render"]).is_file()
        assert (out / front_1["render"]).read_bytes() != (out / front_12["render"]).read_bytes(), (
            "Animated reviewed wing unexpectedly renders an identical frame")
        assert digest(source) == frozen
        confirm_review_sources(plan)
        print(json.dumps({
            "status": "native-multiview-review-capture-verified",
            "blender": bpy.app.version_string,
            "rendered_frames": len(report["images"]),
            "camera_poses_locked": True,
            "rig_parented_accessory_visible": True,
            "source_blend_unchanged": True,
        }, indent=2))


main()
