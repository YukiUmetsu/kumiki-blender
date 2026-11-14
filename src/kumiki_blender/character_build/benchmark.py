"""M0 character benchmark: evidence-linked, repeatable release-readiness inventory.

Collects existing independent character pipeline reports, verifies frozen
CharacterSpec provenance, inventories real GLB tracks, and shows precisely
which numeric/pose/source gates remain absent. No subprocess execution or
private artwork copying. A numeric pass is NEVER artistic approval.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import tempfile
import time
from pathlib import Path

from .character_spec_v2 import open_plan

_REQUIRED_CHECKS = frozenset({"regions", "textures", "pose", "glb"})
_KEYS = {"schema_version", "character_plan", "required_checks", "required_regions",
         "region_reports", "texture_reports", "pose_report", "glb", "required_poses",
         "required_clips", "runtime_observations"}
_JSON_CHUNK = 0x4E4F534A


def _sha(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(root: Path, value: str, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label}: expected a nonempty filename")
    path = (root / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def _report(path: Path, label: str) -> dict:
    if path.stat().st_size > 8_000_000:
        raise ValueError(f"{label}: report exceeds the 8 MB safety budget")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
        raise ValueError(f"{label}: invalid versioned report")
    return payload


def _list(value, label, *, max_count=128):
    if not isinstance(value, list) or len(value) > max_count:
        raise ValueError(f"{label}: expected list of at most {max_count}")
    return value


def _unique_ids(rows, key, label):
    found = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"{label}: missing {key}")
        if row[key] in found:
            raise ValueError(f"{label}: duplicate {key} {row[key]}")
        found.add(row[key])
    return found


def inspect_glb(path: Path, *, required_clips: list[str]) -> dict:
    """Read only GLB JSON chunk; never load multi-GB vertex/image binary data."""
    if path.suffix.lower() != ".glb" or path.stat().st_size > 1_000_000_000:
        raise ValueError("Expected a GLB smaller than 1 GB")
    with path.open("rb") as stream:
        header = stream.read(20)
        if len(header) != 20:
            raise ValueError("GLB truncated")
        magic, version, total, size, kind = struct.unpack("<4sIIII", header)
        if (magic != b"glTF" or version != 2 or total != path.stat().st_size
                or kind != _JSON_CHUNK or size < 2 or size > 16_000_000 or size % 4):
            raise ValueError("GLB has invalid header or oversized JSON chunk")
        document = json.loads(stream.read(size).decode("utf-8"))
        if not isinstance(document, dict) or document.get("asset", {}).get("version") != "2.0":
            raise ValueError("GLB JSON requires asset.version 2.0")
        # Skip all other chunks without reading binary vertex/image payloads.
        offset = 20 + size
        while offset < total:
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise ValueError("GLB contains an incomplete chunk header")
            n, _kind = struct.unpack("<II", chunk)
            if n % 4 or offset + 8 + n > total:
                raise ValueError("GLB contains an invalid binary chunk")
            stream.seek(n, 1)
            offset += 8 + n
        if offset != total:
            raise ValueError("GLB chunks exceed file size")
    nodes = document.get("nodes", [])
    meshes = document.get("meshes", [])
    skins = document.get("skins", [])
    animations = document.get("animations", [])
    accessors = document.get("accessors", [])
    if not all(isinstance(seq, list) for seq in (nodes, meshes, skins, animations, accessors)):
        raise ValueError("GLB has malformed inventory arrays")
    bone_ids = set()
    for skin in skins:
        for joint in skin.get("joints", []):
            if type(joint) is not int or not 0 <= joint < len(nodes):
                raise ValueError("GLB skin refers to a nonexistent joint node")
            bone_ids.add(joint)
    names = []
    channels = []
    for animation in animations:
        name = animation.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("GLB animation clips require unique nonempty names")
        names.append(name)
        count_joint = count_morph = 0
        samplers = animation.get("samplers", [])
        for channel in animation.get("channels", []):
            sampler = channel.get("sampler")
            target = channel.get("target", {})
            node = target.get("node")
            track_path = target.get("path")
            if (type(sampler) is not int or not 0 <= sampler < len(samplers)
                    or type(node) is not int or not 0 <= node < len(nodes)
                    or track_path not in {"translation", "rotation", "scale", "weights"}):
                raise ValueError(f"Malformed animation channel in {name}")
            if track_path == "weights":
                count_morph += 1
            elif node in bone_ids:
                count_joint += 1
        channels.append({"clip": name, "deform_joint_channels": count_joint,
                         "morph_weight_channels": count_morph,
                         "total_channels": len(animation.get("channels", []))})
    morphs = sum(len(primitive.get("targets", []))
                 for mesh in meshes for primitive in mesh.get("primitives", []))
    faces = 0
    for mesh in meshes:
        for primitive in mesh.get("primitives", []):
            indices = primitive.get("indices")
            if (type(indices) is int and 0 <= indices < len(accessors)
                    and primitive.get("mode", 4) == 4):
                faces += accessors[indices].get("count", 0) // 3
    missing = sorted(set(required_clips) - set(names))
    return {
        "glb_sha256": _sha(path),
        "bytes": path.stat().st_size,
        "mesh_count": len(meshes), "node_count": len(nodes),
        "skin_count": len(skins), "joint_count": len(bone_ids),
        "estimated_indexed_triangle_faces": faces,
        "morph_target_count": morphs, "clip_names": names,
        "animation_channels": channels,
        "missing_required_clips": missing,
        "numeric_ready": bool(meshes and skins and bone_ids and names and not missing),
    }


def _render_summary(result: dict) -> str:
    checks = result["checks"]
    status = result["status"]
    lines = [
        f"# Character benchmark — {result['character_id']}", "",
        f"**Status:** {status} (artistic review is always required)", "",
        f"Source CharacterSpec SHA-256: {result['source_sha256']}", "",
        "| Gate | Status | Evidence |", "|---|---|---|",
    ]
    for name in ("regions", "textures", "pose", "glb"):
        gate = checks[name]
        lines.append(
            f"| {name} | {gate['status']} | "
            f"{gate.get('checked', 0)} checked; {len(gate['missing'])} missing; "
            f"{len(gate['failures'])} failed |")
    lines.extend([
        "", f"**Missing required evidence:** {', '.join(result['missing_evidence']) or 'none'}",
        "", f"**Numeric failures:** {', '.join(result['numeric_failures']) or 'none'}",
        "",
        "**Important:** Numeric gates do not verify actual Pip/Quill likeness, "
        "texture realism, untested hidden views, or animation parity.",
        "Inspect original reference images and posed Blender/Kumiki renders.",
        "",
    ])
    return "\n".join(lines)


def character_benchmark(manifest: str | Path, output: str | Path) -> dict:
    started = time.perf_counter()
    manifest = Path(manifest).resolve()
    root = manifest.parent
    manifest_hash = _sha(manifest)
    raw = _report(manifest, "benchmark manifest")
    if (raw.get("schema_version") != 1 or set(raw) - _KEYS or
            not isinstance(raw.get("character_plan"), str)):
        raise ValueError("Invalid character benchmark schema_version 1 manifest")
    required = _list(raw.get("required_checks", sorted(_REQUIRED_CHECKS)), "required_checks", max_count=4)
    if (not required or len(set(required)) != len(required) or not set(required) <= _REQUIRED_CHECKS):
        raise ValueError("required_checks contains duplicate or unsupported gates")
    character_dir = (root / raw["character_plan"]).resolve()
    plan, spec = open_plan(character_dir)
    verified = {
        row["id"]: row.get("verified_views", [])
        for row in spec["regions"] if row.get("verified_views")
    }
    region_ids = _list(raw.get("required_regions", list(verified)), "required_regions")
    if len(set(region_ids)) != len(region_ids) or not set(region_ids) <= set(verified):
        raise ValueError("required_regions must be distinct regions with verified original cameras")
    regions = _list(raw.get("region_reports", []), "region_reports")
    textures = _list(raw.get("texture_reports", []), "texture_reports")
    if not _unique_ids(regions, "region", "region_reports") <= set(verified):
        raise ValueError("Region report is for an unknown/unverified source region")
    _unique_ids(textures, "id", "texture_reports")
    required_clips = _list(raw.get("required_clips", []), "required_clips", max_count=64)
    if (len(set(required_clips)) != len(required_clips)
            or any(not isinstance(name, str) or not name for name in required_clips)):
        raise ValueError("required_clips must be distinct, nonempty names")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"Benchmark revision already exists: {destination}")
    if destination in (character_dir, root):
        raise ValueError("Benchmark cannot overwrite its input directory")
    checks = {
        name: {"status": "not-required" if name not in required else "missing",
               "missing": [], "failures": [], "checked": 0, "items": []}
        for name in sorted(_REQUIRED_CHECKS)
    }

    seen = set()
    for entry in regions:
        rid = entry["region"]
        path = _path(root, entry.get("report"), f"{rid} region report")
        evidence = _report(path, f"{rid} region report")
        expected_views = set(verified[rid])
        views = evidence.get("views", [])
        if (evidence.get("character_id") != spec["character_id"]
                or evidence.get("region_id") != rid
                or evidence.get("source_spec_sha256") != plan["source_sha256"]
                or evidence.get("region_input_sha256") != plan["regions"][rid]["input_sha256"]
                or set(evidence.get("verified_views", [])) != expected_views
                or not isinstance(views, list)
                or {row.get("view") for row in views if isinstance(row, dict)} != expected_views
                or len(views) != len(expected_views)):
            raise ValueError(f"{rid}: source registration/report does not match frozen CharacterSpec")
        for mesh_name, sha_name in (("candidate_mesh", "candidate_sha256"),
                                    ("baseline_mesh", "baseline_sha256")):
            mesh = _path(path.parent, evidence.get(mesh_name), f"{rid} {mesh_name}")
            if _sha(mesh) != evidence.get(sha_name):
                raise ValueError(f"{rid}: original geometry changed since its source QA")
        if type(evidence.get("numeric_gate_pass")) is not bool:
            raise ValueError(f"{rid}: region gate lacks a boolean numeric result")
        fails = evidence.get("failed_views", [])
        if not isinstance(fails, list) or any(v not in expected_views for v in fails):
            raise ValueError(f"{rid}: invalid failed-view list")
        if evidence["numeric_gate_pass"] != (not fails):
            raise ValueError(f"{rid}: internally contradictory silhouette gate")
        for row in views:
            if any(key not in row or not isinstance(row[key], (int, float))
                   for key in ("candidate_iou", "baseline_iou", "candidate_boundary_f1_2px")):
                raise ValueError(f"{rid}: missing numeric source contour measurements")
        is_bad = bool(fails or any(row.get("rejection_reasons") for row in views))
        if is_bad != (not evidence["numeric_gate_pass"]):
            raise ValueError(f"{rid}: per-view rejection reasons contradict gate status")
        checks["regions"]["checked"] += 1
        seen.add(rid)
        checks["regions"]["items"].append({
            "region": rid, "report_sha256": _sha(path),
            "verified_views": sorted(expected_views), "numeric_gate_pass": not is_bad,
            "worst_iou_delta": evidence.get("worst_iou_delta"),
            "worst_contour_delta": min(row.get("boundary_f1_2px_delta", 0) for row in views),
            "artistic_review_required": True,
        })
        if is_bad:
            checks["regions"]["failures"].append(rid)
    checks["regions"]["missing"] = sorted(set(region_ids) - seen)
    if not region_ids:
        checks["regions"]["missing"].append("no-verified-source-regions")

    for entry in textures:
        tid = entry["id"]
        path = _path(root, entry.get("report"), f"{tid} texture report")
        evidence = _report(path, f"{tid} texture report")
        hashes = evidence.get("input_sha256")
        supplied = entry.get("inputs")
        if (not isinstance(hashes, dict) or not isinstance(supplied, dict)
                or set(hashes) != set(supplied) or not hashes):
            raise ValueError(f"{tid}: texture QA needs explicit matching source paths for each input hash")
        for key, value in supplied.items():
            item = _path(root, value, f"{tid}/{key}")
            if _sha(item) != hashes[key]:
                raise ValueError(f"{tid}: input {key} changed since texture QA")
        names = evidence.get("source_view_names", [])
        rows = evidence.get("views", [])
        if (not isinstance(names, list) or not names or len(names) != len(set(names))
                or any(name not in set().union(*[set(v) for v in verified.values()]) for name in names)
                or not isinstance(rows, list) or len(rows) != len(names)
                or {row.get("view") for row in rows} != set(names)
                or type(evidence.get("numeric_gate_pass")) is not bool
                or evidence["numeric_gate_pass"] == bool(evidence.get("failed_views"))):
            raise ValueError(f"{tid}: texture camera coverage or result contradicts benchmark source")
        bad = not evidence["numeric_gate_pass"]
        checks["textures"]["checked"] += 1
        checks["textures"]["items"].append({
            "id": tid, "report_sha256": _sha(path), "source_views": names,
            "numeric_gate_pass": not bad,
            "max_projection_samples": evidence.get("max_projection_samples"),
        })
        if bad:
            checks["textures"]["failures"].append(tid)
    if not textures:
        checks["textures"]["missing"].append("no-source-verified-textures")

    pose_entry = raw.get("pose_report")
    if pose_entry is not None:
        if not isinstance(pose_entry, dict) or set(pose_entry) != {"report", "blend"}:
            raise ValueError("pose_report requires report and source blend paths")
        path = _path(root, pose_entry["report"], "pose report")
        blend = _path(root, pose_entry["blend"], "pose source .blend")
        evidence = _report(path, "pose report")
        # Older pose reports without embedded original .blend SHA are
        # intentionally untrusted: rerun native validate-rig after update.
        if (evidence.get("source_blend_sha256") != _sha(blend)
                or evidence.get("source_blend") != str(blend)):
            raise ValueError("Pose QA report is stale or lacks source-locked .blend provenance")
        rest = evidence.get("rest_skin_weights", {})
        poses = evidence.get("poses", [])
        bad = (evidence.get("status") == "structural-fail" or bool(evidence.get("problems"))
               or not isinstance(rest, dict) or not rest or not isinstance(poses, list)
               or not poses or any(
                   any(type(weights.get(k)) is not int or weights[k] != 0
                       for k in ("unweighted", "over_four_influences", "non_normalized"))
                   for weights in rest.values()
               ))
        weight_zones = _list(evidence.get("rest_weight_zones", []),
                             "pose rest_weight_zones", max_count=64)
        failed_weight_zones = []
        for zone in weight_zones:
            if (not isinstance(zone, dict)
                    or not isinstance(zone.get("mesh"), str)
                    or not isinstance(zone.get("region_group"), str)
                    or type(zone.get("selected_vertices")) is not int
                    or type(zone.get("violating_vertices")) is not int
                    or zone.get("status") not in {"pass", "fail"}):
                raise ValueError("Pose report has malformed semantic weight-zone evidence")
            if (zone["status"] != "pass" or zone["violating_vertices"] > 0
                    or zone["selected_vertices"] < zone.get("minimum_required_vertices", 1)):
                bad = True
                failed_weight_zones.append(f"{zone['mesh']}/{zone['region_group']}")
        known_pose_names = [row.get("name") for row in poses if isinstance(row, dict)]
        if len(known_pose_names) != len(set(known_pose_names)):
            raise ValueError("Pose names are missing or duplicated")
        render_count = 0
        for row in poses:
            if "image" in row:
                _path(path.parent, row["image"], f"pose {row['name']} screenshot")
                render_count += 1
            for metrics in row.get("meshes", {}).values():
                if metrics.get("topology_changed") or metrics.get("status") == "fail":
                    bad = True
        required_poses = _list(raw.get("required_poses", []), "required_poses")
        missing_poses = sorted(set(required_poses) - set(known_pose_names))
        checks["pose"]["checked"] = 1
        checks["pose"]["items"] = [{
            "report_sha256": _sha(path), "blend_sha256": _sha(blend),
            "pose_names": known_pose_names, "screenshot_count": render_count,
            "skinned_meshes": sorted(rest),
            "weight_zone_count": len(weight_zones),
            "failed_weight_zones": failed_weight_zones,
            "collision_candidates": sum(
                len(row.get("collision_candidates", [])) for row in poses),
            "independent_artistic_review_required": True,
        }]
        if bad:
            checks["pose"]["failures"].append("structural-or-skinning")
        checks["pose"]["missing"].extend(missing_poses)
    else:
        checks["pose"]["missing"].append("no-source-locked-pose-report")

    glb_path = raw.get("glb")
    if glb_path is not None:
        path = _path(root, glb_path, "portable GLB")
        inventory = inspect_glb(path, required_clips=required_clips)
        checks["glb"]["checked"] = 1
        checks["glb"]["items"] = [{"file": str(path), **inventory}]
        if not inventory["numeric_ready"]:
            checks["glb"]["failures"].append("missing-skinned-animation-or-clips")
    else:
        checks["glb"]["missing"].append("no-portable-animated-glb")

    observations = _list(raw.get("runtime_observations", []), "runtime_observations")
    for entry in observations:
        if not isinstance(entry, dict) or set(entry) != {
                "stage", "wall_time_s", "cpu_time_s", "peak_rss_mb", "machine"}:
            raise ValueError("Runtime observations require stage, wall/cpu time, peak RSS and machine")
        if not isinstance(entry["stage"], str) or not isinstance(entry["machine"], str):
            raise ValueError("Invalid runtime stage/machine")
        for label in ("wall_time_s", "cpu_time_s", "peak_rss_mb"):
            v = entry[label]
            if type(v) not in (float, int) or not math.isfinite(v) or v < 0:
                raise ValueError(f"Invalid runtime observation {label}")
    for name, value in checks.items():
        if name not in required:
            value["status"] = "not-required"
        else:
            value["status"] = ("fail" if value["failures"] else "missing"
                               if value["missing"] else "numeric-pass-review-required")
    missing = [f"{name}:{entry}" for name in required for entry in checks[name]["missing"]]
    failures = [f"{name}:{entry}" for name in required for entry in checks[name]["failures"]]
    status = ("numeric-gate-failed" if failures else "missing-required-evidence" if missing
              else "numeric-gates-passed-artistic-review-required")
    if _sha(manifest) != manifest_hash:
        raise ValueError("Benchmark manifest was modified during evidence evaluation")
    # Revalidate CharacterSpec (including every original source image hash).
    checked_plan, _ = open_plan(character_dir)
    if checked_plan != plan:
        raise ValueError("Character sources changed during benchmark")
    result = {
        "schema_version": 1,
        "character_id": spec["character_id"],
        "status": status,
        "numeric_gates_pass": status == "numeric-gates-passed-artistic-review-required",
        "artistic_review_required": True,
        "source_character_plan": str(character_dir),
        "source_sha256": plan["source_sha256"],
        "manifest_sha256": manifest_hash,
        "required_checks": sorted(required),
        "required_regions": region_ids,
        "checks": checks,
        "missing_evidence": sorted(missing),
        "numeric_failures": sorted(failures),
        "runtime_observations": observations,
        "runtime_observations_source": "externally_supplied_not_measured_by_benchmark",
        "benchmark_processing_seconds": round(time.perf_counter() - started, 6),
        "limitations": [
            "This collector does not build geometry, run Blender or measure stage CPU/RSS itself.",
            "An external runtime observation is reported, never presented as directly measured.",
            "A numeric pass never certifies Pip/Quill likeness, source depth, real-world animation or material quality.",
            "Pose contact candidates remain advisory and still require visual inspection.",
            "For texture reports, the caller must provide all originally hashed source paths.",
        ],
    }
    dest = Path(output).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".character-benchmark-", dir=dest.parent))
    try:
        (temp / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (temp / "SUMMARY.md").write_text(_render_summary(result), encoding="utf-8")
        temp.rename(dest)
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return result
