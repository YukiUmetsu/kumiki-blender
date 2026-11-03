#!/usr/bin/env python3
"""Actionable, read-only image value and camera-angle gap audit for COLMAP.

Distinguishes observed SfM evidence from hypotheses about missing viewpoints.
Unregistered photos have NO measured camera angle; image semantics require
optional human/vision-supplied annotations. No input files are modified.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Missing analysis script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _norm(vector):
    length = math.sqrt(sum(x * x for x in vector))
    if length < 1e-9 or not math.isfinite(length):
        return None
    return tuple(x / length for x in vector)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _angle(a, b):
    return math.degrees(math.acos(max(-1., min(1., _dot(a, b)))))


def _camera_up(q):
    """Image-up direction in world coordinates from COLMAP world-to-camera q."""
    length = math.sqrt(sum(v*v for v in q))
    if length < 1e-9:
        return None
    w, x, y, z = (v/length for v in q)
    # Negative row 1 of R (camera image +Y points downward).
    return _norm((-2*(x*y + z*w), -(1 - 2*(x*x + z*z)),
                  -2*(y*z - x*w)))


def _poses(model: Path, geometry) -> dict[str, dict]:
    known = geometry._images(model)
    poses = {}
    with (model / "images.txt").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10 or int(parts[0]) not in known:
                raise ValueError("Malformed COLMAP image pose")
            image = known[int(parts[0])]
            poses[image["name"]] = {
                "center": image["center"],
                "up": _camera_up(tuple(float(x) for x in parts[1:5])),
            }
            next(stream, None)
    return poses


def _centroid(model: Path) -> tuple[float, ...] | None:
    """Robust median of reconstructed surface points; not the true statue center."""
    path = model / "points3D.txt"
    if not path.is_file() or path.is_symlink():
        return None
    points = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8:
                continue
            try:
                point = tuple(float(x) for x in fields[1:4])
            except ValueError:
                continue
            if all(math.isfinite(x) for x in point):
                points.append(point)
            if len(points) >= 100_000:
                break
    if len(points) < 10:
        return None
    return tuple(statistics.median(p[i] for p in points) for i in range(3))


def _relative_angles(poses: dict, center: tuple[float, ...],
                     anchor: str) -> dict[str, dict]:
    origin = _norm(_sub(poses[anchor]["center"], center))
    up = poses[anchor]["up"]
    if origin is None or up is None:
        return {}
    up = _norm(tuple(up[i] - _dot(up, origin)*origin[i] for i in range(3)))
    if up is None:
        return {}
    right = _norm(_cross(up, origin))
    if right is None:
        return {}
    angles = {}
    for name, pose in poses.items():
        ray = _norm(_sub(pose["center"], center))
        if ray is None:
            continue
        yaw = math.degrees(math.atan2(_dot(ray, right), _dot(ray, origin)))
        elevation = math.degrees(math.asin(max(-1, min(1, _dot(ray, up)))))
        angles[name] = {
            "relative_azimuth_deg": round(yaw, 1),
            "relative_elevation_deg": round(elevation, 1),
            "angular_separation_from_anchor_deg": round(_angle(ray, origin), 1),
        }
    return angles


def _view_label(yaw: float, front_known: bool) -> str:
    if not front_known:
        return "relative to reference camera (not a verified front/side label)"
    magnitude = abs(yaw)
    side = "right" if yaw > 0 else "left"
    if magnitude <= 12:
        return "front"
    if magnitude <= 65:
        return f"{side} three-quarter"
    if magnitude <= 115:
        return f"{side} profile"
    return f"{side} rear three-quarter or rear"


def _shot_plan(angles: dict[str, dict], *, front_known: bool,
               anchor: str | None, weak_track_count: int,
               missing_count: int, subject_type: str) -> list[dict]:
    """Suggest near-term adjacent views, never fabricate an unregistered pose."""
    landmarks = {
        "statue": "shared head, neck, torso and base landmarks",
        "building": "shared facade, corners, windows and adjacent walls",
        "landscape": "shared ridges, rock formations and stable terrain landmarks",
        "object": "shared distinctive object features and silhouette",
    }
    medium = {
        "statue": "medium-distance, head/neck/upper torso together",
        "building": "medium-distance, facade corner with adjacent walls",
        "landscape": "medium-wide terrain with overlapping stable landmarks",
        "object": "medium-distance, whole object plus identifiable details",
    }[subject_type]
    base = (f"Keep the same subject region in 60–80% overlap with adjacent "
            f"photos; consistent light, focus, distance and focal length. "
            f"Include {landmarks[subject_type]} where visible.")
    result = []
    if not angles:
        return [
            {"priority": 1, "kind": "overlap-bridge", "target_azimuth_deg": None,
             "target_elevation_deg": None, "view": "unknown until cameras register",
             "framing": medium,
             "reason": "No reliable same-model camera positions exist to calculate angle gaps. "
                       "Collect a continuous 10–15 degree sequence around one visible side.",
             "capture_guidance": base, "evidence": "generic-capture-guidance"},
            {"priority": 2, "kind": "detail-and-context", "target_azimuth_deg": None,
             "target_elevation_deg": None, "view": "unknown until cameras register",
             "framing": "alternate close detail and medium context views with shared landmarks",
             "reason": "Ensure detailed photos overlap the wider reconstruction.",
             "capture_guidance": base, "evidence": "generic-capture-guidance"},
        ]
    yaws = sorted(item["relative_azimuth_deg"] for item in angles.values())
    elevations = [item["relative_elevation_deg"] for item in angles.values()]
    candidates = []
    # Bridge measured gaps *between* known cameras first; do not assume
    # a 360-degree orbit is accessible or that the subject has a visible rear.
    for left, right in zip(yaws, yaws[1:]):
        gap = right - left
        if 25 <= gap <= 150:
            steps = math.ceil(gap / 15)
            for step in range(1, steps):
                target = left + gap * step / steps
                candidates.append((0, target, 0., "bridge the gap between registered views"))
    # Expand outward gradually, only from the measured edge of the orbit.
    for yaw in (yaws[0] - 15, yaws[-1] + 15, yaws[0] - 30, yaws[-1] + 30):
        if -150 <= yaw <= 150:
            candidates.append((1, yaw, 0., "extend the registered viewpoint sequence"))
    # Capture elevation diversity only after a base overlapping orbit.
    if max(elevations) - min(elevations) < 15:
        candidates.append((2, statistics.median(yaws), 20.,
                           "add a slightly elevated overlapping view"))
        candidates.append((2, statistics.median(yaws), -15.,
                           "add a lower overlapping view, if physically accessible"))
    seen = set()
    for order, yaw, elevation, why in sorted(candidates):
        key = (round(yaw / 5), round(elevation / 5))
        if key in seen:
            continue
        seen.add(key)
        # Ignore directions already represented by a nearby reconstructed pose.
        if any(math.hypot(yaw-a["relative_azimuth_deg"],
                          elevation-a["relative_elevation_deg"]) < 12
               for a in angles.values()):
            continue
        result.append({
            "priority": len(result) + 1,
            "kind": "overlap-bridge" if order < 2 else "elevation-coverage",
            "target_azimuth_deg": round(yaw, 1),
            "target_elevation_deg": round(elevation, 1),
            "view": _view_label(yaw, front_known),
            "framing": medium if weak_track_count else
                       "same scale and region as adjacent registered photos",
            "reason": why + ("; weak 2D-to-3D track support is widespread"
                              if weak_track_count else ""),
            "capture_guidance": base,
            "evidence": "estimated-colmap-pose / suggested-target-not-measured",
        })
        if len(result) >= 6:
            break
    if missing_count and not result:
        result.append({
            "priority": 1, "kind": "detail-and-context",
            "target_azimuth_deg": None, "target_elevation_deg": None,
            "view": "same region as an existing registered photo",
            "framing": "medium context + overlapping close detail",
            "reason": "Existing registered camera angles are already clustered near the sampled targets; "
                      "prioritize bridging detailed and context photos.",
            "capture_guidance": base, "evidence": "registration-support-diagnostic",
        })
    return result


def _track_contributions(model: Path, pnp) -> dict[str, dict]:
    """Count observations and fragile two-view tracks per registered camera.

    A two-view track would cease to be triangulated if either of its two
    supporting cameras were removed. This is a lower-bound proxy for marginal
    reconstruction value, not a leave-one-out re-optimization.
    """
    registered, tracks = pnp._model_tracks(model)
    lengths = Counter(tracks.values())
    observations = Counter()
    fragile = Counter()
    for (image_id, _), point_id in tracks.items():
        observations[image_id] += 1
        if lengths[point_id] == 2:
            fragile[image_id] += 1
    return {
        name: {
            "observed_3d_tracks": observations[image_id],
            "two_view_tracks_lost_if_camera_removed": fragile[image_id],
            "measurement": "existing COLMAP tracks; not a leave-one-out SfM rerun",
        }
        for image_id, name in registered.items()
    }


def _status(photo: dict, *, registered: bool, alternate: bool,
            candidate: dict | None, near_angle: float | None,
            contribution: dict | None) -> tuple[str, str, list[str]]:
    reasons = []
    if not photo["imported"]:
        return "cannot-use", "high", ["Not imported by COLMAP; fix decoding/staging before judging viewpoint."]
    if photo["keypoints"] < 100:
        reasons.append(f"Only {photo['keypoints']} SIFT keypoints; inspect sharpness, crop, texture and masking.")
    if registered:
        reasons.append("Registered in the largest single sparse model; currently contributes a camera pose.")
        if contribution:
            reasons.append(f"Observes {contribution['observed_3d_tracks']} existing 3D tracks; "
                           f"{contribution['two_view_tracks_lost_if_camera_removed']} "
                           "two-view tracks would lose triangulation if this camera were removed.")
        if near_angle is not None and near_angle < 4 and contribution and (
                contribution["two_view_tracks_lost_if_camera_removed"] == 0):
            reasons.append(f"Another registered view is only {near_angle:.1f} degrees away: "
                           "possible viewpoint redundancy, but texture/track value is not measured by angle alone.")
            return "keep-check-overlap", "medium", reasons
        return "keep", "high", reasons
    if alternate:
        reasons.append("Registered in a separate sparse model; preserve it, but models are not aligned.")
        return "keep-separate-model", "high", reasons
    if photo["strong_neighbors"] == 0:
        reasons.append("No strong geometrically verified neighbor; this photo currently adds no connected SfM view.")
        return "low-current-value", "high", reasons
    if candidate is not None:
        count = candidate["conservative_one_to_one_2d3d"]
        coverage = candidate["one_to_one_4x4_image_coverage"]
        reasons.append(f"{count} unverified one-to-one 2D–3D candidate associations "
                       f"({coverage if coverage is not None else 'unknown'} 4×4 grid coverage).")
        if count >= 24 and coverage is not None and coverage >= 0.1875:
            return "recover-first", "medium", reasons
        if count >= 8:
            reasons.append("Some shared tracks exist, but spatial coverage or count may be insufficient for PnP.")
            return "needs-overlap-bridge", "medium", reasons
    reasons.append("Weak or missing connections to reconstructed 3D tracks; "
                   "its exact viewing angle is unknown until registration.")
    return "low-current-value", "medium", reasons


def analyze(revision: Path, *, source_index: Path | None = None,
            front_photo: str | None = None, annotations: Path | None = None,
            vision_model: str | None = None, subject_type: str = "object",
            vision_endpoint: str = "http://127.0.0.1:11434/api/generate") -> dict:
    if subject_type not in ("object", "statue", "building", "landscape"):
        raise ValueError("subject_type must be object, statue, building or landscape")
    registration = _load_script("diagnose_splat_registration")
    pnp = _load_script("diagnose_sfm_pnp_support")
    geometry = _load_script("analyze_sfm_geometry")
    revision = revision.resolve()
    sparse = revision / "sparse-reconstruction" if (
        revision / "sparse-reconstruction").is_dir() else revision
    graph = registration.analyze(revision, 15)
    support = pnp.analyze(revision, source_index)
    original = pnp._source_names(sparse, source_index)
    if annotations:
        tags = json.loads(annotations.read_text(encoding="utf-8"))
        if not isinstance(tags, dict):
            raise ValueError("Photo annotations must be a JSON object keyed by filename")
    else:
        tags = {}
    models = graph["sparse_models"]
    best = max(models, key=lambda m: (m["registered"], m["model"]), default=None)
    best_name = best["model"] if best else None
    best_files = set(best["filenames"]) if best else set()
    alternate = {name for model in models if model["model"] != best_name
                 for name in model["filenames"]}
    pnp_model = next((m for m in support["models"] if m["model"] == best_name), None)
    candidates = {item["filename"]: item for item in pnp_model["unregistered_images"]} if pnp_model else {}
    contributions = (_track_contributions(sparse / "text-models" / best_name, pnp)
                     if best_name is not None else {})
    poses = {}
    centroid = None
    angles = {}
    anchor = None
    front_known = False
    if best_name is not None:
        folder = sparse / "text-models" / best_name
        poses = _poses(folder, geometry)
        centroid = _centroid(folder)
        if centroid is not None and len(poses) >= 2:
            if front_photo:
                for name in sorted(poses):
                    if front_photo in (name, original.get(name)):
                        anchor, front_known = name, True
                        break
            if anchor is None:
                # No front annotation: use a stable reference camera, not a
                # fabricated cardinal direction.
                anchor = sorted(poses)[0]
            angles = _relative_angles(poses, centroid, anchor)
    vision = {}
    if vision_model:
        classifier = _load_script("classify_photo_views")
        vision = classifier.classify_images(
            sparse, [photo["filename"] for photo in graph["images"]],
            model=vision_model, endpoint=vision_endpoint)
    images = []
    for photo in graph["images"]:
        name = photo["filename"]
        original_name = original.get(name, name)
        measured = angles.get(name)
        closest = min((math.hypot(measured["relative_azimuth_deg"] - other["relative_azimuth_deg"],
                                  measured["relative_elevation_deg"] - other["relative_elevation_deg"])
                       for other_name, other in angles.items() if other_name != name),
                      default=None) if measured else None
        status, confidence, reasons = _status(
            {**photo, "imported": True}, registered=name in best_files,
            alternate=name in alternate, candidate=candidates.get(name),
            near_angle=closest, contribution=contributions.get(name))
        annotation = tags.get(name, tags.get(original_name))
        if annotation is not None and (not isinstance(annotation, dict) or
                                       any(not isinstance(value, str) or len(value) > 120
                                           for value in annotation.values())):
            raise ValueError(f"Invalid annotation for {name}")
        images.append({
            "filename": name,
            "original_filename": original_name,
            "assessment": status,
            "confidence": confidence,
            "registered_in_best_model": name in best_files,
            "registered_in_other_model": name in alternate,
            "keypoints": photo["keypoints"],
            "strong_neighbors": photo["strong_neighbors"],
            "verified_inliers_across_pairs": photo["verified_inliers_across_pairs"],
            "pnp_support": candidates.get(name),
            "track_contribution": contributions.get(name),
            "estimated_view": ({
                **measured, "label": _view_label(measured["relative_azimuth_deg"], front_known),
                "reference": "annotated-front" if front_known else "arbitrary-reference-camera",
            } if measured else None),
            "annotations": annotation,
            "vision_estimate": vision.get(name),
            "reasons": reasons,
            "next_action": (
                "Keep for this model; inspect for true duplicate views before removing."
                if status.startswith("keep") else
                "Retain and try seeded PnP/track recovery before collecting replacement photos."
                if status == "recover-first" else
                "Collect an intermediate photo with 60–80% shared subject overlap; "
                "do not assume this photo's angle is known."),
        })
    for missing in graph["missing_from_database"]:
        name = missing["filename"]
        images.append({
            "filename": name, "original_filename": original.get(name, name),
            "assessment": "cannot-use", "confidence": "high",
            "registered_in_best_model": False, "registered_in_other_model": False,
            "keypoints": 0, "strong_neighbors": 0, "verified_inliers_across_pairs": 0,
            "pnp_support": None, "track_contribution": None, "estimated_view": None,
            "annotations": tags.get(name, tags.get(original.get(name, name))),
            "vision_estimate": None,
            "reasons": ["Staged image was not imported by COLMAP."],
            "next_action": "Repair staging/import and rerun feature extraction.",
        })
    images.sort(key=lambda item: item["original_filename"])
    counts = dict(sorted(Counter(item["assessment"] for item in images).items()))
    required_registered = max(8, math.ceil(0.6 * len(images)))
    weak_tracks = sum(item["assessment"] in ("low-current-value", "needs-overlap-bridge")
                      for item in images)
    shots = _shot_plan(angles, front_known=front_known, anchor=anchor,
                       weak_track_count=weak_tracks,
                       missing_count=len(images) - len(best_files),
                       subject_type=subject_type)
    visual_counts = {}
    if vision:
        visual_counts = {
            field: dict(sorted(Counter(
                item[field] for item in vision.values()
                if item["source"] == "local-vision-model / not verified"
            ).items()))
            for field in ("view", "framing", "elevation", "quality")
        }
        view_counts = visual_counts["view"]
        framing_counts = visual_counts["framing"]
        elevation_counts = visual_counts["elevation"]
        successful_labels = sum(
            item["source"] == "local-vision-model / not verified" and
            (item["view"] != "unknown" or item["framing"] != "unknown")
            for item in vision.values())
        semantic_gaps = [
            ("head-and-torso" if subject_type == "statue" else "medium-context",
             "medium-distance context showing the whole subject and shared detail",
             "Vision labels found no medium-context shot bridging close details and wide views",
             not any(name in framing_counts for name in
                     (("head-and-torso", "medium-context") if subject_type == "statue"
                      else ("medium-context", "head-and-torso")))),
            ("side profiles", "head-and-shoulders side profile with shared landmarks",
             "Vision labels found no clear side profile",
             subject_type in ("object", "statue") and
             not any(name in view_counts for name in ("left-profile", "right-profile"))),
            ("elevated oblique", "medium-distance elevated oblique",
             "Vision labels found no elevated view; capture only where accessible",
             "elevated" not in elevation_counts),
        ]
        for view, framing, reason, needed in semantic_gaps:
            if needed and successful_labels >= max(2, math.ceil(0.7 * len(vision))):
                shots.append({
                    "priority": len(shots) + 1,
                    "kind": "vision-estimated-coverage-gap",
                    "target_azimuth_deg": None, "target_elevation_deg": None,
                    "view": view, "framing": framing, "reason": reason,
                    "capture_guidance": "Overlap existing registered views by 60–80%; "
                                        "the visual model has NOT measured camera angles.",
                    "evidence": "local-vision-model / unverified semantic estimate",
                })
    return {
        "schema_version": 1,
        "source_revision": str(revision),
        "subject_type": subject_type,
        "status": "evidence-based-photo-triage / not a ground-truth view classifier",
        "best_model": best_name,
        "photos": len(images),
        "registered_in_best_model": len(best_files),
        "camera_gate_met": len(best_files) >= required_registered,
        "required_registered_at_current_photo_count": required_registered,
        "additional_registrations_needed_at_current_photo_count":
            max(0, required_registered - len(best_files)),
        "assessments": counts,
        "angle_reference": {
            "filename": anchor, "original_filename": original.get(anchor, anchor) if anchor else None,
            "kind": "annotated-front" if front_known else
                    ("arbitrary-reference-camera" if anchor else "unavailable"),
            "requested_front_photo": front_photo,
            "requested_front_status": ("used" if front_known else
                                       "not-registered-in-best-model" if front_photo else
                                       "not-supplied"),
            "centroid_basis": "median of reconstructed 3D points; not verified object center",
            "limitations": "Relative azimuth/elevation come from estimated cameras in ONE sparse model. "
                           "Camera-image-up defines vertical; roll, focal length, incomplete surface "
                           "coverage and incorrect poses can bias these angles. "
                           "Unregistered photos have no measured angle.",
        },
        "images": images,
        "visual_coverage": {
            "status": "local-model-unverified" if vision else "not-run",
            "model": vision_model, "counts": visual_counts,
            "limitations": "Visual labels are guesses from a local vision model. "
                           "They do not measure angles or prove photo quality.",
        },
        "recommended_photos": shots,
        "capture_notes": [
            "Collect 10–15 degree neighboring views with 60–80% subject overlap; "
            "keep head, neck, and body landmarks in shared frames.",
            "Prioritize consistent, sharp, similarly exposed photographs of the actual subject; "
            "avoid cutouts, replicas, changing shadows and unrelated backgrounds.",
            "Do not delete a photo solely because it is currently unregistered: "
            "it may show unique geometry or belong to a separate useful cohort.",
            "Back/rear and low angles should be requested only where physically accessible. "
            "A verified scan or survey is preferable for inaccessible surfaces.",
        ],
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Photo value and missing-angle report",
        "",
        f"**Reconstruction:** {report['registered_in_best_model']}/{report['photos']} "
        f"registered in model {report['best_model']}; camera gate: "
        f"{'met' if report['camera_gate_met'] else 'not met'}. "
        f"At the current photo count, need "
        f"{report['additional_registrations_needed_at_current_photo_count']} "
        f"more registrations (target "
        f"{report['required_registered_at_current_photo_count']}).",
        "",
        "**Caution:** Low current value means weak evidence for this SfM run, "
        "not that the image is bad or its viewpoint is unnecessary. "
        "Unregistered image angles cannot be measured.",
        "",
        f"**Angle basis:** {report['angle_reference']['kind']} "
        f"({report['angle_reference']['original_filename'] or 'none'}).",
        "",
        "## Photos to review first",
        "",
    ]
    if report["angle_reference"]["requested_front_status"] == "not-registered-in-best-model":
        lines.append("**Front anchor unavailable:** Requested frontal photo is not "
                     "registered in the selected model; angles remain relative "
                     "to an arbitrary camera.")
        lines.append("")
    low = [p for p in report["images"]
           if p["assessment"] in ("low-current-value", "cannot-use")]
    for photo in low:
        lines.append(f"- **{photo['original_filename']}** ({photo['assessment']}): "
                     f"{photo['reasons'][-1]}")
    if not low:
        lines.append("- None have zero/weak evidence under the current thresholds.")
    overlap = [p["original_filename"] for p in report["images"]
               if p["assessment"] == "keep-check-overlap"]
    if overlap:
        lines.append("- **Possible duplicate angles (keep until verified):** " +
                     ", ".join(overlap))
    lines.extend(["", "## Priority new photos", ""])
    for shot in report["recommended_photos"]:
        angle = (f"azimuth {shot['target_azimuth_deg']:+.1f}°, "
                 f"elevation {shot['target_elevation_deg']:+.1f}°"
                 if shot["target_azimuth_deg"] is not None else "angle unmeasured")
        lines.append(f"{shot['priority']}. **{shot['view']}** ({angle}): "
                     f"{shot['framing']}. {shot['reason']}.")
    lines.extend([
        "",
        "## Photo-by-photo",
        "",
        "| Original photo | Assessment | Evidence | Angle | View/framing | Action |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for photo in report["images"]:
        name = photo["original_filename"].replace("|", "/")
        evidence = (f"{photo['strong_neighbors']} strong neighbors; "
                    f"{photo['keypoints']} keypoints")
        if photo["pnp_support"] is not None:
            evidence += f"; {photo['pnp_support']['conservative_one_to_one_2d3d']} 2D–3D candidates"
        if photo["track_contribution"]:
            contribution = photo["track_contribution"]
            evidence += (f"; {contribution['observed_3d_tracks']} observed 3D tracks, "
                         f"{contribution['two_view_tracks_lost_if_camera_removed']} critical two-view")
        view = photo["estimated_view"]
        angle = (f"{view['relative_azimuth_deg']:+.1f}° az, "
                 f"{view['relative_elevation_deg']:+.1f}° el" if view else "unmeasured")
        annotation = photo["annotations"] or {}
        visual = photo["vision_estimate"] or {}
        view_type = annotation.get("view") or visual.get("view") or "unknown"
        framing = annotation.get("framing") or visual.get("framing") or "unknown"
        kind = f"{view_type} / {framing}".replace("|", "/")
        if visual and not annotation:
            kind += " (vision estimate)"
        action = photo["next_action"].replace("|", "/")
        lines.append(f"| {name} | {photo['assessment']} | {evidence} | {angle} | {kind} | {action} |")
    lines.extend(["", "## Limits", "", report["angle_reference"]["limitations"], "",
                  *("- " + note for note in report["capture_notes"]), ""])
    return "\n".join(lines)


def _contact_sheet(revision: Path, report: dict, path: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps
    sparse = revision / "sparse-reconstruction" if (
        revision / "sparse-reconstruction").is_dir() else revision
    photos = report["images"]
    width, height, columns = 350, 254, 4
    canvas = Image.new("RGB", (width*columns, height*math.ceil(len(photos)/columns)),
                       (24, 28, 33))
    draw = ImageDraw.Draw(canvas)
    palette = {
        "keep": (85, 204, 130), "keep-check-overlap": (132, 210, 168),
        "keep-separate-model": (112, 180, 245), "recover-first": (112, 180, 245),
        "needs-overlap-bridge": (235, 191, 89), "low-current-value": (234, 149, 91),
        "cannot-use": (230, 90, 90),
    }
    for index, item in enumerate(photos):
        x, y = index % columns * width, index // columns * height
        photo = sparse / "images" / item["filename"]
        try:
            with Image.open(photo) as src:
                image = ImageOps.contain(ImageOps.exif_transpose(src).convert("RGB"),
                                         (width-20, 175))
                canvas.paste(image, (x+(width-image.width)//2,
                                     y+4+(175-image.height)//2))
        except (OSError, ValueError):
            draw.text((x+10, y+85), "Photo not readable", fill=(245, 125, 125))
        draw.rectangle((x+4, y+184, x+width-4, y+height-4), fill=(42, 47, 54))
        draw.text((x+10, y+188), item["original_filename"][:44], fill=(245, 245, 245))
        draw.text((x+10, y+207), item["assessment"][:36],
                  fill=palette.get(item["assessment"], (220, 220, 220)))
        view = item["estimated_view"]
        detail = (f"az {view['relative_azimuth_deg']:+.0f} el {view['relative_elevation_deg']:+.0f}"
                  if view else "angle unknown")
        pnp = item["pnp_support"]
        detail += f" | {pnp['conservative_one_to_one_2d3d']} 3D candidates" if pnp else ""
        visual = item["vision_estimate"]
        if visual and visual["view"] != "unknown":
            detail = f"{visual['view'][:24]} | " + detail
        draw.text((x+10, y+228), detail[:55], fill=(213, 213, 213))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def _angle_map(report: dict, path: Path) -> bool:
    """SVG orbit overview: reconstructed relative azimuths and *suggested* shots."""
    from xml.sax.saxutils import escape
    known = [item for item in report["images"] if item["estimated_view"]]
    if not known:
        return False
    center, radius = 320, 218
    def xy(degrees, distance):
        radians = math.radians(degrees)
        return (round(center + distance * math.sin(radians), 1),
                round(center - distance * math.cos(radians), 1))
    front = report["angle_reference"]["kind"] == "annotated-front"
    title = "Front-relative camera orbit" if front else "Reference-relative camera orbit"
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="690" viewBox="0 0 640 690">',
        '<rect width="640" height="690" fill="#161c24"/>',
        f'<text x="320" y="36" text-anchor="middle" fill="#f5f7fa" font-size="22">{title}</text>',
        '<text x="320" y="61" text-anchor="middle" fill="#bdc7d1" font-size="12">'
        'Estimated COLMAP poses only. Not surveyed. Unregistered photos have no angles.</text>',
        f'<circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="#627386" stroke-width="2"/>',
        f'<circle cx="{center}" cy="{center}" r="{radius/2}" fill="none" stroke="#3d4b59" stroke-dasharray="4 6"/>',
        '<line x1="320" y1="102" x2="320" y2="538" stroke="#3d4b59"/>',
        '<line x1="102" y1="320" x2="538" y2="320" stroke="#3d4b59"/>',
        '<circle cx="320" cy="320" r="9" fill="#8e9bac"/>',
    ]
    for degree, label in ((0, "FRONT" if front else "REFERENCE"),
                          (90, "+90°"), (-90, "-90°"), (180, "±180°")):
        x, y = xy(degree, radius+27)
        parts.append(f'<text x="{x}" y="{y+5}" fill="#e2e8f0" font-size="13" '
                     f'text-anchor="middle">{label}</text>')
    for shot in report["recommended_photos"]:
        az = shot["target_azimuth_deg"]
        if az is None:
            continue
        x, y = xy(az, radius)
        parts.append(f'<circle cx="{x}" cy="{y}" r="7" fill="none" '
                     f'stroke="#efb65d" stroke-width="3"/>')
    for item in known:
        view = item["estimated_view"]
        az = view["relative_azimuth_deg"]
        x, y = xy(az, radius)
        name = escape(item["original_filename"][:25])
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="#62d49a" stroke="#0d1720" stroke-width="2"/>')
        tx, ty = xy(az, radius-27)
        parts.append(f'<text x="{tx}" y="{ty}" fill="#f1f5f9" font-size="10" '
                     f'text-anchor="middle">{name}</text>')
    parts.extend([
        '<circle cx="95" cy="598" r="6" fill="#62d49a"/>',
        '<text x="110" y="602" fill="#dce4ed" font-size="13">Registered camera</text>',
        '<circle cx="95" cy="627" r="7" fill="none" stroke="#efb65d" stroke-width="3"/>',
        '<text x="110" y="631" fill="#dce4ed" font-size="13">Suggested overlapping shot</text>',
        '<text x="320" y="667" fill="#bdc7d1" text-anchor="middle" font-size="11">'
        'Angles relative to median reconstructed points; not true monument center.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Unused output directory (JSON, Markdown, photo contact sheet)")
    parser.add_argument("--source-index", type=Path, help="Optional cohort-to-original photo mapping")
    parser.add_argument("--front-photo", help="Known frontal staged or original filename (must be registered)")
    parser.add_argument("--annotations", type=Path,
                        help="Optional JSON keyed by filename: view/framing/notes tags; never guessed")
    parser.add_argument("--subject-type", default="object",
                        choices=("object", "statue", "building", "landscape"),
                        help="Choose object-specific framing guidance (default: object)")
    parser.add_argument("--vision-model",
                        help="Optional installed local Ollama vision model tag; photos stay on this Mac")
    parser.add_argument("--vision-endpoint", default="http://127.0.0.1:11434/api/generate",
                        help="Loopback Ollama /api/generate endpoint only")
    args = parser.parse_args()
    try:
        report = analyze(args.revision, source_index=args.source_index,
                         front_photo=args.front_photo, annotations=args.annotations,
                         vision_model=args.vision_model, subject_type=args.subject_type,
                         vision_endpoint=args.vision_endpoint)
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Output already exists: {args.output}")
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "photo-advice.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        (args.output / "photo-advice.md").write_text(_markdown(report), encoding="utf-8")
        _angle_map(report, args.output / "photo-angle-map.svg")
        try:
            _contact_sheet(args.revision.resolve(), report, args.output / "photo-advice.jpg")
        except ImportError:
            print("Pillow not installed: contact sheet skipped.")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Registered {report['registered_in_best_model']}/{report['photos']}; "
          f"camera gate met: {report['camera_gate_met']}; "
          f"need {report['additional_registrations_needed_at_current_photo_count']} "
          f"more registrations at current photo count")
    print("Photo assessment counts:", report["assessments"])
    print(f"Angle reference: {report['angle_reference']['kind']} "
          f"({report['angle_reference']['original_filename']})")
    print("Priority new photos:")
    for shot in report["recommended_photos"]:
        print(f"  {shot['priority']}. {shot['view']}: "
              f"az={shot['target_azimuth_deg']}°, el={shot['target_elevation_deg']}°; "
              f"{shot['reason']}")
    print(f"Report: {args.output / 'photo-advice.md'}")
    print(f"JSON: {args.output / 'photo-advice.json'}")
    if (args.output / "photo-angle-map.svg").is_file():
        print(f"Angle map: {args.output / 'photo-angle-map.svg'}")
    print(f"Contact sheet: {args.output / 'photo-advice.jpg'} (if Pillow installed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
