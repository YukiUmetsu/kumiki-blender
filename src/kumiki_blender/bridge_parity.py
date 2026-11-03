"""Portable, evidence-backed Blender ↔ Kumiki white-mask frame comparison.

This measures projected geometry/camera parity, never texture, light, or realism.
Receipts are intentionally tied to the *byte-identical* capture plan and GLB.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


class BridgeParityError(ValueError):
    """Capture metadata, provenance or silhouettes are untrustworthy."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plan(path: str | Path) -> dict[str, Any]:
    try:
        target = Path(path).resolve(strict=True)
        plan = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(plan, dict) or plan.get("schema_version") != 1:
            raise BridgeParityError("Expected parity plan schema_version 1")
        for key in ("asset_id", "story_id", "action_id"):
            value = plan.get(key)
            if not isinstance(value, str) or not value or len(value) > 150 or not all(c.isalnum() or c in "_-.:" for c in value):
                raise BridgeParityError(f"Invalid {key} in parity plan")
        if not isinstance(plan.get("model_sha256"), str) or len(plan["model_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in plan["model_sha256"]):
            raise BridgeParityError("plan.model_sha256 must be lowercase SHA-256")
        frames = plan.get("frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 24 or any(type(x) is not int or not 0 <= x <= 100000 for x in frames) or sorted(set(frames)) != frames:
            raise BridgeParityError("plan.frames must be 1–24 distinct ascending nonnegative integers")
        width, height = plan.get("width"), plan.get("height")
        if type(width) is not int or type(height) is not int or not 64 <= width <= 4096 or not 64 <= height <= 4096 or width * height > 8_388_608:
            raise BridgeParityError("Plan image dimensions must be 64–4096 and at most 8 megapixels")
        preset = plan.get("camera_preset")
        if not isinstance(preset, dict) or preset.get("schemaVersion") != 1 or preset.get("mode") != "replace-shot":
            raise BridgeParityError("Plan requires camera preset schemaVersion 1, replace-shot")
        camera = preset.get("camera")
        if not isinstance(camera, dict) or camera.get("space") != "world" or not isinstance(camera.get("projection"), dict) or camera["projection"].get("kind") != "perspective":
            raise BridgeParityError("Blender parity currently supports world-space perspective cameras only")
        vec = camera.get("target")
        if not isinstance(vec, list) or len(vec) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in vec):
            raise BridgeParityError("camera.target must be three finite world-space numbers")
        limits = {"azimuthDeg": (-360, 360), "elevationDeg": (-89, 89), "heightOffset": (-1000, 1000),
                  "distanceFactor": (.05, 1000), "rollDeg": (-180, 180)}
        for key, (low, high) in limits.items():
            val = camera.get(key)
            if type(val) not in (int, float) or not math.isfinite(val) or not low <= val <= high:
                raise BridgeParityError(f"Invalid camera.{key}")
        fov = camera["projection"].get("fovDeg")
        if type(fov) not in (int, float) or not math.isfinite(fov) or not 5 <= fov <= 150:
            raise BridgeParityError("Invalid camera FOV")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeParityError(f"Cannot read parity plan: {error}") from error
    return plan


def camera_pose(plan: dict[str, Any]) -> tuple[tuple[float, float, float], tuple[float, float, float], float, float]:
    """Compute a Z-up Blender camera from the Y-up Three.js world-space preset.

    Blender glTF export maps (x,y,z) to (x,z,-y). Inverse is (x,-z,y).
    """
    c = plan["camera_preset"]["camera"]
    az, el = math.radians(c["azimuthDeg"]), math.radians(c["elevationDeg"])
    distance = c["distanceFactor"]
    tx, ty, tz = c["target"]
    three_position = (tx + math.sin(az) * math.cos(el) * distance,
                      ty + math.sin(el) * distance + c["heightOffset"],
                      tz + math.cos(az) * math.cos(el) * distance)
    blender = lambda v: (v[0], -v[2], v[1])
    return blender(three_position), blender((tx, ty, tz)), c["projection"]["fovDeg"], c["rollDeg"]


def frame_name(frame: int) -> str:
    return f"frame-{frame:05d}.png"


def verify_native_source(plan: dict[str, Any], source_blend: str | Path,
                         exported_model: str | Path) -> dict[str, str]:
    """Require native geometry to come from the same authored physics revision.

    A user-supplied `.blend` is *not* sufficient provenance on its own: a
    mismatched source can produce valid-looking but meaningless mask scores.
    The revision's exported GLB must be byte-identical to the web fixture.
    """
    requested_source = Path(source_blend)
    if requested_source.is_symlink():
        raise BridgeParityError("Native source.blend must not be a symbolic link")
    try:
        source = requested_source.resolve(strict=True)
        model = Path(exported_model).resolve(strict=True)
    except OSError as error:
        raise BridgeParityError(f"Cannot locate native revision input: {error}") from error
    if not source.is_file() or source.name != "source.blend" or not model.is_file():
        raise BridgeParityError("Native capture requires a saved source.blend and actual GLB")
    root = source.parent
    manifest_file = root / "manifest.json"
    if manifest_file.is_symlink():
        raise BridgeParityError("Symlinked native revision manifests are not trusted")
    try:
        if manifest_file.stat().st_size > 1_048_576:
            raise BridgeParityError("Native revision manifest exceeds 1 MiB")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeParityError(f"Cannot read native physics revision: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("generator") != "blender-physics-shot-bake":
        raise BridgeParityError("Native source must belong to a kumiki-blender physics bake revision")
    if manifest.get("asset_id") != plan["asset_id"] or not isinstance(manifest.get("physics"), dict) or manifest["physics"].get("action_id") != plan["action_id"]:
        raise BridgeParityError("Native source revision has different asset/action IDs")
    duration = manifest["physics"].get("duration_frames")
    if type(duration) is not int or duration < plan["frames"][-1]:
        raise BridgeParityError("Native physics bake does not cover requested parity frames")
    files, sizes = manifest.get("files"), manifest.get("output_bytes")
    if not isinstance(files, dict) or not isinstance(sizes, dict) or files.get("source") != "source.blend":
        raise BridgeParityError("Native revision must name its source.blend")
    model_name = files.get("model")
    if not isinstance(model_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.glb", model_name):
        raise BridgeParityError("Invalid native revision GLB filename")
    revision_model = root / model_name
    if revision_model.is_symlink() or not revision_model.is_file():
        raise BridgeParityError("Native revision GLB is missing or symlinked")
    if sizes.get("source.blend") != source.stat().st_size or sizes.get(model_name) != revision_model.stat().st_size:
        raise BridgeParityError("Native revision source/model sizes no longer match its manifest")
    model_digest = sha256(model)
    if model_digest != plan["model_sha256"] or sha256(revision_model) != model_digest:
        raise BridgeParityError("Native revision GLB and Kumiki model do not contain identical geometry bytes")
    return {"revision_manifest_sha256": sha256(manifest_file),
            "source_blend_sha256": sha256(source), "model_sha256": model_digest}


def make_receipt(plan_path: Path, model_path: Path, output: Path, renderer: str) -> dict[str, Any]:
    plan = load_plan(plan_path)
    if sha256(model_path) != plan["model_sha256"]:
        raise BridgeParityError("GLB bytes do not match plan.model_sha256")
    if renderer not in ("blender", "kumiki-webgl"):
        raise BridgeParityError("Unknown renderer")
    files: dict[str, str] = {}
    for frame in plan["frames"]:
        name = frame_name(frame)
        path = output / name
        if not path.is_file() or path.is_symlink():
            raise BridgeParityError(f"Missing/unsafe frame {name}")
        files[name] = sha256(path)
    return {"schema_version": 1, "renderer": renderer, "plan_sha256": sha256(plan_path),
            "model_sha256": plan["model_sha256"], "frames": list(plan["frames"]),
            "width": plan["width"], "height": plan["height"], "files": files}


def _receipt(folder: Path, plan: dict[str, Any], plan_digest: str, expected: str) -> dict[str, Any]:
    path = folder / "capture.json"
    if path.is_symlink():
        raise BridgeParityError("Symlinked receipts are not accepted")
    try:
        if path.stat().st_size > 1_048_576:
            raise BridgeParityError(f"{expected} receipt exceeds 1 MiB")
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeParityError(f"Cannot read {expected} receipt: {error}") from error
    if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in (
        ("schema_version", 1), ("renderer", expected), ("plan_sha256", plan_digest),
        ("model_sha256", plan["model_sha256"]), ("frames", plan["frames"]),
        ("width", plan["width"]), ("height", plan["height"]))):
        raise BridgeParityError(f"{expected} receipt differs from plan or renderer")
    if not isinstance(receipt.get("files"), dict) or set(receipt["files"]) != {frame_name(frame) for frame in plan["frames"]}:
        raise BridgeParityError(f"{expected} receipt has missing or unexpected frame names")
    for name, claimed in receipt["files"].items():
        frame = folder / name
        if frame.is_symlink() or not frame.is_file() or claimed != sha256(frame):
            raise BridgeParityError(f"{expected} frame changed since capture: {name}")
    return receipt


def _mask(image: Any, *, threshold: int) -> Any:
    return image.convert("RGB").convert("L").point(lambda value: 255 if value >= threshold else 0, mode="1").convert("L")


def _metrics(blender: Any, web: Any) -> tuple[dict[str, Any], Any]:
    from PIL import Image, ImageChops, ImageStat
    if blender.size != web.size:
        raise BridgeParityError(f"Frame dimensions differ: {blender.size} != {web.size}")
    b, w = _mask(blender, threshold=128), _mask(web, threshold=128)
    bi, wi = ImageStat.Stat(b).sum[0] / 255, ImageStat.Stat(w).sum[0] / 255
    area = blender.width * blender.height
    if not .0005 * area < bi < .98 * area or not .0005 * area < wi < .98 * area:
        raise BridgeParityError("Frame silhouette is empty/nearly full; cannot claim parity")
    overlap = ImageChops.multiply(b, w)
    intersect = ImageStat.Stat(overlap).sum[0] / 255
    iou = intersect / (bi + wi - intersect)
    b_box, w_box = b.getbbox(), w.getbbox()
    assert b_box is not None and w_box is not None
    b_center = ((b_box[0]+b_box[2])/2, (b_box[1]+b_box[3])/2)
    w_center = ((w_box[0]+w_box[2])/2, (w_box[1]+w_box[3])/2)
    centroid_delta = math.dist(b_center, w_center) / math.hypot(*blender.size)
    missing = ImageChops.subtract(b, w)
    unexpected = ImageChops.subtract(w, b)
    overlay = Image.merge("RGB", (missing, overlap, unexpected))
    return ({"iou": round(iou, 6), "center_delta_normalized": round(centroid_delta, 6),
             "blender_coverage": round(bi / area, 6), "web_coverage": round(wi / area, 6),
             "blender_bbox": b_box, "web_bbox": w_box}, overlay)


def compare_captures(plan_file: str | Path, blender_dir: str | Path, web_dir: str | Path,
                     output: str | Path, *, min_iou: float = .90, max_center_delta: float = .035) -> dict[str, Any]:
    """Compare immutable receipts; red=missing in WebGL, blue=extra, green=both.

    A visual pass is geometric agreement only; an artist must still review PBR and motion.
    """
    if any(type(n) not in (float, int) or not math.isfinite(n) or not 0 <= n <= 1 for n in (min_iou, max_center_delta)):
        raise BridgeParityError("Thresholds must be finite fractions [0,1]")
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise BridgeParityError("Install Pillow (kumiki-blender[qa]) for silhouette comparison") from error
    try:
        plan_path = Path(plan_file).resolve(strict=True)
        native_dir, browser_dir = Path(blender_dir).resolve(strict=True), Path(web_dir).resolve(strict=True)
    except OSError as error:
        raise BridgeParityError(f"Cannot locate parity input: {error}") from error
    plan = load_plan(plan_path)
    requested_output = Path(output)
    if requested_output.is_symlink():
        raise BridgeParityError("Comparison output cannot be a symbolic link")
    destination = requested_output.resolve()
    if destination.exists() or any(destination.is_relative_to(root) for root in (native_dir, browser_dir)):
        raise BridgeParityError("Comparison output must be new and outside capture directories")
    _receipt(native_dir, plan, sha256(plan_path), "blender")
    _receipt(browser_dir, plan, sha256(plan_path), "kumiki-webgl")
    results = []
    overlays = []
    for frame in plan["frames"]:
        name = frame_name(frame)
        try:
            with Image.open(native_dir / name) as image:
                if image.size != (plan["width"], plan["height"]):
                    raise BridgeParityError(f"Native capture dimensions disagree with plan: {name}")
                b = image.convert("RGB")
            with Image.open(browser_dir / name) as image:
                if image.size != (plan["width"], plan["height"]):
                    raise BridgeParityError(f"WebGL capture dimensions disagree with plan: {name}")
                w = image.convert("RGB")
        except (OSError, Image.DecompressionBombError) as error:
            raise BridgeParityError(f"Invalid capture PNG: {name}: {error}") from error
        if b.size != (plan["width"], plan["height"]) or w.size != b.size:
            raise BridgeParityError(f"Capture dimensions disagree with plan: {name}")
        metric, overlay = _metrics(b, w)
        passed = metric["iou"] >= min_iou and metric["center_delta_normalized"] <= max_center_delta
        results.append({"frame": frame, "status": "pass" if passed else "review-required", **metric,
                        "overlay": f"overlay-{frame:05d}.png"})
        overlays.append((frame, overlay))
    destination.mkdir(parents=True, exist_ok=False)
    for frame, overlay in overlays:
        overlay.save(destination / f"overlay-{frame:05d}.png")
    sheet = Image.new("RGB", (plan["width"], (plan["height"] + 36)*len(overlays)), "#11181b")
    draw = ImageDraw.Draw(sheet)
    for i, (frame, overlay) in enumerate(overlays):
        y = i * (plan["height"] + 36)
        draw.text((12, y + 10), f"Frame {frame} | IoU {results[i]['iou']:.3f} | {results[i]['status']}", fill="#ffffff")
        sheet.paste(overlay, (0, y + 36))
    sheet.save(destination / "contact-sheet.png")
    report = {"schema_version": 1, "status": "pass" if all(row["status"] == "pass" for row in results) else "review-required",
              "scope": "silhouette geometry and camera only; excludes materials, lighting and realism",
              "plan_sha256": sha256(plan_path), "model_sha256": plan["model_sha256"], "frames": results,
              "contact_sheet": "contact-sheet.png"}
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
