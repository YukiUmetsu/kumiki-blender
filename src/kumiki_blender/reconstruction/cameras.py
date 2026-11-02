"""Turn registered COLMAP text cameras into measured Blender camera poses.

The sparse model has arbitrary scale/orientation. An artist must supply paired
3D survey/mesh landmarks before a camera is usable for matched-view review.
No photo pixels or borrowed geometry are bundled into the camera plan.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .references import ReferenceError


_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_MODEL = re.compile(r"[0-9]+\Z")


def _vector(values: Any, label: str, *, size: int = 3) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ReferenceError(f"{label} must contain {size} finite numbers")
    if any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e8
           for value in values):
        raise ReferenceError(f"{label} must contain {size} finite numbers")
    return tuple(float(value) for value in values)


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sub(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _cross(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _matvec(m: list[list[float]], v: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(_dot(tuple(row), v) for row in m)


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _quat_matrix(q: tuple[float, ...]) -> list[list[float]]:
    norm = math.sqrt(_dot(q, q))
    if not 0.999 <= norm <= 1.001:
        raise ReferenceError("COLMAP quaternion must be normalized")
    w, x, y, z = (value / norm for value in q)
    return [
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ]


def _max_eigen_quaternion(matrix: list[list[float]]) -> tuple[float, ...]:
    """Jacobi diagonalization of symmetric Horn 4x4 matrix; no NumPy needed."""
    a = [row[:] for row in matrix]
    eigenvectors = [[float(i == j) for j in range(4)] for i in range(4)]
    for _ in range(80):
        p, q = max(((i, j) for i in range(4) for j in range(i+1, 4)),
                   key=lambda ij: abs(a[ij[0]][ij[1]]))
        if abs(a[p][q]) < 1e-11:
            break
        angle = 0.5 * math.atan2(2*a[p][q], a[q][q]-a[p][p])
        c, s = math.cos(angle), math.sin(angle)
        # Orthogonal similarity transform with columns (c,-s), (s,c).
        for i in range(4):
            if i != p and i != q:
                ap, aq = a[i][p], a[i][q]
                a[i][p] = a[p][i] = c*ap-s*aq
                a[i][q] = a[q][i] = s*ap+c*aq
        pp, pq, qq = a[p][p], a[p][q], a[q][q]
        a[p][p] = c*c*pp - 2*c*s*pq + s*s*qq
        a[q][q] = s*s*pp + 2*c*s*pq + c*c*qq
        a[p][q] = a[q][p] = 0.0
        for i in range(4):
            vp, vq = eigenvectors[i][p], eigenvectors[i][q]
            eigenvectors[i][p] = c*vp-s*vq
            eigenvectors[i][q] = s*vp+c*vq
    maximum = max(range(4), key=lambda i: a[i][i])
    q = tuple(eigenvectors[i][maximum] for i in range(4))
    if q[0] < 0:
        q = tuple(-v for v in q)
    return q


def _noncollinear(points: list[tuple[float, ...]]) -> bool:
    origin = points[0]
    rays = [_sub(point, origin) for point in points[1:]]
    largest = max((_dot(ray, ray) for ray in rays), default=0.0)
    return largest > 1e-12 and any(
        _dot(_cross(a, b), _cross(a, b)) > largest*largest*1e-8
        for a in rays for b in rays)


def fit_similarity(anchors: list[tuple[tuple[float, ...], tuple[float, ...]]]
                   ) -> tuple[float, list[list[float]], tuple[float, ...], float]:
    """Proper (non-reflecting) least-squares Sim(3) with RMS in target metres."""
    if not 3 <= len(anchors) <= 64:
        raise ReferenceError("Alignment requires 3–64 paired 3D anchors")
    source = [item[0] for item in anchors]
    target = [item[1] for item in anchors]
    if not _noncollinear(source) or not _noncollinear(target):
        raise ReferenceError("Alignment anchors must span non-collinear 3D points on both models")
    n = len(anchors)
    xmean = tuple(sum(point[i] for point in source) / n for i in range(3))
    ymean = tuple(sum(point[i] for point in target) / n for i in range(3))
    xs = [_sub(point, xmean) for point in source]
    ys = [_sub(point, ymean) for point in target]
    covariance = [[sum(x[i]*y[j] for x, y in zip(xs, ys)) for j in range(3)]
                  for i in range(3)]
    xx, xy, xz = covariance[0]
    yx, yy, yz = covariance[1]
    zx, zy, zz = covariance[2]
    nmatrix = [
        [xx+yy+zz, yz-zy, zx-xz, xy-yx],
        [yz-zy, xx-yy-zz, xy+yx, zx+xz],
        [zx-xz, xy+yx, -xx+yy-zz, yz+zy],
        [xy-yx, zx+xz, yz+zy, -xx-yy+zz],
    ]
    rotation = _quat_matrix(_max_eigen_quaternion(nmatrix))
    denominator = sum(_dot(x, x) for x in xs)
    scale = sum(_dot(y, _matvec(rotation, x)) for x, y in zip(xs, ys)) / denominator
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ReferenceError("Alignment scale must be positive; check anchor correspondences")
    translation = _sub(ymean, tuple(scale*v for v in _matvec(rotation, xmean)))
    residuals = [_sub(tuple(scale*v + t for v, t in zip(_matvec(rotation, x), translation)), y)
                 for x, y in anchors]
    rms = math.sqrt(sum(_dot(residual, residual) for residual in residuals) / n)
    return scale, rotation, translation, rms


def _read_anchors(path: Path, asset_id: str) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    try:
        if path.stat().st_size > 64*1024:
            raise ReferenceError("Anchor JSON exceeds 64 KiB")
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        if isinstance(error, ReferenceError):
            raise
        raise ReferenceError(f"Could not read alignment anchors: {error}") from error
    if (not isinstance(raw, dict) or set(raw) != {"schema_version", "asset_id", "anchors"}
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1
            or raw["asset_id"] != asset_id or not isinstance(raw["anchors"], list)):
        raise ReferenceError("Alignment anchor schema/asset_id does not match reconstruction")
    anchors = []
    for index, item in enumerate(raw["anchors"]):
        if not isinstance(item, dict) or set(item) != {"colmap_xyz", "blender_xyz_m"}:
            raise ReferenceError(f"Invalid alignment anchor {index}")
        anchors.append((_vector(item["colmap_xyz"], "colmap_xyz"),
                        _vector(item["blender_xyz_m"], "blender_xyz_m")))
    return anchors


def _read_intrinsics(path: Path) -> dict[int, dict[str, Any]]:
    cameras = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        try:
            camera_id, model = int(parts[0]), parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = tuple(float(value) for value in parts[4:])
            if not 64 <= min(width, height) or width*height > 40_000_000:
                raise ValueError("invalid camera dimensions")
            if model == "PINHOLE" and len(params) == 4:
                fx, fy, cx, cy = params
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                expected = {"SIMPLE_PINHOLE": 3, "SIMPLE_RADIAL": 4, "RADIAL": 5}[model]
                if len(params) != expected or any(abs(v) > 1e-9 for v in params[3:]):
                    raise ReferenceError("COLMAP camera has lens distortion: undistort before Blender matching")
                fx = fy = params[0]
                cx, cy = params[1:3]
            else:
                raise ReferenceError(f"Unsupported COLMAP camera model {model}: undistort to PINHOLE first")
            if (not all(math.isfinite(v) for v in params) or fx <= 0 or fy <= 0
                    or not 0 <= cx <= width or not 0 <= cy <= height):
                raise ValueError("invalid intrinsics")
            if abs(fx-fy) / max(fx, fy) > 0.005:
                raise ReferenceError("Different fx/fy focal lengths need pixel-aspect handling; undistort to square-pixel PINHOLE")
            if camera_id in cameras:
                raise ValueError("duplicate camera ID")
            cameras[camera_id] = {"fx_px": fx, "fy_px": fy, "cx_px": cx, "cy_px": cy,
                                  "width": width, "height": height, "model": model}
        except ReferenceError:
            raise
        except (IndexError, ValueError, OverflowError) as error:
            raise ReferenceError(f"Invalid COLMAP cameras.txt entry: {line[:100]}") from error
    if not cameras:
        raise ReferenceError("COLMAP cameras.txt contains no cameras")
    return cameras


def _read_poses(path: Path) -> list[tuple[int, int, str, tuple[float, ...], tuple[float, ...]]]:
    poses = []
    lines = iter(path.read_text(encoding="utf-8").splitlines())
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        try:
            if len(fields) != 10:
                raise ValueError("pose has unexpected field count")
            image_id, camera_id = int(fields[0]), int(fields[8])
            quaternion = tuple(float(value) for value in fields[1:5])
            translation = tuple(float(value) for value in fields[5:8])
            _vector(quaternion, "pose quaternion", size=4)
            _vector(translation, "pose translation")
            _quat_matrix(quaternion)
            filename = fields[9]
            if not filename or filename.startswith("/") or "\\" in filename or "/" in filename:
                raise ValueError("unsupported staged filename")
            if any(entry[0] == image_id or entry[2] == filename for entry in poses):
                raise ValueError("duplicate image ID or filename")
            poses.append((image_id, camera_id, filename, quaternion, translation))
            next(lines, None)  # COLMAP pairs each pose with one 2D observations row.
        except ReferenceError:
            raise
        except (ValueError, IndexError) as error:
            raise ReferenceError(f"Invalid COLMAP images.txt pose: {line[:100]}") from error
    return poses


def create_camera_plan(reconstruction: str | Path, alignment: str | Path,
                       output: str | Path, *, max_anchor_rms_m: float = 0.10,
                       undistorted_masks: str | Path | None = None) -> dict[str, Any]:
    """Review COLMAP cameras against pinned photos and explicit sculpt-space anchors."""
    if type(max_anchor_rms_m) not in (int, float) or not math.isfinite(max_anchor_rms_m) or max_anchor_rms_m <= 0:
        raise ReferenceError("max_anchor_rms_m must be a positive finite distance")
    root = Path(reconstruction).resolve()
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists():
        raise ReferenceError(f"Camera plan output already exists: {target}")
    if target.is_relative_to(root):
        raise ReferenceError("Camera plan output must stay outside the source reconstruction directory")
    try:
        report = json.loads((root / "reconstruction-report.json").read_text(encoding="utf-8"))
        index = json.loads((root / "staged-reference-index.json").read_text(encoding="utf-8"))
        asset_id = report["asset_id"]
        if not isinstance(asset_id, str) or not _ID.fullmatch(asset_id):
            raise ValueError("invalid asset_id")
        if report["status"] != "candidate / visual-review-required":
            raise ReferenceError("Reconstruction has no successful sparse candidate")
        selected = max(report["models"], key=lambda item: item["registered_images"])
        model_name = selected["model"]
        if not isinstance(model_name, str) or not _MODEL.fullmatch(model_name):
            raise ValueError("invalid sparse model directory")
        corrected = "undistorted_model" in report
        model = root / ("undistorted/text-model" if corrected else "text-models/" + model_name)
        if model.is_symlink() or not model.is_dir():
            raise ValueError("missing safe COLMAP text model")
        if corrected:
            index = json.loads((root / "undistorted-reference-index.json").read_text(encoding="utf-8"))
            photo_root = root / "undistorted/images"
            if report["source_sparse_model"] != model_name:
                raise ValueError("rectified cameras do not match selected original model")
        else:
            photo_root = root / "images"
            if undistorted_masks is not None:
                raise ReferenceError("--undistorted-masks requires a rectified COLMAP reconstruction")
        staged = {image["filename"]: image for image in index["images"]}
        if len(staged) != len(index["images"]):
            raise ValueError("duplicate staged filenames")
        poses = _read_poses(model / "images.txt")
        intrinsics = _read_intrinsics(model / "cameras.txt")
        if len(poses) < 3 or len(poses) != selected["registered_images"]:
            raise ValueError("registered camera count differs from sparse report")
        for _, camera_id, filename, _, _ in poses:
            if filename not in staged:
                raise ReferenceError(f"Registered image {filename} not in staged photo index")
            if camera_id not in intrinsics:
                raise ValueError("COLMAP pose refers to absent camera intrinsics")
            photo = photo_root / filename
            if photo.is_symlink() or not photo.is_file():
                raise ReferenceError(f"Staged photo {filename} is missing")
            with photo.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != staged[filename]["sha256"]:
                raise ReferenceError(f"Staged photo {filename} changed since reconstruction")
            if [intrinsics[camera_id]["width"], intrinsics[camera_id]["height"]] != staged[filename]["dimensions"]:
                raise ReferenceError(f"Camera dimensions differ from staged photo {filename}")
        anchors = _read_anchors(Path(alignment), asset_id)
        scale, rotation, translation, rms = fit_similarity(anchors)
        if rms > max_anchor_rms_m:
            raise ReferenceError(f"Alignment anchor RMS {rms:.4f} m exceeds {max_anchor_rms_m:.4f} m")
        masks_root = Path(undistorted_masks).resolve() if undistorted_masks is not None else None
        if masks_root is not None and (not masks_root.is_dir() or masks_root.is_symlink()):
            raise ReferenceError("Undistorted mask directory is missing or a symbolic link")
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        if isinstance(error, ReferenceError):
            raise
        raise ReferenceError(f"Invalid COLMAP reconstruction for camera planning: {error}") from error

    cameras = []
    for image_id, camera_id, filename, quaternion, tvec in poses:
        world_to_camera = _quat_matrix(quaternion)
        camera_to_world = [[world_to_camera[j][i] for j in range(3)] for i in range(3)]
        center = _matvec(camera_to_world, tuple(-value for value in tvec))
        location = tuple(scale*v+t for v, t in zip(_matvec(rotation, center), translation))
        # COLMAP x-right/y-down/z-forward -> Blender x-right/y-up/z-back.
        right_handed_conversion = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        orientation = _matmul(_matmul(rotation, camera_to_world), right_handed_conversion)
        photo = photo_root / filename
        mask = ((masks_root if corrected else root / "masks") / (filename + ".png")
                if not corrected or masks_root is not None else None)
        if masks_root is not None:
            from PIL import Image
            if mask is None or mask.is_symlink() or not mask.is_file():
                raise ReferenceError(f"Missing rectified silhouette mask for {filename}")
            with Image.open(mask) as mask_image:
                if list(mask_image.size) != [intrinsics[camera_id]["width"], intrinsics[camera_id]["height"]]:
                    raise ReferenceError(f"Rectified silhouette mask dimensions differ for {filename}")
        mask_digest = None
        if mask is not None and mask.is_file() and not mask.is_symlink():
            with mask.open("rb") as stream:
                mask_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if not corrected and "staged_mask_sha256" in staged[filename] and (
                    mask_digest != staged[filename]["staged_mask_sha256"]):
                raise ReferenceError(f"Staged mask {filename} changed since reconstruction")
        cameras.append({
            "image_id": staged[filename]["image_id"], "colmap_image_id": image_id,
            "source_filename": filename,
            "source_photo_path": str(photo),
            "source_photo_sha256": staged[filename]["sha256"],
            "source_mask_path": str(mask) if mask is not None and mask.is_file() and not mask.is_symlink() else None,
            "source_mask_sha256": mask_digest,
            "intrinsics": intrinsics[camera_id], "translation_m": list(location),
            "rotation_3x3": orientation,
        })
    result = {
        "schema_version": 1, "asset_id": asset_id,
        "reconstruction_path": str(root), "sparse_model": model_name,
        "rectified_cameras": corrected,
        "status": "alignment-candidate / artistic-review-required",
        "registered_cameras": len(cameras), "anchor_count": len(anchors),
        "anchor_rms_m": rms, "max_anchor_rms_m": max_anchor_rms_m,
        "similarity": {"scale_m_per_colmap_unit": scale, "rotation_3x3": rotation,
                       "translation_m": list(translation)},
        "cameras": cameras,
        "historical_fidelity_verified": False,
        "limitations": "Source cameras and surveyed anchors support calibrated review, not proof of mesh likeness; distorted cameras need undistortion. Output contains private local photo paths.",
    }
    target.mkdir(parents=True, exist_ok=False)
    (target / "camera-plan.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
