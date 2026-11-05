"""Private, auditable sparse-COLMAP -> dense statue mesh reconstruction.

OpenMVS provides the CPU-capable path (including Apple Silicon machines).
COLMAP PatchMatch is an optional GPU backend. Neither path establishes that
the input photos depict the correct monument or licenses the output textures.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .references import MAX_IMAGES, ReferenceError

_IMAGE_NAME = re.compile(r"[a-z][a-z0-9-]*\.(?:jpg|jpeg|png|webp)\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class DenseSettings:
    backend: str = "openmvs"
    min_registered_images: int = 8
    min_registered_fraction: float = 0.60
    timeout_seconds: int = 7200
    refine: bool = True
    texture: bool = True
    min_mesh_faces: int = 1000
    resolution_level: int = 0
    max_resolution: int = 3200
    number_views: int = 5
    refine_decimate: float = 1.0
    texture_decimate: float = 1.0
    # OpenMVS 2.4 has a known texture sampling/seam-leveling regression that
    # turns valid photographs into black/rainbow atlas patches. Preserve
    # original photographed color by default; users can opt in on fixed builds.
    texture_global_seam_leveling: bool = False
    texture_local_seam_leveling: bool = False

    def validate(self) -> "DenseSettings":
        if self.backend not in ("openmvs", "colmap"):
            raise ReferenceError("Dense backend must be openmvs or colmap")
        if type(self.min_registered_images) is not int or not 3 <= self.min_registered_images <= MAX_IMAGES:
            raise ReferenceError("min_registered_images must be 3–256")
        if (type(self.min_registered_fraction) not in (float, int)
                or not math.isfinite(self.min_registered_fraction)
                or not 0.25 <= self.min_registered_fraction <= 1.0):
            raise ReferenceError("min_registered_fraction must be finite in [0.25, 1.0]")
        if type(self.timeout_seconds) is not int or not 60 <= self.timeout_seconds <= 86400:
            raise ReferenceError("timeout_seconds must be 60–86400")
        if type(self.refine) is not bool or type(self.texture) is not bool:
            raise ReferenceError("refine and texture must be booleans")
        if type(self.min_mesh_faces) is not int or not 100 <= self.min_mesh_faces <= 10_000_000:
            raise ReferenceError("min_mesh_faces must be 100–10,000,000")
        if type(self.resolution_level) is not int or not 0 <= self.resolution_level <= 4:
            raise ReferenceError("resolution_level must be 0–4")
        if type(self.max_resolution) is not int or not 640 <= self.max_resolution <= 16384:
            raise ReferenceError("max_resolution must be 640–16384")
        if type(self.number_views) is not int or not 2 <= self.number_views <= 32:
            raise ReferenceError("number_views must be 2–32")
        if (type(self.refine_decimate) not in (int, float)
                or not math.isfinite(self.refine_decimate)
                or not 0 < self.refine_decimate <= 1):
            raise ReferenceError("refine_decimate must be finite in (0,1]; 1 preserves input faces")
        if (type(self.texture_decimate) not in (int, float)
                or not math.isfinite(self.texture_decimate)
                or not 0 < self.texture_decimate <= 1):
            raise ReferenceError("texture_decimate must be finite in (0,1]; 1 preserves texture faces")
        if (type(self.texture_global_seam_leveling) is not bool
                or type(self.texture_local_seam_leveling) is not bool):
            raise ReferenceError("Texture seam-leveling options must be booleans")
        return self


def _json(path: Path, *, max_bytes: int = 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ReferenceError(f"Missing or oversized reconstruction evidence: {path.name}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReferenceError(f"Cannot parse reconstruction evidence: {path.name}") from error
    if not isinstance(result, dict):
        raise ReferenceError(f"Reconstruction evidence is not a JSON object: {path.name}")
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_regular(path: Path, root: Path) -> None:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ReferenceError(f"Required reconstruction file is missing or linked outside the revision: {path.name}")


def _stage_openmvs_mask_aliases(image_root: Path, masks: list[dict[str, Any]]) -> None:
    """Make OpenMVS stem-based mask paths alongside hash-pinned snapshots.

    The rectified-mask contract names a mask ``photo-0003.png.mask.png``,
    whereas OpenMVS 2.4 looks for ``photo-0003.mask.png``. Preserve the
    original snapshot for resume validation and link an identical alias in
    the private workspace. This can also repair a previously failed stage.
    """
    for entry in masks:
        original = image_root / (entry["filename"] + ".mask.png")
        alias = image_root / (Path(entry["filename"]).stem + ".mask.png")
        _check_regular(original, image_root)
        if _sha256(original) != entry["sha256"]:
            raise ReferenceError(f"OpenMVS mask snapshot changed: {entry['filename']}")
        if alias.exists() or alias.is_symlink():
            _check_regular(alias, image_root)
            if _sha256(alias) != entry["sha256"]:
                raise ReferenceError(f"OpenMVS mask alias changed: {alias.name}")
            continue
        try:
            os.link(original, alias)
        except OSError:
            # Exclusive creation prevents replacing another alias if a
            # concurrent process created it after the existence check.
            with original.open("rb") as source, alias.open("xb") as destination:
                shutil.copyfileobj(source, destination)
        _check_regular(alias, image_root)
        if _sha256(alias) != entry["sha256"]:
            raise ReferenceError(f"OpenMVS mask alias changed during staging: {alias.name}")


def plan_dense_reconstruction(reconstruction: str | Path, output: str | Path,
                              settings: DenseSettings, *,
                              rectified_masks: str | Path | None = None,
                              executables: dict[str, str] | None = None) -> dict[str, Any]:
    """Non-mutating preflight; requires registered, rectified camera evidence."""
    settings.validate()
    if rectified_masks is not None and settings.backend != "openmvs":
        raise ReferenceError(
            "Rectified subject masks are supported only by the OpenMVS backend; "
            "COLMAP PatchMatch does not consume these mask arguments")
    source = Path(reconstruction).resolve()
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists():
        raise ReferenceError(f"Output already exists: {target}")
    if target == source or target.is_relative_to(source):
        raise ReferenceError("Dense output must be outside the sparse reconstruction revision")
    if not source.is_dir() or Path(reconstruction).is_symlink():
        raise ReferenceError("Sparse reconstruction revision does not exist or is a symlink")
    report = _json(source / "reconstruction-report.json")
    index = _json(source / "undistorted-reference-index.json")
    count = report.get("registered_images")
    staged_count = report.get("staged_independent_images")
    if not isinstance(report.get("status"), str) or not report["status"].startswith("candidate") or not (
            type(count) is int and type(staged_count) is int
            and 3 <= count <= staged_count <= MAX_IMAGES):
        raise ReferenceError("Sparse reconstruction is not a viable registered-camera candidate")
    if count < settings.min_registered_images or count / staged_count < settings.min_registered_fraction:
        raise ReferenceError(
            f"Dense reconstruction needs {settings.min_registered_images}+ views and "
            f"{settings.min_registered_fraction:.0%}+ independent-image registration "
            f"(found {count}/{staged_count})")
    images = index.get("images")
    if (not isinstance(images, list) or len(images) != count
            or report.get("undistorted_images") != count):
        raise ReferenceError("Rectified photo index does not match registered cameras")
    undistorted = source / "undistorted"
    image_root = undistorted / "images"
    sparse = undistorted / "sparse"
    if (undistorted.is_symlink() or not image_root.is_dir() or image_root.is_symlink()
            or not sparse.is_dir() or sparse.is_symlink()):
        raise ReferenceError("Undistorted COLMAP images/sparse workspace is missing")
    for basename in ("cameras", "images", "points3D"):
        candidates = [sparse / f"{basename}.{extension}" for extension in ("bin", "txt")]
        if not any(path.is_file() and not path.is_symlink() and path.stat().st_size > 0
                   for path in candidates):
            raise ReferenceError(f"Missing rectified COLMAP {basename} model")
    verified = []
    for entry in images:
        if not isinstance(entry, dict):
            raise ReferenceError("Invalid rectified image index entry")
        name, expected = entry.get("filename"), entry.get("sha256")
        if (not isinstance(name, str) or not _IMAGE_NAME.fullmatch(name)
                or not isinstance(expected, str) or not _SHA256.fullmatch(expected)
                or name in verified):
            raise ReferenceError("Unsafe or duplicate rectified image path/hash")
        path = image_root / name
        _check_regular(path, source)
        if _sha256(path) != expected:
            raise ReferenceError(f"Rectified photo changed since reconstruction: {name}")
        verified.append(name)
    masks = []
    mask_source = None
    if rectified_masks is not None:
        stems = [Path(name).stem for name in verified]
        if len(stems) != len(set(stems)):
            raise ReferenceError("OpenMVS masks require unique rectified image stems")
        mask_source = Path(rectified_masks).resolve()
        if (Path(rectified_masks).is_symlink() or not mask_source.is_dir()
                or mask_source.is_relative_to(source / "undistorted/images")):
            raise ReferenceError("Rectified masks need a regular private directory outside rectified images")
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as error:
            raise ReferenceError("Rectified mask QA requires Pillow (install the qa extra)") from error
        for entry in images:
            name = entry["filename"]
            dimensions = entry.get("dimensions")
            path = mask_source / (name + ".mask.png")
            _check_regular(path, mask_source)
            try:
                with Image.open(path) as image:
                    image.load()
                    if (image.format != "PNG" or image.mode not in ("L", "1")
                            or list(image.size) != dimensions):
                        raise ReferenceError(f"Mask {name} must be grayscale PNG at exact rectified dimensions")
                    colors = image.convert("L").getcolors(maxcolors=3)
                    if (colors is None or {value for _, value in colors} != {0, 255}):
                        raise ReferenceError(f"Mask {name} must contain both black and white binary pixels")
            except ReferenceError:
                raise
            except (OSError, UnidentifiedImageError, ValueError) as error:
                raise ReferenceError(f"Cannot decode rectified binary mask {name}") from error
            masks.append({"filename": name, "path": str(path), "sha256": _sha256(path)})
    if executables is None:
        needed = (("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh") +
                  (("RefineMesh",) if settings.refine else ()) +
                  (("TextureMesh",) if settings.texture else ())) if settings.backend == "openmvs" else ("colmap",)
        executables = {}
        for name in needed:
            found = shutil.which(name)
            if not found:
                raise ReferenceError(f"Required dense reconstruction executable is not installed: {name}")
            executables[name] = found
    return {
        "schema_version": 1,
        "source_revision": str(source), "output": str(target),
        "asset_id": report.get("asset_id"),
        "manifest_sha256": report.get("manifest_sha256"),
        "backend": settings.backend, "registered_images": count,
        "independent_images": staged_count,
        "registered_fraction": round(count / staged_count, 6),
        "undistorted_images": verified,
        "rectified_mask_source": str(mask_source) if mask_source is not None else None,
        "rectified_masks": masks,
        # OpenMVS's mask-label default is -1 (disabled). Merely copying
        # .mask.png files next to photos does NOT apply foreground filtering.
        "mask_application": ({"densify": "ignore-background-label-0",
                              "texture": "ignore-background-label-0" if settings.texture else None,
                              "refine": "not-supported-by-openmvs-2.4" if settings.refine else None}
                             if masks else None),
        "executables": executables,
        "timeout_seconds": settings.timeout_seconds,
        "min_mesh_faces": settings.min_mesh_faces,
        "refine": settings.refine if settings.backend == "openmvs" else False,
        "texture": settings.texture,
        "resolution_level": settings.resolution_level if settings.backend == "openmvs" else None,
        "max_resolution": settings.max_resolution if settings.backend == "openmvs" else None,
        "number_views": settings.number_views if settings.backend == "openmvs" else None,
        "refine_decimate": settings.refine_decimate if settings.backend == "openmvs" else None,
        "texture_decimate": settings.texture_decimate if settings.backend == "openmvs" else None,
        "texture_global_seam_leveling": (
            settings.texture_global_seam_leveling if settings.backend == "openmvs" else None),
        "texture_local_seam_leveling": (
            settings.texture_local_seam_leveling if settings.backend == "openmvs" else None),
        "status": "preflight-only / dense-and-artistic-review-required",
        "distribution_allowed": False,
        "limitations": "Output is private. Photo rights, monument identity, surface accuracy, scale and topology require independent review.",
    }


def _ply_counts(path: Path) -> dict[str, int]:
    """Read a bounded PLY header without loading a potentially enormous scan."""
    if not path.is_file() or path.is_symlink():
        raise ReferenceError("Dense mesh PLY is missing")
    with path.open("rb") as stream:
        header = stream.read(65_537).replace(b"\r\n", b"\n")
    end = header.find(b"end_header\n")
    if end < 0 or end > 65_000 or not header.startswith(b"ply\n"):
        raise ReferenceError("Dense mesh has an invalid or oversized PLY header")
    try:
        lines = header[:end].decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ReferenceError("PLY header is not ASCII") from error
    if not any(line.startswith("format ascii ") or line.startswith("format binary_") for line in lines):
        raise ReferenceError("PLY encoding is missing")
    counts: dict[str, int] = {}
    for line in lines:
        words = line.split()
        if len(words) == 3 and words[:2] in (["element", "vertex"], ["element", "face"]):
            try:
                counts[words[1]] = int(words[2])
            except ValueError as error:
                raise ReferenceError("Invalid PLY geometry count") from error
    if counts.get("vertex", 0) <= 0 or counts.get("face", 0) <= 0:
        raise ReferenceError("Dense mesh contains no vertices or faces")
    return {"vertices": counts["vertex"], "faces": counts["face"]}


def _execute(command: list[str], log: Path, timeout_seconds: int) -> int:
    try:
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=timeout_seconds, check=False)
            return result.returncode
    except (subprocess.TimeoutExpired, OSError) as error:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nDense stage failed to execute: {error}\n")
        return 124 if isinstance(error, subprocess.TimeoutExpired) else 127


def build_dense_reconstruction(reconstruction: str | Path, output: str | Path,
                               settings: DenseSettings, *,
                               rectified_masks: str | Path | None = None,
                               executables: dict[str, str] | None = None,
                               resume: bool = False) -> dict[str, Any]:
    """Reconstruct a private candidate mesh and checkpoint every external stage."""
    if type(resume) is not bool:
        raise ReferenceError("resume must be a boolean")
    target = Path(output).resolve()
    if resume:
        if Path(output).is_symlink() or not target.is_dir() or settings.backend != "openmvs":
            raise ReferenceError("Resume requires an existing, non-linked OpenMVS revision")
        # Full original camera/image hash preflight still runs. The placeholder
        # is never created; this preserves the plan's no-overwrite contract.
        plan = plan_dense_reconstruction(reconstruction, target / ".resume-preflight",
                                         settings, rectified_masks=rectified_masks,
                                         executables=executables)
        plan["output"] = str(target)
        report = _json(target / "dense-report.json")
        if report.get("status") != "review-required" or not isinstance(report.get("commands"), list):
            raise ReferenceError("Only an incomplete dense revision may be resumed")
        for key in ("source_revision", "asset_id", "manifest_sha256", "backend",
                    "registered_images", "independent_images",
                    "resolution_level", "max_resolution", "number_views",
                    "rectified_mask_source", "rectified_masks"):
            previous = report.get(key, [] if key == "rectified_masks" else None)
            if previous != plan.get(key):
                raise ReferenceError(f"Resume cannot change existing dense input/settings: {key}")
        prior = {step.get("stage") for step in report["commands"]
                 if isinstance(step, dict)}
        if plan["rectified_masks"] and any(
                step.get("stage") == "densify" and (
                    "--mask-path" not in step.get("argv", [])
                    or "--ignore-mask-label" not in step.get("argv", []))
                for step in report["commands"] if isinstance(step, dict)):
            raise ReferenceError(
                "This dense revision previously ran without subject-mask filtering. "
                "Start a new dense output revision; resuming would reuse contaminated depth maps")
        # Later, unstarted stages may be explicitly disabled after a failure
        # without invalidating the expensive camera/dense/mesh computation.
        failed_texturing = report.get("failed_stage") in (
            "texture", "texture_output", "texture_mesh_output")
        for key, begun in (("refine", {"refine", "texture"}),
                           ("texture", {"texture"})):
            if report.get(key) != plan[key] and prior.intersection(begun):
                if key == "texture" and failed_texturing and plan[key] is False:
                    continue  # Explicitly accept raw PLY after failed atlas generation.
                raise ReferenceError(f"Resume cannot change a started dense stage: {key}")
        # Older interrupted revisions had no explicit refine decimation flag.
        # It is safe to use the new detail-preserving default only if refinement
        # never began; otherwise the setting must match the prior revision.
        if ("refine_decimate" in report and report["refine_decimate"] != plan["refine_decimate"]
                or "refine_decimate" not in report and any(
                    step.get("stage") == "refine" for step in report["commands"])):
            raise ReferenceError("Cannot change refinement settings in a started revision")
        # TextureMesh may be retried at a lower polygon budget if an earlier
        # texture run failed. Geometry/dense stages remain pinned and intact.
        texture_started = "texture" in prior
        texture_failed = failed_texturing
        old_texture_decimate = report.get("texture_decimate", 1.0)
        if (texture_started and not texture_failed
                and old_texture_decimate != plan["texture_decimate"]):
            raise ReferenceError("Cannot change the texture budget of a completed stage")
        for key in ("texture_global_seam_leveling", "texture_local_seam_leveling"):
            if (texture_started and not texture_failed
                    and report.get(key, True) != plan[key]):
                raise ReferenceError("Cannot change seam leveling of a completed texture stage")
        workspace = target / "workspace"
        if not workspace.is_dir() or workspace.is_symlink() or not (target / "logs").is_dir():
            raise ReferenceError("Interrupted dense workspace or logs are missing")
        for name in plan["undistorted_images"]:
            staged = workspace / "images" / name
            _check_regular(staged, target)
            if _sha256(staged) != _sha256(Path(plan["source_revision"]) / "undistorted/images" / name):
                raise ReferenceError(f"Resume workspace image changed: {name}")
        for entry in plan["rectified_masks"]:
            snapshot = workspace / "images" / (entry["filename"] + ".mask.png")
            _check_regular(snapshot, target)
            if _sha256(snapshot) != entry["sha256"]:
                raise ReferenceError(f"Resume rectified mask changed: {entry['filename']}")
        _stage_openmvs_mask_aliases(workspace / "images", plan["rectified_masks"])
        report["resumed_from"] = {"failed_stage": report.get("failed_stage"),
                                  "exit_code": report.get("exit_code")}
        report.update({"status": "review-required", "failed_stage": None,
                       "exit_code": None, "refine_decimate": plan["refine_decimate"],
                       "texture_decimate": plan["texture_decimate"],
                       "texture_global_seam_leveling": plan["texture_global_seam_leveling"],
                       "texture_local_seam_leveling": plan["texture_local_seam_leveling"],
                       "refine": plan["refine"], "texture": plan["texture"],
                       "mask_application": plan["mask_application"],
                       "rectified_mask_source": plan["rectified_mask_source"],
                       "rectified_masks": plan["rectified_masks"]})
        report.pop("error", None)
        report["resume_skipped_stages"] = []
        previous_stages = {step.get("stage") for step in report["commands"]
                           if isinstance(step, dict)}
    else:
        plan = plan_dense_reconstruction(reconstruction, output, settings,
                                         rectified_masks=rectified_masks,
                                         executables=executables)
        target.mkdir(parents=True, exist_ok=False)
        (target / "logs").mkdir()
        report = {**plan, "status": "review-required", "commands": [],
                  "failed_stage": None, "exit_code": None, "mesh": None}
        previous_stages = set()

    def checkpoint() -> dict[str, Any]:
        (target / "dense-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                   encoding="utf-8")
        return report

    def run(stage: str, command: list[str]) -> bool:
        report["commands"].append({"stage": stage, "argv": command[1:]})
        code = _execute(command, target / "logs" / f"{len(report['commands']):02d}-{stage}.log",
                        settings.timeout_seconds)
        if code:
            report["failed_stage"], report["exit_code"] = stage, code
        checkpoint()
        return code == 0

    def require_file(stage: str, path: Path) -> bool:
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            report.update(failed_stage=stage + "_output", exit_code=2,
                          error=f"{stage} produced no usable {path.name}")
            checkpoint()
            return False
        return True

    checkpoint()
    source = Path(plan["source_revision"])
    exe = plan["executables"]
    workspace = target / "workspace"
    if not resume:
        try:
            # OpenMVS can store image references relative to its working folder.
            # Neither backend writes into the immutable camera revision.
            for path in (source / "undistorted").rglob("*"):
                if path.is_symlink():
                    raise ReferenceError(f"Linked COLMAP input is unsupported: {path.name}")
            shutil.copytree(source / "undistorted", workspace)
            for mask in plan["rectified_masks"]:
                destination = workspace / "images" / (mask["filename"] + ".mask.png")
                shutil.copy2(mask["path"], destination)
                if _sha256(destination) != mask["sha256"]:
                    raise ReferenceError(f"Rectified mask changed while staging: {mask['filename']}")
            _stage_openmvs_mask_aliases(workspace / "images", plan["rectified_masks"])
        except (OSError, ReferenceError) as error:
            report.update(failed_stage="copy_workspace", error=str(error), exit_code=2)
            return checkpoint()
    if settings.backend == "openmvs":
        # OpenMVS 2.4 DensifyPointCloud accepts --mask-path and requires an
        # explicit --ignore-mask-label 0 for black background / white subject.
        # TextureMesh reads the mask filenames embedded in scene_dense.mvs and
        # accepts --ignore-mask-label, but NOT --mask-path in OpenMVS 2.4.
        # RefineMesh 2.4 has neither option; do not send unsupported flags.
        mask_dir = str(workspace / "images") + "/"
        dense_mask_args = (["--mask-path", mask_dir,
                            "--ignore-mask-label", "0"] if plan["rectified_masks"] else [])
        texture_mask_args = (["--ignore-mask-label", "0"]
                             if plan["rectified_masks"] else [])
        scene = workspace / "scene.mvs"
        dense = workspace / "scene_dense.mvs"
        mesh = workspace / "scene_dense_mesh.mvs"
        refined = workspace / "scene_dense_mesh_refined.mvs"
        textured_obj = workspace / "scene_dense_mesh_refined_texture.obj"
        stages: list[tuple[str, str, list[str]]] = [
            ("interface", "InterfaceCOLMAP", ["--input-file", str(workspace),
                                              "--output-file", str(scene)]),
            ("densify", "DensifyPointCloud", ["--input-file", str(scene),
                                             "--output-file", str(dense),
                                             "--resolution-level", str(settings.resolution_level),
                                             "--max-resolution", str(settings.max_resolution),
                                             "--number-views", str(settings.number_views),
                                             "--tower-mode", "0", *dense_mask_args]),
            ("mesh", "ReconstructMesh", ["--input-file", str(dense),
                                        "--output-file", str(mesh)]),
        ]
        if settings.refine:
            stages.append(("refine", "RefineMesh", ["--input-file", str(dense),
                                                   "--mesh-file", str(workspace / "scene_dense_mesh.ply"),
                                                   "--decimate", str(settings.refine_decimate),
                                                   "--resolution-level", str(settings.resolution_level),
                                                   "--output-file", str(refined)]))
        if settings.texture:
            stages.append(("texture", "TextureMesh", [
                "--input-file", str(dense),
                "--mesh-file", str(workspace / ("scene_dense_mesh_refined.ply" if settings.refine
                                                  else "scene_dense_mesh.ply")),
                "--output-file", str(textured_obj), "--export-type", "obj",
                "--decimate", str(settings.texture_decimate),
                "--global-seam-leveling", str(int(settings.texture_global_seam_leveling)),
                "--local-seam-leveling", str(int(settings.texture_local_seam_leveling)),
                "--resolution-level", str(settings.resolution_level),
                *texture_mask_args]))
        ply = workspace / "scene_dense_mesh.ply"
        # OpenMVS 2.4 normally exports ReconstructMesh/RefineMesh as PLY even
        # when --output-file ends in .mvs. Requiring an .mvs archive here
        # rejected a valid 1.7-million-face eagle mesh after 30 minutes of
        # CPU depth reconstruction. TextureMesh similarly writes OBJ+atlases;
        # its material outputs are checked separately below.
        stage_outputs = {"interface": scene, "densify": dense,
                         "mesh": workspace / "scene_dense_mesh.ply",
                         "refine": workspace / "scene_dense_mesh_refined.ply"}
        skip_chain = True
        for stage, app, args in stages:
            existing = stage_outputs.get(stage)
            if (resume and skip_chain and stage in previous_stages
                    and stage != report["resumed_from"]["failed_stage"]
                    and existing is not None
                    and existing.is_file() and not existing.is_symlink()
                    and existing.stat().st_size > 0):
                if stage in ("mesh", "refine"):
                    _ply_counts(existing)
                report["resume_skipped_stages"].append(stage)
                checkpoint()
                continue
            skip_chain = False
            if not run(stage, [exe[app], *args, "--working-folder", str(workspace)]):
                return checkpoint()
            if stage in stage_outputs and not require_file(stage, stage_outputs[stage]):
                return checkpoint()
        if settings.refine:
            ply = workspace / "scene_dense_mesh_refined.ply"
            if not require_file("refine_mesh", ply):
                return checkpoint()
        if settings.texture:
            if not require_file("texture_mesh", textured_obj):
                return checkpoint()
            atlas_paths = sorted(path for path in workspace.glob("scene_dense_mesh_refined_texture*")
                                 if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg")
                                 and not path.is_symlink() and path.stat().st_size > 0)
            textures = [str(path.relative_to(target)) for path in atlas_paths]
            if not textures:
                report.update(failed_stage="texture_output", exit_code=2,
                              error="TextureMesh reported success without texture atlas images")
                return checkpoint()
            report["texture_images"] = textures
            report["textured_obj"] = {"path": str(textured_obj.relative_to(target)),
                                      "sha256": _sha256(textured_obj)}
            report["texture_atlases"] = [{"path": str(path.relative_to(target)),
                                          "sha256": _sha256(path)} for path in atlas_paths]
    else:
        commands = [
            ("patch_match", [exe["colmap"], "patch_match_stereo", "--workspace_path",
                             str(workspace), "--workspace_format", "COLMAP",
                             "--PatchMatchStereo.geom_consistency", "true"]),
            ("fusion", [exe["colmap"], "stereo_fusion", "--workspace_path", str(workspace),
                        "--workspace_format", "COLMAP", "--input_type", "geometric",
                        "--output_path", str(target / "fused.ply")]),
            ("mesh", [exe["colmap"], "poisson_mesher", "--input_path", str(target / "fused.ply"),
                      "--output_path", str(target / "meshed-poisson.ply")]),
        ]
        if settings.texture:
            commands.append(("texture", [exe["colmap"], "mesh_texturer",
                              "--workspace_path", str(workspace),
                              "--input_path", str(target / "meshed-poisson.ply"),
                              "--output_path", str(target / "textured")]))
        ply = target / "meshed-poisson.ply"
        for stage, args in commands:
            if not run(stage, args):
                return checkpoint()
        if settings.texture:
            texture_root = target / "textured"
            if not texture_root.is_dir() or not any(path.is_file() and path.stat().st_size
                                                    for path in texture_root.rglob("*")):
                report.update(failed_stage="texture_output", exit_code=2,
                              error="COLMAP mesh_texturer produced no usable files")
                return checkpoint()
    try:
        metrics = _ply_counts(ply)
        report["mesh"] = {"ply": str(ply.relative_to(target)),
                          "sha256": _sha256(ply), **metrics}
        if settings.backend == "openmvs" and settings.refine:
            raw = workspace / "scene_dense_mesh.ply"
            original = _ply_counts(raw)
            report["raw_mesh"] = {"ply": str(raw.relative_to(target)),
                                  "sha256": _sha256(raw), **original}
            report["refined_face_retention"] = round(
                metrics["faces"] / original["faces"], 5)
        if metrics["faces"] < settings.min_mesh_faces:
            raise ReferenceError(
                f"Dense mesh has only {metrics['faces']} faces; needs {settings.min_mesh_faces}")
    except (ReferenceError, OSError) as error:
        report.update(failed_stage="mesh_validation", error=str(error), exit_code=2)
        return checkpoint()
    report["status"] = "dense-mesh-candidate / artistic-and-rights-review-required"
    return checkpoint()
