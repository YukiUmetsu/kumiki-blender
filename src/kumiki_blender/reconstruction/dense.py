"""Private, auditable sparse-COLMAP -> dense statue mesh reconstruction.

OpenMVS provides the CPU-capable path (including Apple Silicon machines).
COLMAP PatchMatch is an optional GPU backend. Neither path establishes that
the input photos depict the correct monument or licenses the output textures.
"""

from __future__ import annotations

import hashlib
import json
import math
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
        return self


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
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


def plan_dense_reconstruction(reconstruction: str | Path, output: str | Path,
                              settings: DenseSettings, *,
                              executables: dict[str, str] | None = None) -> dict[str, Any]:
    """Non-mutating preflight; requires registered, rectified camera evidence."""
    settings.validate()
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
        "executables": executables,
        "timeout_seconds": settings.timeout_seconds,
        "min_mesh_faces": settings.min_mesh_faces,
        "refine": settings.refine if settings.backend == "openmvs" else False,
        "texture": settings.texture,
        "resolution_level": settings.resolution_level if settings.backend == "openmvs" else None,
        "max_resolution": settings.max_resolution if settings.backend == "openmvs" else None,
        "number_views": settings.number_views if settings.backend == "openmvs" else None,
        "refine_decimate": settings.refine_decimate if settings.backend == "openmvs" else None,
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
                                         settings, executables=executables)
        plan["output"] = str(target)
        report = _json(target / "dense-report.json")
        if report.get("status") != "review-required" or not isinstance(report.get("commands"), list):
            raise ReferenceError("Only an incomplete dense revision may be resumed")
        for key in ("source_revision", "asset_id", "manifest_sha256", "backend",
                    "registered_images", "independent_images", "refine", "texture",
                    "resolution_level", "max_resolution", "number_views"):
            if report.get(key) != plan.get(key):
                raise ReferenceError(f"Resume cannot change existing dense input/settings: {key}")
        # Older interrupted revisions had no explicit refine decimation flag.
        # It is safe to use the new detail-preserving default only if refinement
        # never began; otherwise the setting must match the prior revision.
        if ("refine_decimate" in report and report["refine_decimate"] != plan["refine_decimate"]
                or "refine_decimate" not in report and any(
                    step.get("stage") == "refine" for step in report["commands"])):
            raise ReferenceError("Cannot change refinement settings in a started revision")
        workspace = target / "workspace"
        if not workspace.is_dir() or workspace.is_symlink() or not (target / "logs").is_dir():
            raise ReferenceError("Interrupted dense workspace or logs are missing")
        for name in plan["undistorted_images"]:
            staged = workspace / "images" / name
            _check_regular(staged, target)
            if _sha256(staged) != _sha256(Path(plan["source_revision"]) / "undistorted/images" / name):
                raise ReferenceError(f"Resume workspace image changed: {name}")
        report["resumed_from"] = {"failed_stage": report.get("failed_stage"),
                                  "exit_code": report.get("exit_code")}
        report.update({"status": "review-required", "failed_stage": None,
                       "exit_code": None, "refine_decimate": plan["refine_decimate"]})
        report.pop("error", None)
        report["resume_skipped_stages"] = []
        previous_stages = {step.get("stage") for step in report["commands"]
                           if isinstance(step, dict)}
    else:
        plan = plan_dense_reconstruction(reconstruction, output, settings, executables=executables)
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
        except (OSError, ReferenceError) as error:
            report.update(failed_stage="copy_workspace", error=str(error), exit_code=2)
            return checkpoint()
    if settings.backend == "openmvs":
        scene = workspace / "scene.mvs"
        dense = workspace / "scene_dense.mvs"
        mesh = workspace / "scene_dense_mesh.mvs"
        refined = workspace / "scene_dense_mesh_refined.mvs"
        stages: list[tuple[str, str, list[str]]] = [
            ("interface", "InterfaceCOLMAP", ["--input-file", str(workspace),
                                              "--output-file", str(scene)]),
            ("densify", "DensifyPointCloud", ["--input-file", str(scene),
                                             "--output-file", str(dense),
                                             "--resolution-level", str(settings.resolution_level),
                                             "--max-resolution", str(settings.max_resolution),
                                             "--number-views", str(settings.number_views),
                                             "--tower-mode", "0"]),
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
                "--output-file", str(workspace / "scene_dense_mesh_refined_texture.mvs")]))
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
            textures = sorted(str(path.relative_to(target)) for path in target.rglob("*")
                              if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg")
                              and not path.is_symlink()
                              and not path.is_relative_to(workspace / "images"))
            if not textures:
                report.update(failed_stage="texture_output", exit_code=2,
                              error="TextureMesh reported success without texture atlas images")
                return checkpoint()
            report["texture_images"] = textures
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
