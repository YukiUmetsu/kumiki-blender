"""Single entry point from overlapping private photographs to an editable GLB.

This composes existing sparse, dense and Blender workers. No model is created
when registration/meshing fails; a stage report explains where to retry.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..glb_audit import AuditError, audit_glb
from .dense import DenseSettings, build_dense_reconstruction
from .references import ReferenceError, ReferenceSet, audit_reference_set


@dataclass(frozen=True)
class PhotoModelSettings:
    engine: str = "pycolmap"
    dense_backend: str = "openmvs"
    colmap: str = "colmap"
    blender: str = "blender"
    min_registered_images: int = 8
    min_registered_fraction: float = 0.25
    sparse_timeout_seconds: int = 3600
    dense_timeout_seconds: int = 7200
    max_image_size: int = 3200
    texture: bool = True
    render_previews: bool = True
    matching_preset: str = "sift"
    matching_device: str = "colmap-cpu"
    feature_threads: int | None = None
    matching_threads: int | None = None
    sift_first_octave: int | None = None
    max_colmap_rss_mb: int | None = None
    matching_block_size: int | None = None
    matching_max_features: int | None = None
    matching_pairs_per_process: int | None = None

    def validate(self) -> None:
        from .colmap import _validate_matching_preset

        if self.engine not in ("pycolmap", "colmap"):
            raise ReferenceError("Sparse engine must be pycolmap or colmap")
        _validate_matching_preset(self.matching_preset, self.engine)
        if self.matching_device not in ("colmap-cpu", "mps"):
            raise ReferenceError("matching_device must be colmap-cpu or mps")
        if self.matching_device == "mps" and (self.engine != "colmap" or self.matching_preset != "sift-lightglue"):
            raise ReferenceError("MPS matching requires --engine colmap --matching-preset sift-lightglue")
        if self.matching_device == "mps" and self.matching_pairs_per_process is None:
            raise ReferenceError("MPS matching requires --matching-pairs-per-process")
        for name in ("feature_threads", "matching_threads"):
            count = getattr(self, name)
            if count is not None and (type(count) is not int or not 1 <= count <= 64):
                raise ReferenceError(f"{name} must be 1–64")
        if self.sift_first_octave not in (None, -1, 0):
            raise ReferenceError("sift_first_octave must be -1 or 0")
        if (self.max_colmap_rss_mb is not None and
                (type(self.max_colmap_rss_mb) is not int or
                 (self.max_colmap_rss_mb != 0 and not 1024 <= self.max_colmap_rss_mb <= 131072))):
            raise ReferenceError("max_colmap_rss_mb must be 0 or 1024–131072")
        if (self.matching_block_size is not None and
                (type(self.matching_block_size) is not int or not 1 <= self.matching_block_size <= 50)):
            raise ReferenceError("matching_block_size must be 1–50")
        if (self.matching_max_features is not None and
                (type(self.matching_max_features) is not int or
                 not 256 <= self.matching_max_features <= 32768)):
            raise ReferenceError("matching_max_features must be 256–32768")
        if (self.matching_pairs_per_process is not None and
                (type(self.matching_pairs_per_process) is not int or
                 not 1 <= self.matching_pairs_per_process <= 1024)):
            raise ReferenceError("matching_pairs_per_process must be 1–1024")
        if self.engine != "colmap" and any(value is not None for value in
                                           (self.feature_threads, self.matching_threads,
                                            self.sift_first_octave, self.max_colmap_rss_mb,
                                            self.matching_block_size, self.matching_max_features,
                                            self.matching_pairs_per_process)):
            raise ReferenceError("COLMAP memory controls require --engine colmap")
        if self.sift_first_octave is not None and self.matching_preset not in ("sift", "sift-lightglue"):
            raise ReferenceError("SIFT first octave applies only to SIFT-based matching")
        if self.dense_backend not in ("openmvs", "colmap"):
            raise ReferenceError("Dense backend must be openmvs or colmap")
        if not isinstance(self.colmap, str) or not self.colmap.strip():
            raise ReferenceError("colmap must be a nonempty executable name or path")
        if not isinstance(self.blender, str) or not self.blender.strip():
            raise ReferenceError("blender must be a nonempty executable name or path")
        if type(self.min_registered_images) is not int or not 3 <= self.min_registered_images <= 256:
            raise ReferenceError("min_registered_images must be 3–256")
        if (type(self.min_registered_fraction) not in (int, float)
                or not math.isfinite(self.min_registered_fraction)
                or not 0.25 <= self.min_registered_fraction <= 1.0):
            raise ReferenceError("min_registered_fraction must be in [0.25, 1.0]")
        if (type(self.sparse_timeout_seconds) is not int
                or not 30 <= self.sparse_timeout_seconds <= 86400):
            raise ReferenceError("sparse_timeout_seconds must be 30–86400")
        if (type(self.dense_timeout_seconds) is not int
                or not 60 <= self.dense_timeout_seconds <= 86400):
            raise ReferenceError("dense_timeout_seconds must be 60–86400")
        if type(self.max_image_size) is not int or not 512 <= self.max_image_size <= 16384:
            raise ReferenceError("max_image_size must be 512–16384")
        if type(self.texture) is not bool or type(self.render_previews) is not bool:
            raise ReferenceError("texture and render_previews must be booleans")
        DenseSettings(backend=self.dense_backend,
                      min_registered_images=self.min_registered_images,
                      min_registered_fraction=self.min_registered_fraction,
                      timeout_seconds=self.dense_timeout_seconds,
                      texture=self.texture).validate()


def _dependencies(settings: PhotoModelSettings) -> tuple[dict[str, str], list[str]]:
    found: dict[str, str] = {}
    missing: list[str] = []
    names = []
    if settings.engine == "pycolmap":
        if importlib.util.find_spec("pycolmap") is None:
            missing.append("pycolmap (pip install pycolmap)")
        else:
            found["pycolmap"] = "python-module"
    else:
        names.append(settings.colmap)
    if settings.matching_device == "mps":
        for name in ("numpy", "torch", "lightglue"):
            if importlib.util.find_spec(name) is None:
                missing.append(f"{name} (MPS matching dependency)")
            else:
                found[name] = "python-module"
    if settings.dense_backend == "openmvs":
        names.extend(("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh"))
        names.append("RefineMesh")
        if settings.texture:
            names.append("TextureMesh")
    else:
        names.append(settings.colmap)
    names.append(settings.blender)
    for name in dict.fromkeys(names):
        executable = shutil.which(name)
        if executable:
            found[name] = executable
        else:
            missing.append(name)
    return found, missing


def plan_photo_model(references: ReferenceSet, output: str | Path,
                     settings: PhotoModelSettings, *,
                     dependencies: tuple[dict[str, str], list[str]] | None = None) -> dict[str, Any]:
    """Read-only intake and tool preflight. Photo count is not overlap evidence."""
    settings.validate()
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists():
        raise ReferenceError(f"Photo-model output already exists: {target}")
    if any(target == image.path.parent or target.is_relative_to(image.path.parent)
           for image in references.images):
        raise ReferenceError("Photo-model output must be outside the input photo directories")
    audit = audit_reference_set(references)
    if audit["independent_images"] < 8:
        raise ReferenceError("Photo-to-GLB requires at least eight distinct photographs")
    found, missing = dependencies if dependencies is not None else _dependencies(settings)
    return {
        "schema_version": 1, "asset_id": references.asset_id,
        "output": str(target), "source_images": audit["source_images"],
        "independent_images": audit["independent_images"],
        "source_manifest_sha256": references.manifest_sha256,
        "missing_photo_rights": any(image.source_url == "local-unattributed"
                                    for image in references.images),
        "sparse_engine": settings.engine, "matching_preset": settings.matching_preset,
        "matching_device": settings.matching_device,
        "memory_controls": {"feature_threads": settings.feature_threads,
                            "matching_threads": settings.matching_threads,
                            "sift_first_octave": settings.sift_first_octave,
                            "max_colmap_rss_mb": settings.max_colmap_rss_mb,
                            "matching_block_size": settings.matching_block_size,
                            "matching_max_features": settings.matching_max_features,
                            "matching_pairs_per_process": settings.matching_pairs_per_process},
        "dense_backend": settings.dense_backend,
        "min_registered_images": settings.min_registered_images,
        "min_registered_fraction": settings.min_registered_fraction,
        "texture_requested": settings.texture,
        "render_previews": settings.render_previews,
        "detected_dependencies": found, "missing_dependencies": missing,
        "status": "blocked-missing-dependencies" if missing else "ready-to-reconstruct",
        "distribution_allowed": False, "historical_fidelity_verified": False,
        "limitations": "Input coverage does not guarantee camera overlap. A GLB is created only when real sparse registration, dense meshing and Blender export succeed. Scale, anatomy and rights still need review.",
    }


def _default_blender_runner(command: list[str], log: Path, timeout: int) -> int:
    try:
        with log.open("w", encoding="utf-8") as stream:
            return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False).returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nBlender export failed: {error}\n")
        return 124 if isinstance(error, subprocess.TimeoutExpired) else 127


def build_photo_model(
    references: ReferenceSet, output: str | Path, settings: PhotoModelSettings,
    *, dependencies: tuple[dict[str, str], list[str]] | None = None,
    sparse_builder: Callable[..., dict[str, Any]] | None = None,
    dense_builder: Callable[..., dict[str, Any]] | None = None,
    blender_runner: Callable[[list[str], Path, int], int] | None = None,
) -> dict[str, Any]:
    """Build an immutable private GLB revision, checkpointing every stage."""
    plan = plan_photo_model(references, output, settings, dependencies=dependencies)
    if plan["missing_dependencies"]:
        raise ReferenceError("Cannot build photo model; missing: " + ", ".join(plan["missing_dependencies"]))
    entry = Path(__file__).resolve().parents[3] / "scripts/blender_entry.py"
    if not entry.is_file():
        raise ReferenceError("Blender GLB export requires a source checkout with scripts/blender_entry.py")
    target = Path(plan["output"])
    target.mkdir(parents=True, exist_ok=False)
    (target / "logs").mkdir()
    report = {**plan, "status": "review-required", "completed_stages": [],
              "failed_stage": None, "model_glb": None}

    def checkpoint() -> dict[str, Any]:
        (target / "photo-model-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    def fail(stage: str, reason: str) -> dict[str, Any]:
        report.update(status="review-required", failed_stage=stage, error=reason)
        return checkpoint()

    checkpoint()
    from .colmap import build_sparse_reconstruction

    sparse_fn = sparse_builder or build_sparse_reconstruction
    dense_fn = dense_builder or build_dense_reconstruction
    try:
        sparse = sparse_fn(references, target / "sparse", colmap=settings.colmap,
                           engine=settings.engine, timeout_seconds=settings.sparse_timeout_seconds,
                           undistort=True, max_image_size=settings.max_image_size,
                           matching_preset=settings.matching_preset,
                           matching_device=settings.matching_device,
                           feature_threads=settings.feature_threads,
                           matching_threads=settings.matching_threads,
                           sift_first_octave=settings.sift_first_octave,
                           max_colmap_rss_mb=settings.max_colmap_rss_mb,
                           matching_block_size=settings.matching_block_size,
                           matching_max_features=settings.matching_max_features,
                           matching_pairs_per_process=settings.matching_pairs_per_process)
        report["sparse"] = {key: sparse.get(key) for key in
                            ("status", "registered_images", "staged_independent_images",
                             "failed_stage", "error")}
        checkpoint()
        registered = sparse.get("registered_images")
        staged = sparse.get("staged_independent_images")
        if (not str(sparse.get("status", "")).startswith("candidate")
                or type(registered) is not int or type(staged) is not int or staged < registered
                or registered < settings.min_registered_images
                or registered / max(staged, 1) < settings.min_registered_fraction
                or sparse.get("undistorted_images") != registered):
            return fail("sparse-registration", "Not enough overlapping photos registered and undistorted; inspect sparse/reconstruction-report.json before attempting a mesh")
        report["completed_stages"].append("sparse-registration")
        checkpoint()

        dense_settings = DenseSettings(backend=settings.dense_backend,
                                       min_registered_images=settings.min_registered_images,
                                       min_registered_fraction=settings.min_registered_fraction,
                                       timeout_seconds=settings.dense_timeout_seconds,
                                       texture=settings.texture)
        dense = dense_fn(target / "sparse", target / "dense", dense_settings)
        report["dense"] = {key: dense.get(key) for key in
                           ("status", "failed_stage", "error", "mesh")}
        checkpoint()
        if not str(dense.get("status", "")).startswith("dense-mesh-candidate"):
            return fail("dense-mesh", "Dense reconstruction did not produce a validated surface; inspect dense/dense-report.json")
        report["completed_stages"].append("dense-mesh")
        checkpoint()

        command = [plan["detected_dependencies"][settings.blender],
                   "--background", "--factory-startup", "--python-exit-code", "1",
                   "--python", str(entry), "--", "render-dense-blender-review",
                   "--dense-revision", str(target / "dense"), "--output", str(target / "review"),
                   "--matched-views", str(min(8, registered) if settings.render_previews else 0),
                   "--matched-resolution-percent", "65"]
        if settings.texture and settings.dense_backend == "openmvs":
            command.append("--textured")
        if not settings.render_previews:
            command.append("--no-renders")
        runner = blender_runner or _default_blender_runner
        code = runner(command, target / "logs/blender-export.log",
                      settings.dense_timeout_seconds)
        report["blender_exit_code"] = code
        checkpoint()
        record_path = target / "review/blender-review-report.json"
        if code or not record_path.is_file() or record_path.is_symlink():
            return fail("blender-export", "Blender export failed; inspect logs/blender-export.log")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        name = record.get("review_glb")
        expected = ("private-textured-review.glb" if settings.texture
                    and settings.dense_backend == "openmvs" else "private-dense-review.glb")
        path = target / "review" / expected
        if (not str(record.get("status", "")).startswith("private-Blender-mesh-candidate")
                or name != expected or path.is_symlink() or not path.is_file()
                or path.stat().st_size == 0):
            return fail("blender-export", "Blender did not produce an audited review GLB")
        # A successful subprocess alone does not prove that a portable mesh GLB
        # exists. Re-audit before handing an artifact to downstream consumers.
        try:
            report["glb_audit"] = audit_glb(path)
        except AuditError as error:
            return fail("blender-export", f"Exported GLB failed format/mesh audit: {error}")
        shutil.copyfile(path, target / "model.glb")
        with (target / "model.glb").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report.update(status="private-GLB-candidate / visual-and-rights-review-required",
                      model_glb="model.glb", model_sha256=digest,
                      review_blend=str(Path("review") / record.get("review_blend", "")))
        report["completed_stages"].append("blender-export")
        return checkpoint()
    except (ReferenceError, OSError, ValueError, RuntimeError, KeyError) as error:
        stage = ("sparse-registration" if "sparse-registration" not in report["completed_stages"]
                 else "dense-mesh" if "dense-mesh" not in report["completed_stages"]
                 else "blender-export")
        return fail(stage, str(error))
