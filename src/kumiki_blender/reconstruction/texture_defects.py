"""Conservative multi-view photo artifact candidates, never automatic retouching.

An outlier in one UV atlas against three independently visible, registered
camera views may be baked lighting, glare, a projection fault or real damage.
Candidates require artist review; no source texture is ever modified.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .blender_dense import _textured_obj_plan
from .dense import _json, _sha256
from .references import ReferenceError
from .surface_finish import load_surface_recipe
from .uv_coverage import rasterize_uv_coverage


@dataclass(frozen=True)
class DefectDetectionOptions:
    color_difference: int = 45
    local_excess: int = 20
    spatial_radius: int = 12
    max_side: int = 8192
    max_mesh_faces: int = 5_000_000

    def validate(self) -> "DefectDetectionOptions":
        for name, low, high in (("color_difference", 16, 128),
                                ("local_excess", 8, 96),
                                ("spatial_radius", 2, 64),
                                ("max_side", 64, 8192),
                                ("max_mesh_faces", 1, 5_000_000)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ReferenceError(f"{name} must be an integer in [{low}, {high}]")
        return self


def _inputs(dense_revision: str | Path, surface_recipe: str | Path,
            output: str | Path, options: DefectDetectionOptions
            ) -> tuple[Path, Path, dict[str, Any], dict[str, Path], dict[str, Any]]:
    from PIL import Image

    options.validate()
    source_input, target_input, guide_input = Path(dense_revision), Path(output), Path(surface_recipe)
    source, target = source_input.resolve(), target_input.resolve()
    if (source_input.is_symlink() or not source.is_dir() or target_input.is_symlink()
            or target.exists() or target.is_relative_to(source) or source.is_relative_to(target)
            or not target.parent.is_dir() or guide_input.is_symlink() or not guide_input.is_file()):
        raise ReferenceError("Defect detection needs a valid dense revision, surface recipe and unused output")
    report = _json(source / "dense-report.json")
    if not str(report.get("status", "")).startswith("dense-mesh-candidate"):
        raise ReferenceError("Defect detection requires a completed dense-mesh candidate")
    pinned = _textured_obj_plan(source, report, options.max_mesh_faces)
    atlases = {Path(item["path"]).name: Path(item["path"])
               for item in pinned["texture_atlases"]}
    if len(atlases) != len(pinned["texture_atlases"]):
        raise ReferenceError("Material texture atlas filenames must be unique")
    specs = load_surface_recipe(surface_recipe, atlases)
    total_pixels = 0
    for name, path in atlases.items():
        with Image.open(path) as image:
            if image.format not in ("PNG", "JPEG") or max(image.size) > options.max_side:
                raise ReferenceError("Defect candidate source must be a PNG/JPEG within max_side")
            total_pixels += image.width * image.height * len(specs[name].observations)
        if len(specs[name].observations) < 3:
            raise ReferenceError("Defect detection requires 3–8 registered UV photo observations per atlas")
    if total_pixels > 512_000_000:
        raise ReferenceError("Defect evidence exceeds 512 million atlas-view texels")
    return source, target, pinned, atlases, specs


def detect_texture_defects(dense_revision: str | Path, surface_recipe: str | Path,
                           output: str | Path, *, options: DefectDetectionOptions = DefectDetectionOptions()
                           ) -> dict[str, Any]:
    """Emit per-atlas candidate masks and QA; never apply inpainting implicitly."""
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter
    import numpy as np

    source, target, pinned, atlases, specs = _inputs(
        dense_revision, surface_recipe, output, options)
    report_sha, recipe_sha = (_sha256(source / "dense-report.json"),
                             _sha256(Path(surface_recipe)))
    original_sha = {name: _sha256(path) for name, path in atlases.items()}
    sizes: dict[str, tuple[int, int]] = {}
    for name, path in atlases.items():
        with Image.open(path) as photo:
            sizes[name] = photo.size
    coverage, faces = rasterize_uv_coverage(Path(pinned["source_obj"]),
                                           Path(pinned["source_mtl"]), sizes)
    if (_sha256(source / "dense-report.json") != report_sha
            or _sha256(Path(surface_recipe)) != recipe_sha):
        raise ReferenceError("Defect sources changed during preflight")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-defects-", dir=target.parent))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "photographic-texture-defect-candidates / manual-review-required",
        "distribution_allowed": False,
        "automatic_cleanup_performed": False,
        "source_dense_revision": str(source),
        "source_dense_report_sha256": report_sha,
        "source_obj_sha256": pinned["source_obj_sha256"],
        "source_surface_recipe_sha256": recipe_sha,
        "source_atlas_sha256": original_sha,
        "options": {field: getattr(options, field) for field in options.__dataclass_fields__},
        "atlases": {},
        "limitations": ("Outliers relative to registered UV views are not proof of photographic damage. "
                        "True cracks, patina and consistent defects may be invisible to this test. "
                        "Shared lighting, camera pose or imperfect reprojection can produce false positives. "
                        "Manually edit each candidate mask before passing it to surface finishing."),
    }
    try:
        for index, (name, atlas_path) in enumerate(atlases.items(), start=1):
            if _sha256(atlas_path) != original_sha[name]:
                raise ReferenceError("Texture atlas changed during defect analysis")
            spec = specs[name]
            from .multiview_fusion import fuse_uv_photo_observations

            with Image.open(atlas_path) as raw:
                photo = raw.copy()
            fused, fusion_quality = fuse_uv_photo_observations(
                photo, spec.regions, spec.observations,
                strength=1.0, max_ev=2.0, protect_black=12)
            source_rgb = photo.convert("RGB")
            corrected_rgb = fused.convert("RGB")
            delta = ImageChops.difference(source_rgb, corrected_rgb)
            red, green, blue = delta.split()
            max_delta = ImageChops.lighter(ImageChops.lighter(red, green), blue)
            local = max_delta.filter(ImageFilter.GaussianBlur(options.spatial_radius))
            excess = ImageChops.subtract(max_delta, local)
            # Disallow UV seams and unclassified texels: only interior of a
            # reviewed material region can be proposed for artifact cleanup.
            material_interior = Image.new("L", photo.size, 0)
            for region in spec.regions:
                mask = region.mask.open_verified().filter(ImageFilter.MinFilter(3))
                material_interior = ImageChops.lighter(material_interior, mask)
            valid = ImageChops.multiply(coverage[name], material_interior)
            if "A" in photo.getbands():
                valid = ImageChops.multiply(valid, photo.getchannel("A"))
            valid = ImageChops.multiply(valid, source_rgb.convert("L").point(
                lambda value: 255 if value > 16 else 0))
            # Fusion intentionally preserves pixels with <3 independent views.
            # Count support ourselves, rather than treating unchanged pixels
            # as proof that a pixel was confidently observed.
            support = np.zeros((photo.height, photo.width), dtype=np.uint8)
            for observation in spec.observations:
                view = observation.photo.open_verified()
                confidence = np.asarray(observation.confidence.open_verified()) >= 128
                photo_valid = (np.asarray(view.convert("RGB")).max(axis=2) >= 16)
                if "A" in view.getbands():
                    photo_valid &= np.asarray(view.getchannel("A")) >= 128
                support += (confidence & photo_valid).astype(np.uint8)
                view.close()
            considered = (np.asarray(valid) >= 128) & (support >= 3)
            score = (np.asarray(max_delta) >= options.color_difference)
            localized = (np.asarray(excess) >= options.local_excess)
            candidate = considered & score & localized
            page = stage / f"atlas-{index:02d}"
            page.mkdir()
            mask_path = page / "candidate-damage-mask.png"
            Image.fromarray(np.where(candidate, 255, 0).astype(np.uint8), "L").save(mask_path)
            severity = np.where(candidate, np.asarray(excess), 0).astype(np.uint8)
            severity_path = page / "defect-severity.png"
            Image.fromarray(severity, "L").save(severity_path)
            overlay = ImageEnhance.Color(source_rgb).enhance(.7).convert("RGBA")
            red_layer = Image.new("RGBA", photo.size, (240, 60, 58, 150))
            overlay.paste(red_layer, mask=Image.fromarray(np.where(candidate, 160, 0).astype(np.uint8), "L"))
            overlay_path = page / "review-overlay.png"
            overlay.convert("RGB").save(overlay_path)
            total = photo.width * photo.height
            n_candidates = int(np.count_nonzero(candidate))
            report["atlases"][name] = {
                "source_sha256": original_sha[name],
                "observation_count": len(spec.observations),
                "uv_face_count": faces[name],
                "supported_texels": int(np.count_nonzero(considered)),
                "candidate_texels": n_candidates,
                "candidate_fraction": round(n_candidates / total, 6),
                "cleanup_area_within_five_percent": n_candidates <= total * .05,
                "candidate_mask": {"path": str(mask_path.relative_to(stage)),
                                   "sha256": _sha256(mask_path)},
                "severity": {"path": str(severity_path.relative_to(stage)),
                             "sha256": _sha256(severity_path)},
                "overlay": {"path": str(overlay_path.relative_to(stage)),
                            "sha256": _sha256(overlay_path)},
                "fusion": fusion_quality,
            }
            if not n_candidates:
                report["atlases"][name]["note"] = "No localized outliers met thresholds; this is not proof of a defect-free texture"
            photo.close()
            fused.close()
        manifest = stage / "texture-defect-report.json"
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if target.exists():
            raise ReferenceError("Defect output appeared during processing; refusing overwrite")
        os.rename(stage, target)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
