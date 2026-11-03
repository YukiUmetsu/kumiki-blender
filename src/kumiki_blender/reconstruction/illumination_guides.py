"""Conservative UV illumination guidance from independently registered views.

This is an editable photographic candidate, not physical intrinsic-image
decomposition. Registered photo disagreement can reveal *relative* baked
lighting; persistent shadows and unobserved surfaces cannot be recovered.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .blender_dense import _textured_obj_plan
from .dense import _json, _sha256
from .material_separation import _copied_evidence, _copied_optional
from .references import ReferenceError
from .surface_finish import load_surface_recipe
from .uv_coverage import rasterize_uv_coverage


@dataclass(frozen=True)
class IlluminationGuideOptions:
    smoothing_radius: int = 32
    max_illumination_ev: float = 1.5
    max_side: int = 8192
    max_mesh_faces: int = 5_000_000

    def validate(self) -> "IlluminationGuideOptions":
        if type(self.smoothing_radius) is not int or not 4 <= self.smoothing_radius <= 256:
            raise ReferenceError("smoothing_radius must be an integer in [4, 256]")
        if (type(self.max_illumination_ev) not in (int, float)
                or not math.isfinite(self.max_illumination_ev)
                or not .25 <= self.max_illumination_ev <= 2.0):
            raise ReferenceError("max_illumination_ev must be finite in [0.25, 2.0]")
        if type(self.max_side) is not int or not 64 <= self.max_side <= 8192:
            raise ReferenceError("max_side must be an integer in [64, 8192]")
        if type(self.max_mesh_faces) is not int or not 1 <= self.max_mesh_faces <= 5_000_000:
            raise ReferenceError("max_mesh_faces must be in [1, 5000000]")
        return self


def _linear_luminance(rgb: Any, np: Any) -> Any:
    sample = rgb.astype(np.float32) / 255
    linear = np.where(sample <= .04045, sample / 12.92,
                      ((sample + .055) / 1.055) ** 2.4)
    return linear[..., 0] * .2126 + linear[..., 1] * .7152 + linear[..., 2] * .0722


def estimate_uv_illumination(dense_revision: str | Path, surface_recipe: str | Path,
                             output: str | Path,
                             *, options: IlluminationGuideOptions = IlluminationGuideOptions()
                             ) -> dict[str, Any]:
    """Create hash-pinned per-material illumination maps for artist review."""
    from PIL import Image, ImageFilter
    import numpy as np

    options.validate()
    source_input, recipe_input, target_input = (Path(dense_revision), Path(surface_recipe),
                                               Path(output))
    source, target = source_input.resolve(), target_input.resolve()
    if (source_input.is_symlink() or not source.is_dir() or recipe_input.is_symlink()
            or not recipe_input.is_file() or target_input.is_symlink() or target.exists()
            or target.is_relative_to(source) or source.is_relative_to(target)
            or not target.parent.is_dir()):
        raise ReferenceError("Illumination guidance requires a dense revision, existing recipe and unused output")
    dense_report = _json(source / "dense-report.json")
    if not str(dense_report.get("status", "")).startswith("dense-mesh-candidate"):
        raise ReferenceError("Illumination guidance requires a completed dense-mesh candidate")
    pinned = _textured_obj_plan(source, dense_report, options.max_mesh_faces)
    atlases = {Path(item["path"]).name: Path(item["path"])
               for item in pinned["texture_atlases"]}
    if len(atlases) != len(pinned["texture_atlases"]):
        raise ReferenceError("Atlas filenames must be unique")
    specs = load_surface_recipe(surface_recipe, atlases)
    sizes: dict[str, tuple[int, int]] = {}
    pixels = 0
    for name, atlas in atlases.items():
        with Image.open(atlas) as original:
            if original.format not in ("JPEG", "PNG") or max(original.size) > options.max_side:
                raise ReferenceError("Illumination source must be a PNG/JPEG within max_side")
            sizes[name] = original.size
            pixels += original.width * original.height * len(specs[name].observations)
        if len(specs[name].observations) < 3:
            raise ReferenceError("Illumination estimation requires 3–8 UV-registered observations")
    if pixels > 512_000_000:
        raise ReferenceError("Illumination evidence exceeds 512 million atlas-view texels")
    source_sha = {name: _sha256(path) for name, path in atlases.items()}
    recipe_sha = _sha256(recipe_input)
    report_sha = _sha256(source / "dense-report.json")
    mapped, faces = rasterize_uv_coverage(Path(pinned["source_obj"]),
                                          Path(pinned["source_mtl"]), sizes)
    if _sha256(recipe_input) != recipe_sha or _sha256(source / "dense-report.json") != report_sha:
        raise ReferenceError("Photographic source changed during illumination preflight")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-illumination-", dir=target.parent))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "relative-uv-illumination-candidate / artistic-review-required",
        "distribution_allowed": False,
        "source_dense_revision": str(source),
        "source_dense_report_sha256": report_sha,
        "source_obj_sha256": pinned["source_obj_sha256"],
        "source_atlas_sha256": source_sha,
        "source_surface_recipe_sha256": recipe_sha,
        "options": {name: getattr(options, name) for name in options.__dataclass_fields__},
        "atlases": {},
        "files": [],
        "limitations": ("Guidance is a smoothed relative illumination *candidate* based on multiview median, "
                        "not calibrated physical albedo. It cannot remove shared or occluded shadows, "
                        "recover burned-out detail, or determine surface roughness. Reject misregistered "
                        "views, color casts and transient defects before approving the map."),
    }
    recipe: dict[str, Any] = {"schema_version": 1, "atlases": {}}
    try:
        from .multiview_fusion import fuse_uv_photo_observations

        for index, (name, atlas_path) in enumerate(atlases.items(), start=1):
            if _sha256(atlas_path) != source_sha[name]:
                raise ReferenceError("Atlas changed while deriving illumination")
            page = f"atlas-{index:02d}"
            spec = specs[name]
            with Image.open(atlas_path) as original:
                photo = original.copy()
            fused, fusion_report = fuse_uv_photo_observations(
                photo, spec.regions, spec.observations,
                strength=1.0, max_ev=options.max_illumination_ev, protect_black=12)
            rgb = photo.convert("RGB")
            restored = fused.convert("RGB")
            # Count real independent UV visibility; fusion intentionally keeps
            # unsupported pixels unchanged, which is not evidence of albedo.
            support = np.zeros((photo.height, photo.width), dtype=np.uint8)
            for observation in spec.observations:
                view = observation.photo.open_verified()
                conf = np.asarray(observation.confidence.open_verified()) >= 128
                valid = np.asarray(view.convert("RGB")).max(axis=2) >= 16
                if "A" in view.getbands():
                    valid &= np.asarray(view.getchannel("A")) >= 128
                support += (valid & conf).astype(np.uint8)
                view.close()
            uv_mapped = np.asarray(mapped[name]) >= 128
            covered = uv_mapped & (support >= 3)
            orig_alpha = (np.asarray(photo.convert("RGBA").getchannel("A")) >= 128
                          if "A" in photo.getbands() else np.ones(covered.shape, dtype=bool))
            log_ev = np.zeros(covered.shape, dtype=np.float32)
            for y0 in range(0, photo.height, 128):
                y1 = min(photo.height, y0 + 128)
                box = (0, y0, photo.width, y1)
                base = np.asarray(rgb.crop(box))
                consensus = np.asarray(restored.crop(box))
                covered[y0:y1] &= orig_alpha[y0:y1] & (base.max(axis=2) >= 16)
                luma = _linear_luminance(base, np)
                target_luma = _linear_luminance(consensus, np)
                log_ev[y0:y1] = np.clip(np.log2((luma + .015) / (target_luma + .015)),
                                         -options.max_illumination_ev, options.max_illumination_ev)
            field = np.full(covered.shape, 128, dtype=np.uint8)
            valid_guide = np.zeros(covered.shape, dtype=np.uint8)
            material_qa = []
            for region in spec.regions:
                material = np.asarray(region.mask.open_verified()) >= 128
                valid = material & covered
                samples = log_ev[valid]
                if samples.size < 64:
                    material_qa.append({"id": region.id, "supported_texels": int(samples.size),
                                        "status": "insufficient-coverage / neutral-only"})
                    continue
                # Normalize independently per material, preserving actual
                # material brightness while removing within-region gradients.
                baseline = float(np.median(samples))
                scaled = np.clip((log_ev - baseline) * 56 + 128, 1, 255).astype(np.uint8)
                active = Image.fromarray((valid * 255).astype(np.uint8), "L")
                weighted = Image.fromarray(np.where(valid, scaled, 0).astype(np.uint8), "L")
                weighted_smooth = np.asarray(weighted.filter(
                    ImageFilter.GaussianBlur(options.smoothing_radius)), dtype=np.float32)
                coverage_smooth = np.asarray(active.filter(
                    ImageFilter.GaussianBlur(options.smoothing_radius)), dtype=np.float32)
                reliable = valid & (coverage_smooth >= 24)
                ev_smooth = np.clip((weighted_smooth * 255 /
                                     np.maximum(coverage_smooth, 1) - 128) / 56,
                                    -options.max_illumination_ev, options.max_illumination_ev)
                # The surface finisher interprets 128 as neutral illumination
                # and divides by the encoded multiplier in linear RGB.
                guide = np.uint8(np.clip(128 * np.exp2(ev_smooth) + .5, 16, 255))
                field[reliable] = guide[reliable]
                valid_guide[reliable] = 255
                material_qa.append({"id": region.id, "supported_texels": int(samples.size),
                                    "guided_texels": int(np.count_nonzero(reliable)),
                                    "median_relative_ev_before_normalization": round(baseline, 5),
                                    "p05_ev": round(float(np.percentile(samples, 5)), 5),
                                    "p95_ev": round(float(np.percentile(samples, 95)), 5),
                                    "positive_ev_clip_fraction": round(float(np.count_nonzero(
                                        reliable & (ev_smooth > .99))) /
                                        max(1, np.count_nonzero(reliable)), 6),
                                    "status": "candidate / review-required"})
            page_dir = stage / page
            page_dir.mkdir()
            guide_path = page_dir / "illumination-map.png"
            visibility_path = page_dir / "guided-coverage.png"
            Image.fromarray(field, "L").save(guide_path)
            Image.fromarray(valid_guide, "L").save(visibility_path)
            preview = rgb.copy()
            preview.thumbnail((768, 768))
            preview_size = preview.size
            guide_preview = Image.fromarray(field, "L").resize(preview_size, Image.Resampling.BILINEAR)
            guide_factor = np.asarray(guide_preview, dtype=np.float32) / 128
            base = np.asarray(preview, dtype=np.float32) / 255
            lin = np.where(base <= .04045, base / 12.92, ((base + .055) / 1.055) ** 2.4)
            corrected = np.clip(lin / np.maximum(guide_factor[..., None], .125), 0, 1)
            corrected = np.where(corrected <= .0031308, corrected * 12.92,
                                 1.055 * corrected ** (1 / 2.4) - .055)
            review = Image.new("RGB", (preview.width * 3, preview.height))
            review.paste(preview, (0, 0))
            review.paste(restored.resize(preview_size, Image.Resampling.BILINEAR), (preview.width, 0))
            review.paste(Image.fromarray(np.uint8(np.clip(corrected * 255 + .5, 0, 255)), "RGB"),
                         (preview.width * 2, 0))
            preview_path = page_dir / "before-fused-corrected.png"
            review.save(preview_path)
            for dest in (guide_path, visibility_path, preview_path):
                report["files"].append({"path": str(dest.relative_to(stage)),
                                        "sha256": _sha256(dest)})
            atlas_entry = _copied_evidence(replace(spec, illumination_map=None), stage,
                                           page, report["files"])
            atlas_entry["illumination_map"] = str(guide_path.relative_to(stage))
            atlas_entry["regions"] = []
            for region in spec.regions:
                r: dict[str, Any] = {
                    "id": region.id,
                    "mask": _copied_optional(region.mask, stage,
                                              f"{page}/regions/{region.id}.png", report["files"]),
                    "roughness": region.roughness,
                }
                if region.roughness_map is not None:
                    r["roughness_map"] = _copied_optional(
                        region.roughness_map, stage,
                        f"{page}/regions/{region.id}-roughness.png", report["files"])
                atlas_entry["regions"].append(r)
            recipe["atlases"][name] = atlas_entry
            report["atlases"][name] = {
                "view_count": len(spec.observations), "uv_faces": faces[name],
                "supported_fraction_of_uv": round(float(np.count_nonzero(covered)) /
                                                   max(1, np.count_nonzero(uv_mapped)), 6),
                "guided_fraction_of_uv": round(float(np.count_nonzero(valid_guide)) /
                                                max(1, np.count_nonzero(uv_mapped)), 6),
                "materials": material_qa,
                "previous_illumination_map_sha256": (spec.illumination_map.sha256
                                                     if spec.illumination_map else None),
                "fusion": fusion_report,
            }
            photo.close()
            fused.close()
        recipe_path = stage / "surface-recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        load_surface_recipe(recipe_path, atlases)
        report["files"].append({"path": recipe_path.name, "sha256": _sha256(recipe_path)})
        report_path = stage / "illumination-guide-report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if target.exists():
            raise ReferenceError("Illumination output appeared during processing")
        os.rename(stage, target)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
