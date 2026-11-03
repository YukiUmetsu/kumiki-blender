"""Reviewable material-region proposals from UV-registered photographic evidence.

Seed masks are *examples*, not a physical material measurement. Multi-angle
median observations reduce transient shadows, but ambiguity is left unassigned.
All inputs are hash-pinned; no source atlas or observation is ever rewritten.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .blender_dense import _textured_obj_plan
from .dense import _json, _sha256
from .references import ReferenceError
from .surface_finish import AtlasSurfaceSpec, PinnedSurfaceImage, load_surface_recipe
from .uv_coverage import rasterize_uv_coverage


_COLORS = ((231, 84, 87), (67, 157, 229), (91, 197, 130),
           (236, 183, 75), (172, 115, 229), (71, 210, 203),
           (221, 116, 179), (202, 207, 84))


@dataclass(frozen=True)
class SeparationOptions:
    min_margin: float = .25
    max_distance: float = 3.5
    min_evidence_views: int = 2
    max_side: int = 8192
    max_mesh_faces: int = 5_000_000

    def validate(self) -> "SeparationOptions":
        for name, minimum, maximum in (("min_margin", .05, .8),
                                        ("max_distance", 1.0, 10.0)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
                raise ReferenceError(f"{name} must be finite in [{minimum}, {maximum}]")
        if type(self.min_evidence_views) is not int or not 2 <= self.min_evidence_views <= 8:
            raise ReferenceError("min_evidence_views must be an integer in [2, 8]")
        if type(self.max_side) is not int or not 64 <= self.max_side <= 8192:
            raise ReferenceError("max_side must be an integer in [64, 8192]")
        if type(self.max_mesh_faces) is not int or not 1 <= self.max_mesh_faces <= 5_000_000:
            raise ReferenceError("max_mesh_faces must be an integer in [1, 5000000]")
        return self


def _pin_dense(dense_revision: str | Path, output: str | Path,
               options: SeparationOptions) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    source_input, target_input = Path(dense_revision), Path(output)
    source, target = source_input.resolve(), target_input.resolve()
    if (source_input.is_symlink() or target_input.is_symlink() or not source.is_dir()
            or target.exists() or source == target or target.is_relative_to(source)
            or source.is_relative_to(target)):
        raise ReferenceError("Material separation requires an existing dense revision and a new external output")
    report = _json(source / "dense-report.json")
    if not str(report.get("status", "")).startswith("dense-mesh-candidate"):
        raise ReferenceError("Material separation needs a completed dense-mesh candidate")
    pinned = _textured_obj_plan(source, report, options.max_mesh_faces)
    atlases = {Path(item["path"]).name: Path(item["path"])
               for item in pinned["texture_atlases"]}
    if len(atlases) != len(pinned["texture_atlases"]):
        raise ReferenceError("Texture atlas basenames must be unique")
    if not target.parent.is_dir():
        raise ReferenceError("Material proposal output parent must already exist")
    return source, target, pinned, atlases


def plan_material_separation(dense_revision: str | Path, seed_recipe: str | Path,
                             output: str | Path, *, options: SeparationOptions = SeparationOptions()
                             ) -> dict[str, Any]:
    """Validate a normal surface recipe whose masks contain material seed strokes."""
    from PIL import Image, ImageChops

    options.validate()
    source, target, pinned, atlases = _pin_dense(dense_revision, output, options)
    if (Path(seed_recipe).is_symlink() or not Path(seed_recipe).is_file()
            or Path(seed_recipe).resolve().is_relative_to(target)):
        raise ReferenceError("Seed surface recipe must be a separate existing local JSON file")
    specs = load_surface_recipe(seed_recipe, atlases)
    total_pixels = 0
    sizes: dict[str, tuple[int, int]] = {}
    for name, atlas in atlases.items():
        with Image.open(atlas) as original:
            width, height = original.size
            if original.format not in ("PNG", "JPEG") or max(width, height) > options.max_side:
                raise ReferenceError("Source atlas must be PNG/JPEG within max_side")
            total_pixels += width * height * max(1, len(specs[name].observations))
            sizes[name] = (width, height)
        if not 2 <= len(specs[name].regions) <= len(_COLORS):
            raise ReferenceError("Each atlas must have 2–8 seed material regions")
        if specs[name].observations and len(specs[name].observations) < options.min_evidence_views:
            raise ReferenceError("Not enough registered observations for min_evidence_views")
        for region in specs[name].regions:
            with Image.open(region.mask.path) as image:
                area = image.histogram()
                if sum(area[1:255]):
                    raise ReferenceError("Material seed masks must be binary (0/255) PNGs")
                if area[255] < 16:
                    raise ReferenceError(f"Material seed {region.id} needs at least 16 labeled texels")
    if total_pixels > 512_000_000:
        raise ReferenceError("Material separation exceeds 512 million projected atlas texels")
    coverage, faces = rasterize_uv_coverage(
        Path(pinned["source_obj"]), Path(pinned["source_mtl"]),
        sizes)
    for name, spec in specs.items():
        with Image.open(atlases[name]) as atlas:
            opaque = atlas.convert("RGBA").getchannel("A")
            colored = atlas.convert("RGB").convert("L").point(
                lambda value: 255 if value >= 12 else 0)
            valid = ImageChops.multiply(ImageChops.multiply(coverage[name], opaque), colored)
        for region in spec.regions:
            seed = region.mask.open_verified()
            if ImageChops.subtract(seed, valid).getbbox():
                raise ReferenceError(f"Material seed {region.id} extends beyond valid photographed OBJ UV texels")
    return {
        "schema_version": 1,
        "status": "seed-guided-material-proposal / review-required",
        "source_dense_revision": str(source),
        "source_dense_report_sha256": _sha256(source / "dense-report.json"),
        "source_obj_sha256": pinned["source_obj_sha256"],
        "source_atlas_sha256": {name: _sha256(path) for name, path in atlases.items()},
        "seed_recipe_sha256": _sha256(Path(seed_recipe)),
        "seed_recipe": str(Path(seed_recipe).resolve()),
        "output": str(target),
        "options": {field: getattr(options, field) for field in options.__dataclass_fields__},
        "atlases": {name: {"regions": [item.id for item in spec.regions],
                           "views": [view.id for view in spec.observations],
                           "uv_face_count": faces[name]}
                    for name, spec in specs.items()},
    }


def _features(rgb: Any, np: Any) -> Any:
    """Lightness-downweighted linear RGB log-chroma; not measured albedo."""
    srgb = rgb.astype(np.float32) / 255.0
    linear = np.where(srgb <= .04045, srgb / 12.92,
                      ((srgb + .055) / 1.055) ** 2.4)
    red, green, blue = (linear[..., index] for index in range(3))
    luma = .2126 * red + .7152 * green + .0722 * blue
    return np.stack((np.log((red + .012) / (green + .012)),
                     np.log((blue + .012) / (green + .012)),
                     .35 * np.log(luma + .03)), axis=-1)


def _evidence_chunk(atlas: Any, observations: list[tuple[Any, Any]],
                    y0: int, y1: int, minimum_views: int, np: Any) -> tuple[Any, Any, Any]:
    rgb = np.asarray(atlas.crop((0, y0, atlas.width, y1)).convert("RGB"))
    alpha = (np.asarray(atlas.crop((0, y0, atlas.width, y1)).getchannel("A")) >= 128
             if "A" in atlas.getbands() else np.ones(rgb.shape[:2], dtype=bool))
    valid = alpha & (rgb.max(axis=2) >= 12)
    original = _features(rgb, np)
    if not observations:
        return original, valid, np.ones(valid.shape, dtype=np.uint8)
    features = []
    weights = []
    support = np.zeros(valid.shape, dtype=np.uint8)
    for photo, confidence in observations:
        patch = np.asarray(photo.crop((0, y0, atlas.width, y1)).convert("RGB"))
        strength = np.asarray(confidence.crop((0, y0, atlas.width, y1)), dtype=np.float32)
        quality = strength >= 128
        if "A" in photo.getbands():
            quality &= np.asarray(photo.crop((0, y0, atlas.width, y1)).getchannel("A")) >= 128
        quality &= patch.max(axis=2) >= 12
        support += quality.astype(np.uint8)
        features.append(_features(patch, np))
        weights.append(np.where(quality, strength, 0))
    view_features = np.stack(features, axis=0)
    view_weights = np.stack(weights, axis=0)
    # Weighted median is resistant to one strongly shaded view, while giving
    # face-on high-confidence camera projections more influence.
    order = np.argsort(view_features, axis=0)
    sorted_values = np.take_along_axis(view_features, order, axis=0)
    sorted_weights = np.take_along_axis(view_weights[..., None], order, axis=0)
    total = np.sum(sorted_weights, axis=0)
    rank = np.argmax(np.cumsum(sorted_weights, axis=0) >= total[None, ...] * .5,
                     axis=0)
    median = np.take_along_axis(sorted_values, rank[None, ...], axis=0)[0]
    enough = support >= minimum_views
    # Two disagreeing observations cannot identify which one contains shadow
    # or specular glare. Keep the texel unassigned until a third view exists.
    only_two = support == 2
    minimum = np.min(np.where(view_weights[..., None] > 0, view_features, np.inf), axis=0)
    maximum = np.max(np.where(view_weights[..., None] > 0, view_features, -np.inf), axis=0)
    discrepancy = maximum - minimum
    unstable = only_two & ((np.max(discrepancy[..., :2], axis=2) > .3)
                           | (discrepancy[..., 2] > .3))
    return np.where(enough[..., None], median, original), valid & enough & ~unstable, support


def _copied_evidence(spec: AtlasSurfaceSpec, stage: Path, page: str,
                     report_files: list[dict[str, str]]) -> dict[str, Any]:
    """Copy and verify optional observations/maps so output recipe is stand-alone."""
    entry: dict[str, Any] = {}

    def copy(pin: PinnedSurfaceImage, relative: str) -> str:
        data = pin.path.read_bytes()
        if (pin.path.is_symlink() or hashlib.sha256(data).hexdigest() != pin.sha256):
            raise ReferenceError("Surface evidence changed while producing proposal")
        dest = stage / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        report_files.append({"path": relative, "sha256": _sha256(dest),
                             "source_sha256": pin.sha256})
        return relative

    for property_name in ("illumination_map", "damage_mask", "height_map"):
        pin = getattr(spec, property_name)
        if pin is not None:
            entry[property_name] = copy(pin, f"{page}/evidence/{property_name}.png")
    if spec.displacement_scale_m is not None:
        entry["displacement_scale_m"] = spec.displacement_scale_m
    if spec.observations:
        entry["observations"] = []
        for index, obs in enumerate(spec.observations, start=1):
            entry["observations"].append({
                "id": obs.id,
                "photo": copy(obs.photo, f"{page}/evidence/view-{index:02d}.png"),
                "confidence": copy(obs.confidence,
                                   f"{page}/evidence/confidence-{index:02d}.png"),
            })
    return entry


def propose_material_regions(dense_revision: str | Path, seed_recipe: str | Path,
                             output: str | Path, *, options: SeparationOptions = SeparationOptions()
                             ) -> dict[str, Any]:
    """Expand user strokes into conservative, confidence-masked UV material regions."""
    from PIL import Image, ImageChops
    import numpy as np

    plan = plan_material_separation(dense_revision, seed_recipe, output, options=options)
    target = Path(plan["output"])
    source = Path(plan["source_dense_revision"])
    dense_report = _json(source / "dense-report.json")
    pinned = _textured_obj_plan(source, dense_report, options.max_mesh_faces)
    atlases = {Path(item["path"]).name: Path(item["path"])
               for item in pinned["texture_atlases"]}
    specs = load_surface_recipe(seed_recipe, atlases)
    if (_sha256(source / "dense-report.json") != plan["source_dense_report_sha256"]
            or _sha256(Path(seed_recipe)) != plan["seed_recipe_sha256"]):
        raise ReferenceError("Dense or seed recipe changed after preflight")
    # Coverage from actual OBJ UVs, not unused black or transparent padding.
    sizes = {}
    for name, path in atlases.items():
        with Image.open(path) as photo:
            sizes[name] = photo.size
    coverage, _ = rasterize_uv_coverage(Path(pinned["source_obj"]),
                                       Path(pinned["source_mtl"]), sizes)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-materials-", dir=target.parent))
    files: list[dict[str, str]] = []
    recipe: dict[str, Any] = {"schema_version": 1, "atlases": {}}
    report: dict[str, Any] = {**plan, "output": str(target), "files": files,
                              "distribution_allowed": False,
                              "review_required": True,
                              "warnings": ["Material labels are guided photo-color hypotheses, not physical measurements.",
                                           "Shadows, grime, patina and similarly colored materials can be confused.",
                                           "Roughness values are authored priors, not measured reflectance."]}
    try:
        for index, (name, path) in enumerate(atlases.items(), start=1):
            if _sha256(path) != plan["source_atlas_sha256"][name]:
                raise ReferenceError("Dense texture atlas changed after preflight")
            spec, page = specs[name], f"atlas-{index:02d}"
            atlas = Image.open(path)
            atlas.load()
            views = [(obs.photo.open_verified(), obs.confidence.open_verified())
                     for obs in spec.observations]
            seed_images = [region.mask.open_verified() for region in spec.regions]
            seed_arrays = [np.asarray(seed) == 255 for seed in seed_images]
            mapped = np.asarray(coverage[name]) == 255
            prototypes: list[Any] = []
            spreads: list[Any] = []
            # Limit seed samples without biasing toward top rows or one corner.
            samples: list[list[Any]] = [[] for _ in seed_arrays]
            sample_steps = [max(1, int(math.sqrt(float(np.count_nonzero(seed)) / 8000)))
                            for seed in seed_arrays]
            for y0 in range(0, atlas.height, 128):
                y1 = min(y0 + 128, atlas.height)
                features, valid, _ = _evidence_chunk(atlas, views, y0, y1,
                                                    options.min_evidence_views, np)
                for item, seed in enumerate(seed_arrays):
                    step = sample_steps[item]
                    yy = (step - y0 % step) % step
                    pick = seed[y0:y1][yy::step, ::step] & valid[yy::step, ::step] & mapped[y0:y1][yy::step, ::step]
                    if np.any(pick):
                        samples[item].append(features[yy::step, ::step][pick])
            for region, pieces in zip(spec.regions, samples):
                if not pieces or sum(len(part) for part in pieces) < 12:
                    raise ReferenceError(f"Material seed {region.id} lacks 12 valid photographic samples")
                values = np.concatenate(pieces, axis=0)
                center = np.median(values, axis=0)
                prototypes.append(center)
                spreads.append(np.maximum(np.median(np.abs(values - center), axis=0) * 1.4826,
                                          np.array((.08, .08, .10), dtype=np.float32)))
            # A wide prior protects against labeling shadows as different materials.
            scale = np.maximum(np.median(np.stack(spreads), axis=0),
                               np.array((.12, .12, .15), dtype=np.float32))
            for first in range(len(prototypes)):
                for second in range(first + 1, len(prototypes)):
                    if float(np.linalg.norm((prototypes[first] - prototypes[second]) / scale)) < 1.2:
                        raise ReferenceError("Seed materials have indistinguishable photographic colors; add evidence or manual masks")
            labels = np.full((atlas.height, atlas.width), -1, dtype=np.int8)
            confidence = np.zeros(labels.shape, dtype=np.uint8)
            support_count = np.zeros(labels.shape, dtype=np.uint8)
            counts = [0] * len(spec.regions)
            for y0 in range(0, atlas.height, 128):
                y1 = min(y0 + 128, atlas.height)
                features, valid, support = _evidence_chunk(atlas, views, y0, y1,
                                                           options.min_evidence_views, np)
                valid &= mapped[y0:y1]
                distances = np.stack([np.sqrt(np.sum(((features - center) / scale) ** 2, axis=2))
                                      for center in prototypes], axis=0)
                best = np.argmin(distances, axis=0)
                ordered = np.sort(distances, axis=0)
                margin = (ordered[1] - ordered[0]) / np.maximum(ordered[1], .001)
                accept = valid & (ordered[0] <= options.max_distance) & (margin >= options.min_margin)
                chunk_labels = np.where(accept, best, -1).astype(np.int8)
                chunk_conf = np.where(accept, np.minimum(255, margin * 255), 0).astype(np.uint8)
                for material, seed in enumerate(seed_arrays):
                    chosen = seed[y0:y1] & mapped[y0:y1]
                    chunk_labels[chosen] = material
                    chunk_conf[chosen] = 255
                labels[y0:y1] = chunk_labels
                confidence[y0:y1] = chunk_conf
                support_count[y0:y1] = support
            atlas_entry = _copied_evidence(spec, stage, page, files)
            atlas_entry["regions"] = []
            per_region = []
            for region_index, region in enumerate(spec.regions):
                file_name = f"{page}/material-{region.id}.png"
                stage_file = stage / file_name
                stage_file.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.where(labels == region_index, 255, 0).astype(np.uint8), "L").save(stage_file)
                files.append({"path": file_name, "sha256": _sha256(stage_file)})
                record: dict[str, Any] = {"id": region.id, "mask": file_name,
                                          "roughness": region.roughness}
                if region.roughness_map is not None:
                    record["roughness_map"] = _copied_optional(region.roughness_map, stage,
                        f"{page}/evidence/roughness-{region.id}.png", files)
                atlas_entry["regions"].append(record)
                counts[region_index] = int(np.count_nonzero(labels == region_index))
                per_region.append({"id": region.id, "assigned_texels": counts[region_index],
                                   "roughness_source": "authored-prior"})
            recipe["atlases"][name] = atlas_entry
            unknown = np.where(mapped & (labels < 0), 255, 0).astype(np.uint8)
            unknown_path = stage / page / "unassigned.png"
            Image.fromarray(unknown, "L").save(unknown_path)
            conf_path = stage / page / "confidence.png"
            Image.fromarray(confidence, "L").save(conf_path)
            preview = np.full((*labels.shape, 3), 32, dtype=np.uint8)
            for region_index, color in enumerate(_COLORS[:len(spec.regions)]):
                preview[labels == region_index] = color
            preview[~mapped] = (0, 0, 0)
            preview_path = stage / page / "labels-preview.png"
            Image.fromarray(preview, "RGB").save(preview_path)
            for item in (unknown_path, conf_path, preview_path):
                files.append({"path": str(item.relative_to(stage)), "sha256": _sha256(item)})
            report["atlases"][name].update({
                "regions": per_region,
                "uv_mapped_texels": int(np.count_nonzero(mapped)),
                "assigned_texels": int(np.count_nonzero(labels >= 0)),
                "unassigned_texels": int(np.count_nonzero(unknown)),
                "min_observation_support": options.min_evidence_views if views else 0,
                "observation_support_coverage": (round(float(np.count_nonzero(support_count >= options.min_evidence_views)) /
                                                        max(1, np.count_nonzero(mapped)), 6) if views else None),
                "unassigned_mask": str(unknown_path.relative_to(stage)),
                "confidence_mask": str(conf_path.relative_to(stage)),
                "labels_preview": str(preview_path.relative_to(stage)),
            })
            if not views:
                report["warnings"].append(f"{name}: atlas-only evidence may include baked sunlight/shadows")
            atlas.close()
            for photo, mask in views:
                photo.close()
                mask.close()
        recipe_path = stage / "surface-recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        files.append({"path": recipe_path.name, "sha256": _sha256(recipe_path)})
        # Validate finished masks before publication; an unassigned texel is intentional.
        load_surface_recipe(recipe_path, atlases)
        report_path = stage / "material-separation-report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if target.exists():
            raise ReferenceError("Material output appeared while building; refusing overwrite")
        os.rename(stage, target)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _copied_optional(pin: PinnedSurfaceImage, stage: Path, relative: str,
                     files: list[dict[str, str]]) -> str:
    data = pin.path.read_bytes()
    if pin.path.is_symlink() or hashlib.sha256(data).hexdigest() != pin.sha256:
        raise ReferenceError("Roughness source changed while preparing material proposal")
    dest = stage / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    files.append({"path": relative, "sha256": _sha256(dest), "source_sha256": pin.sha256})
    return relative
