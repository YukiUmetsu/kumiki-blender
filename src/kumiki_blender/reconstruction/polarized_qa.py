"""Private UV-aligned cross/parallel-polarized photographic specular QA.

Polarization separation reveals a view/light-dependent highlight contrast,
NOT a measured roughness value. In particular, it cannot invert a BRDF
without known geometry, light distribution, camera response and calibration.
"""

from __future__ import annotations

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
from .illumination_guides import _linear_luminance
from .references import ReferenceError
from .surface_finish import _pinned_image, _unique_pairs, load_surface_recipe
from .uv_coverage import rasterize_uv_coverage


@dataclass(frozen=True)
class PolarizedQAOptions:
    max_side: int = 8192
    min_views: int = 3
    max_mesh_faces: int = 5_000_000

    def validate(self) -> "PolarizedQAOptions":
        if type(self.max_side) is not int or not 64 <= self.max_side <= 8192:
            raise ReferenceError("Polarized max_side must be in [64, 8192]")
        if type(self.min_views) is not int or not 3 <= self.min_views <= 8:
            raise ReferenceError("Polarized min_views must be in [3, 8]")
        if type(self.max_mesh_faces) is not int or not 1 <= self.max_mesh_faces <= 5_000_000:
            raise ReferenceError("Polarized max_mesh_faces must be in [1, 5000000]")
        return self


def _captures(path: Path, sizes: dict[str, tuple[int, int]],
              min_views: int) -> dict[str, list[dict[str, Any]]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ReferenceError("Polarized capture manifest must be an unlinked JSON file <= 64 KiB")
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                          parse_constant=lambda n: (_ for _ in ()).throw(ValueError(n)))
    except (ValueError, UnicodeError) as exc:
        raise ReferenceError(f"Invalid polarized manifest: {exc}") from exc
    if (not isinstance(data, dict) or set(data) != {"schema_version", "atlases"}
            or type(data["schema_version"]) is not int or data["schema_version"] != 1
            or not isinstance(data["atlases"], dict) or set(data["atlases"]) != set(sizes)):
        raise ReferenceError("Polarized manifest needs exactly the dense atlas names, schema_version 1")
    result: dict[str, list[dict[str, Any]]] = {}
    total = 0
    for name, raw in data["atlases"].items():
        if not isinstance(raw, list) or not min_views <= len(raw) <= 8:
            raise ReferenceError("Each atlas needs 3–8 independent polarization pairs")
        ids: set[str] = set()
        unique_paths: set[Path] = set()
        records: list[dict[str, Any]] = []
        for pair in raw:
            if not isinstance(pair, dict) or set(pair) != {
                    "id", "parallel", "cross", "confidence", "same_exposure",
                    "unchanged_lighting", "analyzer_rotation_degrees"}:
                raise ReferenceError("Capture pair needs id, 3 images and declared controlled conditions")
            identifier = pair["id"]
            if (not isinstance(identifier, str) or not identifier
                    or len(identifier) > 64 or not identifier.isascii()
                    or not all(c.isalnum() or c in "-_" for c in identifier)
                    or identifier in ids):
                raise ReferenceError("Polarized capture IDs must be distinct safe ASCII labels")
            ids.add(identifier)
            if (pair["same_exposure"] is not True
                    or pair["unchanged_lighting"] is not True
                    or type(pair["analyzer_rotation_degrees"]) not in (float, int)
                    or not math.isfinite(pair["analyzer_rotation_degrees"])
                    or not 85 <= pair["analyzer_rotation_degrees"] <= 95):
                raise ReferenceError("Polarized captures require equal exposure, fixed light and 90-degree analyzer rotation")
            record: dict[str, Any] = {"id": identifier}
            for key in ("parallel", "cross", "confidence"):
                pin = _pinned_image(path.resolve().parent, pair[key], sizes[name],
                                    grayscale=(key == "confidence"))
                if pin.path in unique_paths:
                    raise ReferenceError("Polarized captures must use distinct independent evidence files")
                unique_paths.add(pin.path)
                record[key] = pin
            records.append(record)
            total += sizes[name][0] * sizes[name][1]
        result[name] = records
    if total > 256_000_000:
        raise ReferenceError("Polarization pair evidence exceeds 256 million atlas-view texels")
    return result


def inspect_polarized_uv(dense_revision: str | Path, surface_recipe: str | Path,
                         capture_manifest: str | Path, output: str | Path,
                         *, options: PolarizedQAOptions = PolarizedQAOptions()) -> dict[str, Any]:
    """Produce a conservative specular-contrast preview, never a roughness map."""
    from PIL import Image, ImageOps
    import numpy as np

    options.validate()
    dense_input, recipe_input, capture_input, target_input = map(
        Path, (dense_revision, surface_recipe, capture_manifest, output))
    source, target = dense_input.resolve(), target_input.resolve()
    if (dense_input.is_symlink() or not source.is_dir() or recipe_input.is_symlink()
            or capture_input.is_symlink() or target_input.is_symlink() or target.exists()
            or not target.parent.is_dir() or target.is_relative_to(source)
            or source.is_relative_to(target) or target == recipe_input.resolve()
            or target == capture_input.resolve()):
        raise ReferenceError("Polarized QA needs a dense revision, controlled captures and unused output")
    dense_report = _json(source / "dense-report.json")
    if not str(dense_report.get("status", "")).startswith("dense-mesh-candidate"):
        raise ReferenceError("Polarized QA requires a completed dense-mesh candidate")
    pinned = _textured_obj_plan(source, dense_report, options.max_mesh_faces)
    atlases = {Path(item["path"]).name: Path(item["path"])
               for item in pinned["texture_atlases"]}
    if len(atlases) != len(pinned["texture_atlases"]):
        raise ReferenceError("Polarized atlas filenames must be unique")
    sizes = {}
    for name, atlas in atlases.items():
        with Image.open(atlas) as image:
            if max(image.size) > options.max_side:
                raise ReferenceError("Polarized QA atlas exceeds max_side")
            sizes[name] = image.size
    source_hashes = {name: _sha256(path) for name, path in atlases.items()}
    manifest_hash = _sha256(capture_input)
    recipe_hash = _sha256(recipe_input)
    report_hash = _sha256(source / "dense-report.json")
    specs = load_surface_recipe(recipe_input, atlases)
    captures = _captures(capture_input, sizes, options.min_views)
    uv_masks, _ = rasterize_uv_coverage(Path(pinned["source_obj"]),
                                        Path(pinned["source_mtl"]), sizes)
    if (_sha256(capture_input) != manifest_hash or _sha256(recipe_input) != recipe_hash
            or _sha256(source / "dense-report.json") != report_hash):
        raise ReferenceError("Polarized capture recipe changed during preflight")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-polarized-", dir=target.parent))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "polarized-specular-contrast-candidate / human-review-required",
        "distribution_allowed": False,
        "roughness_measured": False,
        "source_dense_revision": str(source),
        "source_surface_recipe": str(recipe_input.resolve()),
        "source_capture_manifest": str(capture_input.resolve()),
        "source_dense_report_sha256": report_hash,
        "source_surface_recipe_sha256": recipe_hash,
        "source_capture_manifest_sha256": manifest_hash,
        "source_atlas_sha256": source_hashes,
        "options": {name: getattr(options, name) for name in options.__dataclass_fields__},
        "atlases": {}, "files": [],
        "limitations": ("Analyzer rotation/exposure/light stability are operator declarations, not sensor verified. "
                        "Only UV-registered, linearly responding, unsaturated controlled captures are useful. "
                        "Specular contrast is view/light and Fresnel dependent, NOT a physical roughness "
                        "measurement. Never plug this guide directly into a PBR roughness socket."),
    }
    try:
        for index, (name, path) in enumerate(atlases.items(), 1):
            if (_sha256(path) != source_hashes[name] or _sha256(recipe_input) != recipe_hash
                    or _sha256(capture_input) != manifest_hash
                    or _sha256(source / "dense-report.json") != report_hash):
                raise ReferenceError("Polarized source evidence changed during inspection")
            width, height = sizes[name]
            active = np.asarray(uv_masks[name]) >= 128
            # Material mask is a guard: atlas padding and unlabelled UV regions
            # must never masquerade as measured glossy geometry.
            material = np.zeros((height, width), dtype=bool)
            for region in specs[name].regions:
                material |= np.asarray(region.mask.open_verified()) >= 128
            active &= material
            contrast = np.zeros((height, width), dtype=np.uint8)
            coverage = np.zeros((height, width), dtype=np.uint8)
            ignored_negative = 0
            ignored_saturated = 0
            # Memory-map per-pair candidates rather than retaining every 8K
            # photo and its float32 ratios in RAM at the same time.
            scratch = stage / f".pairs-{index:02d}.mmap"
            values = np.memmap(scratch, mode="w+", dtype=np.float16,
                               shape=(len(captures[name]), height, width))
            for pair_index, pair in enumerate(captures[name]):
                parallel = pair["parallel"].open_verified()
                cross = pair["cross"].open_verified()
                conf = pair["confidence"].open_verified()
                parallel_rgb = parallel.convert("RGB")
                cross_rgb = cross.convert("RGB")
                p_alpha = parallel.getchannel("A") if "A" in parallel.getbands() else None
                c_alpha = cross.getchannel("A") if "A" in cross.getbands() else None
                for y in range(0, height, 96):
                    bottom = min(height, y + 96)
                    box = (0, y, width, bottom)
                    p_rgb = np.asarray(parallel_rgb.crop(box))
                    c_rgb = np.asarray(cross_rgb.crop(box))
                    p = _linear_luminance(p_rgb, np)
                    c = _linear_luminance(c_rgb, np)
                    sat = (p_rgb.max(axis=2) >= 250) | (c_rgb.max(axis=2) >= 250)
                    valid = (np.asarray(conf.crop(box)) >= 128) & ~sat & (c >= .025)
                    if p_alpha is not None:
                        valid &= np.asarray(p_alpha.crop(box)) >= 128
                    if c_alpha is not None:
                        valid &= np.asarray(c_alpha.crop(box)) >= 128
                    ignored_negative += int(np.count_nonzero(valid & (p + .01 < c)))
                    ignored_saturated += int(np.count_nonzero(sat & active[y:bottom]))
                    valid &= p + .01 >= c
                    ratio = np.clip((p-c)/(p+.025), 0, 1)
                    values[pair_index, y:bottom] = np.where(valid, ratio, np.nan)
                parallel.close(); cross.close(); conf.close()
                parallel_rgb.close(); cross_rgb.close()
                if p_alpha is not None:
                    p_alpha.close()
                if c_alpha is not None:
                    c_alpha.close()
            values.flush()
            for y in range(0, height, 96):
                bottom = min(height, y + 96)
                stripe = np.asarray(values[:, y:bottom], dtype=np.float32)
                n = np.count_nonzero(np.isfinite(stripe), axis=0)
                eligible = active[y:bottom] & (n >= options.min_views)
                if np.any(eligible):
                    # nanmedian operates only on pixels supported by every
                    # required independent pair; do not propagate NaNs.
                    value = np.nanmedian(stripe[:, eligible], axis=0)
                    contrast[y:bottom][eligible] = np.uint8(np.clip(value * 255 + .5, 0, 255))
                    coverage[y:bottom][eligible] = 255
            del values
            scratch.unlink()
            folder = stage / f"atlas-{index:02d}"
            folder.mkdir()
            field_path = folder / "specular-contrast.png"
            valid_path = folder / "supported-uv.png"
            Image.fromarray(contrast, "L").save(field_path)
            Image.fromarray(coverage, "L").save(valid_path)
            with Image.open(path) as original:
                rgb = original.convert("RGB")
                rgb.thumbnail((768, 768))
            page = Image.new("RGB", (rgb.width*3, rgb.height), "#17171b")
            page.paste(rgb, (0, 0))
            page.paste(ImageOps.colorize(Image.fromarray(contrast, "L").resize(
                rgb.size, Image.Resampling.NEAREST), "#101923", "#fcbb69"), (rgb.width, 0))
            page.paste(Image.fromarray(coverage, "L").resize(rgb.size), (rgb.width*2, 0))
            preview_path = folder / "source-contrast-coverage.png"
            page.save(preview_path)
            for item in (field_path, valid_path, preview_path):
                report["files"].append({"path": str(item.relative_to(stage)), "sha256": _sha256(item)})
            samples = contrast[coverage >= 128]
            report["atlases"][name] = {
                "capture_pair_count": len(captures[name]),
                "capture_pairs": [
                    {"id": entry["id"], **{key+"_sha256": entry[key].sha256
                                               for key in ("parallel", "cross", "confidence")}}
                    for entry in captures[name]],
                "supported_fraction_of_mapped_material_uv": round(float(np.count_nonzero(coverage)) /
                                                                    max(1, np.count_nonzero(active)), 6),
                "median_relative_specular_contrast": (round(float(np.median(samples))/255, 5)
                                                       if samples.size else None),
                "p95_relative_specular_contrast": (round(float(np.percentile(samples, 95))/255, 5)
                                                    if samples.size else None),
                "specular_fraction_above_10_percent": (round(float(np.count_nonzero(samples >= 26)) /
                                                            max(1, samples.size), 6)),
                "rejected_negative_pair_texels": ignored_negative,
                "rejected_saturated_pair_texels": ignored_saturated,
                "status": ("unmeasured-roughness / photographic-contrast-review-only"
                           if samples.size else "insufficient-controlled-pair-support / no-evidence"),
            }
            for pair in captures[name]:
                for key in ("parallel", "cross", "confidence"):
                    if _sha256(pair[key].path) != pair[key].sha256:
                        raise ReferenceError("Polarized input changed while rendering its candidate")
        report_path = stage / "polarized-qa-report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if target.exists():
            raise ReferenceError("Polarized QA output appeared during processing")
        os.rename(stage, target)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
