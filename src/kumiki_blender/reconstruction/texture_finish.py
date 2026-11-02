"""Non-destructive exposure/white-balance corrections for private MVS UV atlases.

No geometry, UV coordinates or original images are changed. Missing/black
texels are *not* fabricated. Every new atlas remains an artist-review candidate.
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dense import _json, _sha256
from .references import ReferenceError


@dataclass(frozen=True)
class TextureFinishSettings:
    exposure_ev: float = 0.0
    gamma: float = 1.0
    saturation: float = 1.0
    shadow_lift: float = 0.0
    neutralize_cast: bool = False
    protect_black: int = 12
    balance_atlases: bool = False
    balance_uv_seams: bool = False
    uv_aware_stats: bool = False
    max_balance_ev: float = 0.75
    max_seam_ev: float = 0.75
    min_seam_anchors: int = 8
    auto_exposure_luma: float | None = None
    max_auto_exposure_ev: float = 1.0

    def validate(self) -> "TextureFinishSettings":
        for name, low, high in (("exposure_ev", -3.0, 3.0),
                                ("gamma", 0.4, 2.5),
                                ("saturation", 0.0, 2.5),
                                ("shadow_lift", 0.0, 0.35),
                                ("max_balance_ev", 0.0, 2.0),
                                ("max_seam_ev", 0.0, 2.0),
                                ("max_auto_exposure_ev", 0.0, 2.0)):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
                raise ReferenceError(f"{name} must be finite in [{low}, {high}]")
        target = self.auto_exposure_luma
        if (target is not None and (type(target) not in (float, int)
                or not math.isfinite(target) or not 32 <= target <= 224)):
            raise ReferenceError("auto_exposure_luma must be finite in [32, 224]")
        if (type(self.neutralize_cast) is not bool or type(self.balance_atlases) is not bool
                or type(self.uv_aware_stats) is not bool or type(self.balance_uv_seams) is not bool):
            raise ReferenceError("neutralize_cast, balance_atlases, balance_uv_seams and uv_aware_stats must be booleans")
        if self.balance_atlases and self.balance_uv_seams:
            raise ReferenceError("Choose either median atlas balancing or UV seam balancing, not both")
        if type(self.min_seam_anchors) is not int or not 4 <= self.min_seam_anchors <= 256:
            raise ReferenceError("min_seam_anchors must be an integer between 4 and 256")
        if type(self.protect_black) is not int or not 0 <= self.protect_black <= 32:
            raise ReferenceError("protect_black must be an integer between 0 and 32")
        return self


def _valid_texel_sample(photo: Any, protect_black: int,
                        uv_coverage: Any = None) -> tuple[Any, Any, Any]:
    from PIL import Image, ImageChops

    sample = photo.copy()
    sample.thumbnail((768, 768), Image.Resampling.BILINEAR)
    rgb = sample.convert("RGB")
    gray = rgb.convert("L")
    valid = gray.point(lambda value: 255 if value >= max(16, protect_black) else 0)
    if "A" in sample.getbands() or "transparency" in sample.info:
        # Transparent white padding is just as uncovered as black padding.
        alpha = sample.convert("RGBA").getchannel("A")
        opaque = alpha.point(lambda value: 255 if value >= 128 else 0)
        valid = ImageChops.multiply(valid, opaque)
    if uv_coverage is not None:
        if uv_coverage.size != valid.size or uv_coverage.mode != "L":
            raise ReferenceError("UV coverage does not match the sampled atlas dimensions")
        valid = ImageChops.multiply(valid, uv_coverage)
    return rgb, gray, valid


def _sample_luminance(photo: Any, protect_black: int,
                      uv_coverage: Any = None) -> float | None:
    from PIL import ImageStat

    _rgb, gray, valid = _valid_texel_sample(photo, protect_black, uv_coverage)
    if not valid.getbbox():
        return None
    # Median is less sensitive to highlights or shadow areas than the mean.
    return float(ImageStat.Stat(gray, valid).median[0])


def _atlas_correction(photo: Any, settings: TextureFinishSettings, *,
                      balance_ev: float = 0.0,
                      seam_ev: float = 0.0,
                      auto_ev: float = 0.0,
                      uv_coverage: Any = None) -> tuple[Any, dict[str, Any]]:
    from PIL import Image, ImageEnhance, ImageStat

    original = photo.convert("RGB")
    # Ignore both black padding and transparent padding when estimating color.
    sample, sample_gray, valid = _valid_texel_sample(photo, settings.protect_black,
                                                   uv_coverage)
    if uv_coverage is not None and not valid.getbbox():
        raise ReferenceError("UV-mapped atlas has no valid colored texels to correct")
    stats = ImageStat.Stat(sample, valid)
    gains = [1.0, 1.0, 1.0]
    if settings.neutralize_cast and valid.getbbox():
        means = stats.mean[:3]
        neutral = sum(means) / 3
        gains = [max(0.75, min(1.25, neutral / max(1.0, mean))) for mean in means]
    exposure = 2**(settings.exposure_ev + balance_ev + seam_ev + auto_ev)
    channels = original.split()
    corrected_channels = []
    for channel, gain in zip(channels, gains):
        lut = []
        for value in range(256):
            normalized = value / 255
            lifted = normalized ** (1 / settings.gamma) * exposure * gain
            lifted += settings.shadow_lift * (1-normalized)**2
            lut.append(round(max(0.0, min(1.0, lifted)) * 255))
        corrected_channels.append(channel.point(lut))
    corrected = Image.merge("RGB", tuple(corrected_channels))
    if settings.saturation != 1:
        corrected = ImageEnhance.Color(corrected).enhance(settings.saturation)
    # Soft matte at dark regions prevents correction from turning missing texels
    # into invented painted surfaces. Dark legitimate stone needs manual review.
    threshold = settings.protect_black
    matte = original.convert("L").point(
        lambda value: max(0, min(255, round((value - threshold) * 255 / 20))))
    corrected = Image.composite(corrected, original, matte)
    total_samples = sample_gray.width * sample_gray.height
    count = (uv_coverage.histogram()[255] if uv_coverage is not None
             else total_samples)
    valid_pixels = valid.histogram()[255]
    invalid_pixels = count - valid_pixels
    after_sample = corrected.copy()
    after_sample.thumbnail((768, 768), Image.Resampling.BILINEAR)
    highlight_pixels = sum(after_sample.convert("L").histogram(mask=valid)[250:])
    highlight_fraction = highlight_pixels / max(1, valid_pixels)
    warnings = []
    if invalid_pixels / max(1, count) > 0.1:
        warnings.append("High proportion of dark/transparent uncovered texels; correction cannot recover detail")
    if highlight_fraction > 0.1:
        warnings.append("Bright texture clipping exceeds 10%; lower exposure or shadow lift")
    return corrected, {
        "approx_uncovered_dark_fraction": round(invalid_pixels / max(1, count), 5),
        "uv_mapped_atlas_fraction": (round(count / total_samples, 5)
                                      if uv_coverage is not None else None),
        "approx_highlight_clip_fraction_after": round(highlight_fraction, 5),
        "sampled_mean_rgb": [round(x, 2) for x in stats.mean[:3]],
        "neutralization_gains": [round(x, 5) for x in gains],
        "atlas_balance_ev": round(balance_ev, 5),
        "uv_seam_balance_ev": round(seam_ev, 5),
        "auto_exposure_ev": round(auto_ev, 5),
        "warnings": warnings,
    }


def finish_dense_textures(dense_revision: str | Path, output: str | Path,
                          settings: TextureFinishSettings) -> dict[str, Any]:
    """Copy pinned UV assets into a separately hashed color-corrected revision."""
    settings.validate()
    try:
        from PIL import Image, ImageChops, ImageDraw
    except ImportError as error:
        raise ReferenceError("Texture finishing requires Pillow (install the qa extra)") from error

    # Uses the same OBJ/MTL/atlas trust boundary as the Blender GLB importer.
    from .blender_dense import _textured_obj_plan

    source = Path(dense_revision).resolve()
    target = Path(output).resolve()
    if (Path(dense_revision).is_symlink() or not source.is_dir()
            or Path(output).is_symlink() or target.exists()
            or target == source or target.is_relative_to(source)):
        raise ReferenceError("Texture finishing needs an existing dense revision and unused external output")
    dense_report = _json(source / "dense-report.json")
    if not str(dense_report.get("status", "")).startswith("dense-mesh-candidate"):
        raise ReferenceError("Dense texture correction requires a completed candidate mesh")
    pinned = _textured_obj_plan(source, dense_report, max_mesh_faces=10_000_000)
    original_obj, original_mtl = Path(pinned["source_obj"]), Path(pinned["source_mtl"])
    seam_balance: dict[str, float] = {}
    seam_diagnostics: dict[str, Any] | None = None
    if settings.balance_uv_seams:
        from .uv_seams import balance_uv_seams

        seam_balance, seam_diagnostics = balance_uv_seams(
            original_obj, original_mtl,
            {Path(item["path"]).name: Path(item["path"])
             for item in pinned["texture_atlases"]},
            protect_black=settings.protect_black, max_ev=settings.max_seam_ev,
            min_anchors=settings.min_seam_anchors)
    uv_coverage: dict[str, Any] = {}
    uv_faces: dict[str, int] = {}
    if settings.uv_aware_stats:
        from .uv_coverage import rasterize_uv_coverage

        sizes = {}
        for entry in pinned["texture_atlases"]:
            path = Path(entry["path"])
            with Image.open(path) as atlas:
                thumbnail = atlas.copy()
                thumbnail.thumbnail((768, 768), Image.Resampling.BILINEAR)
                sizes[path.name] = thumbnail.size
        uv_coverage, uv_faces = rasterize_uv_coverage(
            Path(pinned["source_obj"]), Path(pinned["source_mtl"]), sizes)
    # Compute atlas-to-atlas brightness ratios from valid (non-black) texels.
    # A bounded gain per page can reduce *between-page* seams, but it cannot
    # repair UV chart edges or different lighting within a single atlas.
    balance: dict[str, float] = {}
    brightness: dict[str, float | None] = {}
    auto_ev = 0.0
    if settings.balance_atlases or settings.auto_exposure_luma is not None:
        for entry in pinned["texture_atlases"]:
            atlas_path = Path(entry["path"])
            with Image.open(atlas_path) as image:
                brightness[atlas_path.name] = _sample_luminance(
                    image, settings.protect_black, uv_coverage.get(atlas_path.name))
        if settings.uv_aware_stats and any(value is None for value in brightness.values()):
            raise ReferenceError("UV atlas lacks valid colored texels for exposure balancing")
        usable = [value for value in brightness.values() if value is not None]
        if usable:
            target_luma = statistics.median(usable)
            if settings.auto_exposure_luma is not None:
                desired_ev = math.log2(settings.auto_exposure_luma / max(1, target_luma))
                auto_ev = max(-settings.max_auto_exposure_ev,
                              min(settings.max_auto_exposure_ev, desired_ev))
            if settings.balance_atlases and len(usable) > 1:
                for filename, value in brightness.items():
                    if value is not None:
                        ev = math.log2(target_luma / max(1, value))
                        balance[filename] = max(-settings.max_balance_ev,
                                                min(settings.max_balance_ev, ev))
    target.mkdir(parents=True, exist_ok=False)
    (target / "review").mkdir()
    shutil.copy2(original_obj, target / original_obj.name)
    shutil.copy2(original_mtl, target / original_mtl.name)
    report: dict[str, Any] = {
        "schema_version": 1, "source_dense_revision": str(source),
        "source_dense_report_sha256": _sha256(source / "dense-report.json"),
        "source_obj_sha256": pinned["source_obj_sha256"],
        "source_mtl_sha256": _sha256(original_mtl),
        "source_texture_atlases": [
            {"name": Path(item["path"]).name, "sha256": item["sha256"]}
            for item in pinned["texture_atlases"]],
        "settings": {key: getattr(settings, key) for key in settings.__dataclass_fields__},
        "textured_obj": {"path": original_obj.name,
                         "sha256": _sha256(target / original_obj.name)},
        "mtl": {"path": original_mtl.name,
                "sha256": _sha256(target / original_mtl.name)},
        "texture_atlases": [],
        "status": "incomplete / texture-review-required",
        "distribution_allowed": False,
    }
    if seam_diagnostics is not None:
        report["uv_seam_balance"] = seam_diagnostics

    def checkpoint() -> None:
        (target / "texture-finish-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")

    checkpoint()
    for item in pinned["texture_atlases"]:
        atlas = Path(item["path"])
        try:
            with Image.open(atlas) as photo:
                photo.load()
                if photo.format not in ("JPEG", "PNG"):
                    raise ReferenceError("Texture correction expects a JPEG or PNG atlas")
                corrected, quality = _atlas_correction(
                    photo, settings, balance_ev=balance.get(atlas.name, 0.0),
                    seam_ev=seam_balance.get(atlas.name, 0.0),
                    auto_ev=auto_ev, uv_coverage=uv_coverage.get(atlas.name))
                output_path = target / atlas.name
                if photo.format == "JPEG":
                    corrected.save(output_path, "JPEG", quality=95, subsampling=0)
                else:
                    if "A" in photo.getbands() or "transparency" in photo.info:
                        corrected.putalpha(photo.convert("RGBA").getchannel("A"))
                    corrected.save(output_path, "PNG", optimize=True)
                # A small side-by-side makes gross exposure and cast regressions
                # visible without opening a huge 8K atlas in an image editor.
                before = photo.convert("RGB")
                after = corrected.convert("RGB")
                before.thumbnail((600, 600))
                after.thumbnail((600, 600))
                canvas = Image.new("RGB", (1200, 630), (24, 27, 31))
                canvas.paste(before, (0, 24))
                canvas.paste(after, (600, 24))
                labels = ImageDraw.Draw(canvas)
                labels.text((8, 5), "Original pinned atlas", fill=(235, 235, 235))
                labels.text((608, 5), "Corrected review candidate", fill=(235, 235, 235))
                preview_path = target / "review" / (atlas.name + ".jpg")
                canvas.save(preview_path, "JPEG", quality=84)
                uv_dark_review = None
                if settings.uv_aware_stats:
                    sample, _gray, valid = _valid_texel_sample(
                        photo, settings.protect_black, uv_coverage[atlas.name])
                    dark_on_mesh = ImageChops.subtract(uv_coverage[atlas.name], valid)
                    tint = dark_on_mesh.point(lambda value: round(value * 0.65))
                    shaded = Image.composite(Image.new("RGB", sample.size, (252, 59, 43)),
                                             sample, tint)
                    overview = Image.new("RGB", (sample.width, sample.height+26),
                                         (24, 27, 31))
                    overview.paste(shaded, (0, 26))
                    ImageDraw.Draw(overview).text(
                        (7, 5), "Red: dark/transparent texels inside actual mesh UV faces",
                        fill=(236, 238, 239))
                    uv_path = target / "review" / (atlas.name + ".uv-dark.jpg")
                    overview.save(uv_path, "JPEG", quality=90)
                    uv_dark_review = str(uv_path.relative_to(target))
            report["texture_atlases"].append({
                "path": atlas.name, "sha256": _sha256(output_path),
                "dimensions": list(corrected.size), "source_sha256": item["sha256"],
                "comparison_preview": str(preview_path.relative_to(target)),
                "source_median_luma": brightness.get(atlas.name),
                "uv_mapped_faces": uv_faces.get(atlas.name),
                "uv_dark_review": uv_dark_review,
                **quality,
            })
        except Exception as error:
            report.update(error=str(error), failed_atlas=atlas.name)
            checkpoint()
            raise
        checkpoint()
    report["status"] = "candidate / corrected-atlas-artistic-review-required"
    checkpoint()
    return report
