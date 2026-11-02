"""Non-destructive exposure/white-balance corrections for private MVS UV atlases.

No geometry, UV coordinates or original images are changed. Missing/black
texels are *not* fabricated. Every new atlas remains an artist-review candidate.
"""

from __future__ import annotations

import json
import math
import shutil
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

    def validate(self) -> "TextureFinishSettings":
        for name, low, high in (("exposure_ev", -3.0, 3.0),
                                ("gamma", 0.4, 2.5),
                                ("saturation", 0.0, 2.5),
                                ("shadow_lift", 0.0, 0.35)):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
                raise ReferenceError(f"{name} must be finite in [{low}, {high}]")
        if type(self.neutralize_cast) is not bool:
            raise ReferenceError("neutralize_cast must be a boolean")
        if type(self.protect_black) is not int or not 0 <= self.protect_black <= 32:
            raise ReferenceError("protect_black must be an integer between 0 and 32")
        return self


def _atlas_correction(photo: Any, settings: TextureFinishSettings) -> tuple[Any, dict[str, Any]]:
    from PIL import Image, ImageEnhance, ImageStat

    original = photo.convert("RGB")
    sample = original.copy()
    sample.thumbnail((768, 768), Image.Resampling.BILINEAR)
    sample_gray = sample.convert("L")
    # OpenMVS may leave black/uncovered texels. Do not estimate color from them.
    valid = sample_gray.point(lambda value: 255 if value >= max(settings.protect_black, 16) else 0)
    stats = ImageStat.Stat(sample, valid)
    gains = [1.0, 1.0, 1.0]
    if settings.neutralize_cast and valid.getbbox():
        means = stats.mean[:3]
        neutral = sum(means) / 3
        gains = [max(0.75, min(1.25, neutral / max(1.0, mean))) for mean in means]
    exposure = 2**settings.exposure_ev
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
    dark_pixels = sum(sample_gray.histogram()[:max(threshold, 16)])
    count = sample_gray.width * sample_gray.height
    return corrected, {
        "approx_uncovered_dark_fraction": round(dark_pixels / count, 5),
        "sampled_mean_rgb": [round(x, 2) for x in stats.mean[:3]],
        "neutralization_gains": [round(x, 5) for x in gains],
        "warnings": (["High proportion of dark/possibly uncovered texels; correction cannot recover detail"]
                     if dark_pixels / count > 0.1 else []),
    }


def finish_dense_textures(dense_revision: str | Path, output: str | Path,
                          settings: TextureFinishSettings) -> dict[str, Any]:
    """Copy pinned UV assets into a separately hashed color-corrected revision."""
    settings.validate()
    try:
        from PIL import Image, ImageDraw
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
                corrected, quality = _atlas_correction(photo, settings)
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
            report["texture_atlases"].append({
                "path": atlas.name, "sha256": _sha256(output_path),
                "dimensions": list(corrected.size), "source_sha256": item["sha256"],
                "comparison_preview": str(preview_path.relative_to(target)),
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
