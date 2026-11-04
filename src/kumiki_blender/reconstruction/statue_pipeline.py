"""Resumable, fail-closed photo-to-GLB workflow for statues.

This is an orchestration layer over existing audited commands. Candidate
segmentation is never represented as verified ground-truth geometry.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..glb_audit import AuditError, audit_glb
from .references import ReferenceError, audit_reference_set
from .splat_input import load_photo_folder


@dataclass(frozen=True)
class StatuePipelineSettings:
    front_photo: str | None = None
    blender: str = "blender"
    openmvs_dir: Path | None = None
    sam2_checkpoint: Path | None = None
    sam2_size: str = "small"
    sam2_device: str = "mps"
    matching_device: str = "mps"
    mask_guides: Path | None = None
    masks_dir: Path | None = None
    mask_cache_dir: Path | None = None
    photo_cache_dir: Path | None = None
    auto_download_model: bool = True
    ai_bridges: bool = False
    auto_recover_cameras: bool = True
    ai_bridge_workflow: Path | None = None
    min_registered_fraction: float = 0.80
    max_mixed_mask_fraction: float = 0.12
    max_colmap_rss_mb: int = 12288
    timeout_seconds: int = 14400

    def validate(self) -> None:
        if self.front_photo is not None and (not self.front_photo or
                 Path(self.front_photo).name != self.front_photo):
            raise ReferenceError("--front-photo must be an original filename, not a path")
        if not 0.25 <= self.min_registered_fraction <= 1:
            raise ReferenceError("min_registered_fraction must be between 0.25 and 1")
        if not 0 <= self.max_mixed_mask_fraction <= 1:
            raise ReferenceError("max_mixed_mask_fraction must be between 0 and 1")
        if self.sam2_size not in ("tiny", "small", "base_plus", "large"):
            raise ReferenceError("Unsupported SAM2 model size")
        if self.sam2_device not in ("mps", "cpu", "auto"):
            raise ReferenceError("Unsupported SAM2 device")
        if self.matching_device not in ("mps", "torch-cpu"):
            raise ReferenceError("Statue LightGlue matching device must be mps or torch-cpu")
        if not 1024 <= self.max_colmap_rss_mb <= 131072:
            raise ReferenceError("max_colmap_rss_mb must be between 1024 and 131072")
        if not 60 <= self.timeout_seconds <= 86400:
            raise ReferenceError("timeout_seconds must be between 60 and 86400")
        for name, path in (("mask guides", self.mask_guides),
                           ("SAM2 checkpoint", self.sam2_checkpoint)):
            if path is not None and (path.is_symlink() or not path.is_file()):
                raise ReferenceError(f"{name} must be an existing regular file: {path}")
        if self.masks_dir is not None and (self.masks_dir.is_symlink() or not self.masks_dir.is_dir()):
            raise ReferenceError("--masks-dir must be a real directory of reviewed rectified PNG masks")
        if (self.mask_cache_dir is not None and
                (self.mask_cache_dir.is_symlink() or
                 (self.mask_cache_dir.exists() and not self.mask_cache_dir.is_dir()))):
            raise ReferenceError("--mask-cache-dir must not be a symlink or file")
        if (self.photo_cache_dir is not None and
                (self.photo_cache_dir.is_symlink() or
                 (self.photo_cache_dir.exists() and not self.photo_cache_dir.is_dir()))):
            raise ReferenceError("--photo-cache-dir must not be a symlink or file")
        if self.ai_bridge_workflow is not None and (
                self.ai_bridge_workflow.is_symlink() or not self.ai_bridge_workflow.is_file()):
            raise ReferenceError("AI bridge workflow must be an existing regular file")


_SAM2_FILES = {size: f"sam2.1_hiera_{size}.pt"
               for size in ("tiny", "small", "base_plus", "large")}
_SAM2_OFFICIAL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"


def _resolve_sam2_checkpoint(explicit: Path | None, size: str, *,
                             allow_download: bool = True,
                             cache_root: Path | None = None,
                             download_progress: Callable[[int, int | None], None] | None = None) -> tuple[Path, str]:
    """Find existing official weights, or download once from Meta's fixed URL."""
    filename = _SAM2_FILES[size]
    if explicit is not None:
        return explicit.resolve(), "explicit"
    candidates = [Path(value) for key in ("KUMIKI_SAM2_CHECKPOINT", "SAM2_CHECKPOINT")
                  if (value := os.environ.get(key))]
    cache = cache_root or (Path.home() / ".cache/kumiki-blender/models")
    candidates += [cache / filename,
                   Path.home() / ".cache/torch/hub/checkpoints" / filename,
                   Path.home() / "Downloads" / filename,
                   Path.cwd() / "checkpoints" / filename]
    for path in candidates:
        if path.is_symlink():
            continue
        if path.is_file() and 0 < path.stat().st_size < 2_000_000_000:
            return path.resolve(), "discovered"
    if not allow_download:
        raise ReferenceError(f"SAM2 {filename} was not found locally; allow automatic model download or pass --sam2-checkpoint")
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        raise ReferenceError("Model cache must be a real directory")
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / filename
    if destination.is_symlink():
        raise ReferenceError("Refusing symbolic link as SAM2 download target")
    temporary: Path | None = None
    try:
        request = urllib.request.Request(_SAM2_OFFICIAL + filename,
                                         headers={"User-Agent": "kumiki-blender/1"})
        with urllib.request.urlopen(request, timeout=120) as source:
            if source.url.split(":", 1)[0] != "https":
                raise ReferenceError("Model download redirected to non-HTTPS URL")
            content_length = source.headers.get("Content-Length")
            expected = (int(content_length) if content_length and content_length.isdigit()
                        else None)
            with tempfile.NamedTemporaryFile(dir=cache, prefix=".sam2-", delete=False) as stream:
                temporary = Path(stream.name)
                count = 0
                while block := source.read(1024 * 1024):
                    count += len(block)
                    if count > 1_000_000_000:
                        raise ReferenceError("Model download exceeds 1 GB safety limit")
                    stream.write(block)
                    if download_progress is not None:
                        download_progress(count, expected)
        if count < 1_000_000:
            raise ReferenceError("Model download was unexpectedly small; refusing checkpoint")
        os.replace(temporary, destination)
        temporary = None
        return destination, "official-meta-download"
    except (OSError, ValueError) as error:
        raise ReferenceError(f"Automatic SAM2 download failed: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _infer_front_photo(sparse: Path, camera: dict[str, Any]) -> dict[str, Any]:
    """Suggest a front anchor from existing photo labels; never claim visual proof."""
    index = _read_report(sparse / "staged-reference-index.json")
    registered = set(camera.get("registered_filenames") or [])
    ranked = []
    for item in index.get("images", []):
        if not isinstance(item, dict) or item.get("filename") not in registered:
            continue
        name = str(item.get("original_filename") or item["filename"])
        words = set(re.findall(r"[a-z]+", name.lower()))
        score = (12 * bool({"front", "frontal"} & words)
                 + 3 * bool("face" in words)
                 - 9 * bool({"left", "right", "side", "profile"} & words)
                 - 20 * bool({"rear", "back", "behind", "top", "aerial"} & words))
        if score > 0:
            ranked.append((score, name))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    if ranked and ranked[0][0] >= 10:
        return {"filename": ranked[0][1], "source": "filename-hint / unverified",
                "confidence": "tentative", "other_candidates": [name for _, name in ranked[1:4]]}
    return {"filename": None, "source": "unknown / arbitrary-reference-camera",
            "confidence": "none", "other_candidates": [name for _, name in ranked[:4]]}


def _read_report(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReferenceError(f"Expected completed stage report is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError, OSError) as error:
        raise ReferenceError(f"Invalid stage report: {path}") from error
    if not isinstance(value, dict):
        raise ReferenceError(f"Stage report must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_mask_revision(mask_source: Path, sparse: Path) -> dict[str, Any]:
    """Require exact source binding and content hashes for supplied masks."""
    manifest = _read_report(mask_source.parent / "mask-report.json")
    if manifest.get("source_revision") != str(sparse.resolve()):
        raise ReferenceError("Reviewed masks belong to another camera revision; regenerate masks for these new photos")
    expected = {item.get("filename"): item.get("mask_sha256")
                for item in manifest.get("images", []) if isinstance(item, dict)}
    if not expected:
        raise ReferenceError("Reviewed mask revision has no per-image hash manifest")
    actual = set()
    for name, digest in expected.items():
        if (not isinstance(name, str) or Path(name).name != name or "\\" in name
                or not isinstance(digest, str) or len(digest) != 64):
            raise ReferenceError("Mask revision contains invalid filename/hash metadata")
        path = mask_source / (name + ".mask.png")
        if path.is_symlink() or not path.is_file():
            raise ReferenceError(f"Mask in reviewed revision is missing: {name}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ReferenceError(f"Mask was modified without creating a new review revision: {name}")
        actual.add(path.name)
    if actual != {file.name for file in mask_source.glob("*.mask.png")}:
        raise ReferenceError("Reviewed mask revision contains untracked mask files")
    return manifest


def _coverage_warnings(coverage: dict[str, Any], registered_fraction: float,
                       threshold: float) -> list[str]:
    warnings = []
    if registered_fraction < threshold:
        warnings.append(f"Only {registered_fraction:.1%} registered; require at least {threshold:.1%}")
    reference = coverage.get("angle_reference", {})
    if reference.get("kind") != "annotated-front":
        # Camera registration can be audited without pretending that the
        # arbitrary coordinate frame points at the object's anatomical front.
        return warnings
    images = [item for item in coverage.get("images", [])
              if item.get("registered_in_best_model") and isinstance(item.get("estimated_view"), dict)]
    azimuths = [item["estimated_view"].get("relative_azimuth_deg") for item in images]
    elevations = [item["estimated_view"].get("relative_elevation_deg") for item in images]
    left = sum(type(angle) in (int, float) and angle <= -65 for angle in azimuths)
    right = sum(type(angle) in (int, float) and angle >= 65 for angle in azimuths)
    elevated = sum(type(angle) in (int, float) and angle >= 12 for angle in elevations)
    if left < 2 or right < 2:
        warnings.append(f"Weak side coverage: {left} front-image-left and {right} front-image-right profile views (need >=2 each)")
    if elevated < 2:
        warnings.append(f"Weak elevated coverage: only {elevated} cameras above 12 degrees (need >=2)")
    return warnings


def _mask_warnings(mask_report: dict[str, Any], audit: dict[str, Any],
                   limit: float) -> list[str]:
    warnings = []
    images = mask_report.get("images", [])
    if not images:
        warnings.append("No foreground masks were generated")
    for item in images:
        name = str(item.get("filename", "unknown"))
        fraction = item.get("foreground_fraction")
        if type(fraction) not in (float, int) or not 0.05 <= fraction <= 0.85:
            warnings.append(f"{name}: foreground extent looks implausible")
        if item.get("missing_parts") or item.get("background_seed_violations"):
            warnings.append(f"{name}: anatomy missing or background prompts violated")
        if any("Foreground touches photo border" in warning
               for warning in item.get("warnings", [])):
            warnings.append(f"{name}: segmentation needs guided anatomy/boundary review")
    mixed = audit.get("mixed_label_fraction")
    if type(mixed) not in (float, int) or mixed > limit:
        warnings.append(f"Cross-view mask disagreement {mixed!r} exceeds {limit:.0%}")
    # Majority-vote tracks can all agree on the same non-statue background.
    # No numerical check can prove that the full silhouette is accurate.
    return warnings


def _possible_blue_sky_leaks(sparse: Path, masks: Path,
                             image_names: list[str]) -> list[str]:
    """Flag clear blue pixels included by an outdoor statue mask.

    This is only a conservative rejection heuristic; zero findings never
    constitute approval of the subject silhouette or the stone background.
    """
    from PIL import Image
    import numpy as np

    warnings = []
    for name in image_names:
        with Image.open(sparse / "undistorted/images" / name) as source, Image.open(masks / (name + ".mask.png")) as binary:
            small = source.convert("RGB")
            small.thumbnail((256, 256))
            mask = binary.convert("L").resize(small.size, Image.Resampling.NEAREST)
            rgb = np.asarray(small, dtype=np.float32)
            active = np.asarray(mask) > 127
            if not active.any():
                warnings.append(f"{name}: mask is entirely background")
                continue
            rows = np.arange(rgb.shape[0])[:, None]
            blue_sky = ((rgb[:, :, 2] > 120) &
                        (rgb[:, :, 2] > 1.15 * rgb[:, :, 0]) &
                        (rgb[:, :, 2] > 1.06 * rgb[:, :, 1]) &
                        (rows < 0.65 * rgb.shape[0]))
            fraction = float((active & blue_sky).sum()) / float(active.sum())
            if fraction > 0.025:
                warnings.append(f"{name}: possible blue-sky contamination ({fraction:.1%} of foreground)")
    return warnings


RESUME_INPUTS_CHANGED_ERROR = "Resume refused: source photos or quality settings changed; create a new output"


def build_statue_glb(
    photos: Path, asset_id: str, output: Path, settings: StatuePipelineSettings,
    *, resume: bool = False,
    runner: Callable[[list[str], Path, dict[str, str]], int] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build once, or resume from validated stage receipts without redoing SfM."""
    def emit(kind: str, phase: str, detail: str = "") -> None:
        if progress_callback is not None:
            progress_callback({"kind": kind, "phase": phase, "detail": detail})

    emit("start", "preflight", "Checking local tools, photos and model weights")
    settings.validate()
    if settings.masks_dir is None and any(importlib.util.find_spec(name) is None
                                          for name in ("torch", "sam2")):
        raise ReferenceError("SAM2 is not installed; install official facebookresearch/sam2 and PyTorch first")
    checkpoint_file, checkpoint_source = (None, "reviewed-masks") if (
        settings.masks_dir is not None and settings.sam2_checkpoint is None) else (
        _resolve_sam2_checkpoint(settings.sam2_checkpoint, settings.sam2_size,
                                 allow_download=settings.auto_download_model and settings.masks_dir is None,
                                 download_progress=lambda count, total: emit(
                                     "tick", "preflight", "Downloading official SAM2 weights: "
                                     + (f"{count * 100 // total}% ({count // 1048576} MiB)"
                                        if total else f"{count // 1048576} MiB"))))
    refs = load_photo_folder(photos, asset_id)
    audit_reference_set(refs)
    target = output.resolve()
    if output.is_symlink() or target.is_relative_to(photos.resolve()):
        raise ReferenceError("Output must be a real directory outside source photographs")
    report_path = target / "statue-pipeline-report.json"
    source_fingerprint = refs.manifest_sha256
    settings_payload = {
        "front_photo": settings.front_photo,
        "sam2_checkpoint": str(checkpoint_file) if checkpoint_file else None,
        "sam2_checkpoint_sha256": (_sha256_file(checkpoint_file) if checkpoint_file else None),
        "sam2_size": settings.sam2_size,
        "sam2_device": settings.sam2_device,
        "mask_guides": str(settings.mask_guides.resolve()) if settings.mask_guides else None,
        "guides_sha256": hashlib.sha256(settings.mask_guides.read_bytes()).hexdigest() if settings.mask_guides else None,
        "min_registered_fraction": settings.min_registered_fraction,
        "max_mixed_mask_fraction": settings.max_mixed_mask_fraction,
        "max_colmap_rss_mb": settings.max_colmap_rss_mb,
        "ai_bridges": settings.ai_bridges,
        "auto_recover_cameras": settings.auto_recover_cameras,
        "ai_bridge_workflow": (str(settings.ai_bridge_workflow.resolve())
                               if settings.ai_bridge_workflow else None),
        "ai_bridge_workflow_sha256": (_sha256_file(settings.ai_bridge_workflow)
                                      if settings.ai_bridge_workflow else None),
    }
    # Preserve compatibility with immutable older MPS revision fingerprints.
    if settings.matching_device != "mps":
        settings_payload["matching_device"] = settings.matching_device
    settings_fingerprint = hashlib.sha256(json.dumps(
        settings_payload, sort_keys=True).encode()).hexdigest()
    # Older, blocked revisions can resume camera recovery after upgrading the
    # CLI without replaying their completed multi-hour COLMAP matching stage.
    previous_settings = {key: value for key, value in settings_payload.items()
                         if key != "auto_recover_cameras"}
    previous_fingerprint = hashlib.sha256(json.dumps(
        previous_settings, sort_keys=True).encode()).hexdigest()
    if resume:
        report = _read_report(report_path)
        compatible_upgrade = (settings.auto_recover_cameras
            and str(report.get("status", "")).startswith("blocked")
            and report.get("settings_sha256") == previous_fingerprint)
        if (report.get("source_manifest_sha256") != source_fingerprint
                or (report.get("settings_sha256") != settings_fingerprint and not compatible_upgrade)
                or report.get("asset_id") != asset_id):
            raise ReferenceError(RESUME_INPUTS_CHANGED_ERROR)
        if compatible_upgrade:
            report["settings_sha256"] = settings_fingerprint
            report["resume_upgrade"] = "enabled verified camera recovery without rerunning photos"
        if report.get("failed_worker", {}).get("stage") in report.get("completed_stages", []):
            report.pop("failed_worker", None)
        if report.get("status") == "private-GLB-candidate / review-required":
            final = target / "model.glb"
            if final.is_symlink() or not final.is_file():
                raise ReferenceError("Previously completed model.glb is missing; original result cannot be trusted")
            if _sha256_file(final) != report.get("model_sha256"):
                raise ReferenceError("Previously completed model.glb changed; refuse unverified resume")
            emit("done", "preflight")
            emit("finished", "export", "Already completed; existing GLB verified")
            return report
    else:
        if target.exists():
            raise ReferenceError(f"Output already exists; pass --resume or choose a fresh path: {target}")
        executable_path = os.environ.get("PATH", "")
        if settings.openmvs_dir:
            executable_path = f"{settings.openmvs_dir.resolve()}:{executable_path}"
        missing = [name for name in ("colmap", "InterfaceCOLMAP", "DensifyPointCloud",
                                    "ReconstructMesh", settings.blender)
                   if shutil.which(name, path=executable_path) is None]
        if missing:
            raise ReferenceError("Missing local reconstruction dependencies: " + ", ".join(missing))
        target.mkdir(parents=True)
        (target / "logs").mkdir()
        report = {"schema_version": 1, "asset_id": asset_id,
                  "output": str(target), "source_manifest_sha256": source_fingerprint,
                  "settings_sha256": settings_fingerprint,
                  "status": "running", "completed_stages": [], "warnings": [],
                  "distribution_allowed": False, "historical_fidelity_verified": False,
                  "model_glb": None}
    report["model_checkpoint"] = ({"path": str(checkpoint_file), "source": checkpoint_source,
                                    "sha256": _sha256_file(checkpoint_file)} if checkpoint_file else None)

    def checkpoint(status: str, stage: str | None = None, warnings: list[str] | None = None):
        report.update(status=status, stopped_stage=stage, warnings=warnings or [])
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if status.startswith("blocked"):
            emit("blocked", "mask_audit" if (stage or "").startswith("mask-audit-") else
                 stage or "preflight", "; ".join(warnings or []) or status)
        return report

    active_phase = "preflight"
    if runner is None:
        from .live_progress import run_monitored_worker

        def runner(argv: list[str], log: Path, environment: dict[str, str]) -> int:
            return run_monitored_worker(
                argv, log, environment, settings.timeout_seconds, stage=active_phase,
                root=target, emit=progress_callback)

    environment = os.environ.copy()
    if settings.openmvs_dir:
        environment["PATH"] = f"{settings.openmvs_dir.resolve()}:{environment.get('PATH', '')}"
    executable = [sys.executable, "-m", "kumiki_blender.cli"]

    def run_stage(name: str, argv: list[str], receipt: Path) -> dict[str, Any]:
        nonlocal active_phase
        phase = ("mask_audit" if name.startswith("mask-audit-") else
                 "camera_recovery" if name.startswith("camera_recovery") else
                 "recovery_coverage" if name.startswith("recovery_coverage") else name)
        emit("start", phase, f"Log: {target / 'logs' / (name + '.log')}")
        if name in report["completed_stages"]:
            emit("skip", phase)
            return _read_report(receipt)
        active_phase = phase
        code = runner(argv, target / "logs" / f"{name}.log", environment)
        if code:
            log_path = target / "logs" / f"{name}.log"
            try:
                with log_path.open("rb") as logfile:
                    logfile.seek(max(0, log_path.stat().st_size - 4096))
                    tail = logfile.read(4096).decode("utf-8", "replace")
                lines = [line.strip() for line in tail.splitlines() if line.strip()]
                diagnostic = lines[-1] if lines else ""
            except OSError:
                diagnostic = ""
            if not diagnostic or diagnostic == "error:":
                diagnostic = "Worker returned no exception details; retry this stage directly with Python -X faulthandler"
            report["failed_worker"] = {"stage": name, "exit_code": code,
                                       "diagnostic": diagnostic[:500]}
            checkpoint("blocked / worker-failed", name,
                       [f"Stage exited {code}: {diagnostic[:250]}; inspect logs/{name}.log"])
            raise ReferenceError(f"{name} failed; inspect {target / 'logs' / (name + '.log')}")
        record = _read_report(receipt)
        report["completed_stages"].append(name)
        if report.get("failed_worker", {}).get("stage") == name:
            # A successful resume supersedes this stage's previous native
            # error. Keep old logs on disk, not as a current failure status.
            report.pop("failed_worker", None)
        checkpoint("running")
        emit("done", phase)
        return record

    sparse = target / "cameras"
    prepared = target / "photo-preprocess"
    coverage_dir = target / "coverage"
    mask_dir = target / "masks"
    mask_source = settings.masks_dir.resolve() if settings.masks_dir else mask_dir / "masks"
    shared_cache = settings.mask_cache_dir.resolve() if settings.mask_cache_dir else target.parent / "_statue_mask_cache"
    photo_cache = settings.photo_cache_dir.resolve() if settings.photo_cache_dir else target.parent / "_statue_photo_cache"
    dense = target / "dense"
    review = target / "review"

    checkpoint("running")
    emit("done", "preflight")
    prep_argv = executable + ["prepare-statue-photos", "--photos", str(photos.resolve()),
        "--asset-id", asset_id, "--output", str(prepared), "--cache-dir", str(photo_cache),
        "--sam2-size", settings.sam2_size, "--sam2-device", settings.sam2_device]
    if settings.masks_dir is not None:
        prep_argv.append("--no-foreground")
    elif checkpoint_file is not None:
        prep_argv += ["--sam2-checkpoint", str(checkpoint_file)]
    if resume and prepared.exists() and "photo_preprocess" not in report["completed_stages"]:
        prep_argv.append("--resume")
    prepared_report = run_stage("photo_preprocess", prep_argv,
                                prepared / "photo-preprocess-report.json")
    if not str(prepared_report.get("status", "")).startswith("candidate"):
        return checkpoint("blocked / photo-preprocess-required", "photo_preprocess",
                          ["No valid source-photo preprocessing cache"])
    report["photo_cache"] = {"normalized_hits": prepared_report.get("normalized_hits"),
                             "normalized_misses": prepared_report.get("normalized_misses"),
                             "foreground_hits": prepared_report.get("foreground_hits"),
                             "foreground_misses": prepared_report.get("foreground_misses"),
                             "qa_hits": prepared_report.get("qa_hits"),
                             "qa_misses": prepared_report.get("qa_misses"),
                             "qa_warnings": prepared_report.get("qa_warning_count"),
                             "mask_qa": prepared_report.get("mask_qa"),
                             "auxiliary_model_status": prepared_report.get("auxiliary_model_status"),
                             "refinement_attempts": prepared_report.get("refinement_attempts"),
                             "refinement_accepted": prepared_report.get("refinement_accepted")}
    report["early_photo_qa"] = "photo-preprocess/photo-qa-contact-sheet.jpg"
    camera_argv = executable + ["reconstruct-photos", "--photos", str(photos.resolve()),
        "--asset-id", asset_id, "--output", str(sparse), "--engine", "colmap",
        "--normalized-cache-dir", str(photo_cache),
        "--matching-preset", "sift-lightglue", "--matching-device", settings.matching_device,
        "--source-qa-report", str(prepared / "photo-preprocess-report.json"),
        "--matching-pairs-per-process", "8", "--max-colmap-rss-mb", str(settings.max_colmap_rss_mb),
        "--undistort", "--timeout-seconds", str(settings.timeout_seconds)]
    if resume and sparse.exists() and "cameras" not in report["completed_stages"]:
        camera_argv.append("--resume")
    camera = run_stage("cameras", camera_argv,
        sparse / "reconstruction-report.json")
    registered = camera.get("registered_images", 0)
    total = camera.get("staged_independent_images", 0)
    if not isinstance(registered, int) or not isinstance(total, int) or total <= 0:
        return checkpoint("blocked / registration-insufficient", "cameras", ["Invalid registration counts"])
    if not str(camera.get("status", "")).startswith("candidate") or camera.get("undistorted_images") != registered:
        return checkpoint("blocked / registration-insufficient", "cameras", ["Camera reconstruction is not a successful undistorted candidate"])

    front = ({"filename": settings.front_photo, "source": "user-annotation",
              "confidence": "explicit"} if settings.front_photo else
             _infer_front_photo(sparse, camera))
    report["front_selection"] = front

    script = Path(__file__).resolve().parents[3] / "scripts" / "audit_photo_coverage.py"
    run_stage("coverage", [sys.executable, str(script), str(sparse),
                           "--output", str(coverage_dir), "--subject-type", "statue"]
              + (["--front-photo", front["filename"]] if front["filename"] else []),
              coverage_dir / "photo-advice.json")
    coverage = _read_report(coverage_dir / "photo-advice.json")
    issues = _coverage_warnings(coverage, registered / total, settings.min_registered_fraction)
    report["coverage_report"] = str(Path("coverage/photo-advice.md"))
    report["registered_images"] = registered
    report["source_images"] = total
    report["unregistered_photo_advice"] = coverage.get("existing_photo_priorities", [])
    if issues and camera.get("unregistered_filenames") and settings.auto_recover_cameras:
        candidate_source = sparse
        initial_registered = registered
        recovery_rounds = []
        # Previously blocked revisions can try the new confidence-ranked
        # cross-group matcher without replacing immutable prior results or
        # repeating the (potentially hours-long) original camera solve.
        old_followup = target / "camera-recovery-followup/camera-recovery-report.json"
        retry_old_followup = (resume and "camera_recovery_followup" in report["completed_stages"]
                              and old_followup.is_file() and not old_followup.is_symlink()
                              and "cohort_attempts" not in _read_report(old_followup))
        for round_number in range(1, 4 if retry_old_followup else 3):
            stage = ("camera_recovery" if round_number == 1 else
                     "camera_recovery_followup" if round_number == 2 else
                     "camera_recovery_confidence")
            coverage_stage = ("recovery_coverage" if round_number == 1 else
                              "recovery_coverage_followup" if round_number == 2 else
                              "recovery_coverage_confidence")
            recovery = target / ("camera-recovery" if round_number == 1 else
                                 "camera-recovery-followup" if round_number == 2 else
                                 "camera-recovery-confidence")
            new_coverage_dir = target / ("recovered-coverage" if round_number == 1 else
                                         "recovered-coverage-followup" if round_number == 2 else
                                         "recovered-coverage-confidence")
            recovered = run_stage(stage, executable + [
                "recover-registered-cameras", "--reconstruction", str(candidate_source),
                "--output", str(recovery), "--timeout-seconds", str(settings.timeout_seconds)],
                recovery / "camera-recovery-report.json")
            report["camera_recovery_report"] = str(recovery.relative_to(target) /
                                                  "camera-recovery-report.json")
            report["automatic_camera_recovery_status"] = recovered.get("status", "unknown")
            recovery_rounds.append({"stage": stage, "source": str(candidate_source),
                                    "status": recovered.get("status"),
                                    "new_cameras": recovered.get("new_cameras", 0)})
            report["camera_recovery_rounds"] = recovery_rounds
            if recovered.get("candidate_accepted") is not True:
                if retry_old_followup and round_number == 2:
                    # The old second attempt is a recorded failure, not a
                    # reason to lose the valid candidate from round one.
                    continue
                break
            recovered_camera = _read_report(recovery / "reconstruction-report.json")
            candidate_registered = recovered_camera.get("registered_images", 0)
            candidate_total = recovered_camera.get("staged_independent_images", 0)
            if (not isinstance(candidate_registered, int) or not isinstance(candidate_total, int)
                    or candidate_total <= 0 or candidate_registered <= (initial_registered if round_number == 1
                                                                          else previous_registered)
                    or recovered_camera.get("undistorted_images") != candidate_registered):
                break
            previous_registered = candidate_registered
            report["camera_recovery_new_cameras"] = candidate_registered - initial_registered
            run_stage(coverage_stage, [sys.executable, str(script), str(recovery),
                "--output", str(new_coverage_dir), "--subject-type", "statue"]
                + (["--front-photo", front["filename"]] if front["filename"] else []),
                new_coverage_dir / "photo-advice.json")
            proposed = _read_report(new_coverage_dir / "photo-advice.json")
            revised_issues = _coverage_warnings(
                proposed, candidate_registered / candidate_total, settings.min_registered_fraction)
            report["camera_recovery_coverage_warnings"] = revised_issues
            report["verified_camera_candidate"] = str(recovery)
            report["verified_camera_candidate_registered"] = candidate_registered
            if not revised_issues:
                sparse = recovery
                camera = recovered_camera
                coverage = proposed
                coverage_dir = new_coverage_dir
                registered, total = candidate_registered, candidate_total
                issues = []
                report["registered_images"] = registered
                report["source_images"] = total
                report["coverage_report"] = str(new_coverage_dir.relative_to(target) / "photo-advice.md")
                report["unregistered_photo_advice"] = coverage.get("existing_photo_priorities", [])
                report["active_camera_revision"] = str(recovery)
                break
            if not recovered_camera.get("unregistered_filenames"):
                break
            candidate_source = recovery
    else:
        emit("skip", "camera_recovery")
    if "active_camera_revision" not in report:
        emit("skip", "recovery_coverage")
    if settings.ai_bridges:
        emit("start", "ai_bridges", "Generating private synthetic reference previews")
        from .ai_bridge_previews import build_ai_bridge_previews
        bridge_output = target / "ai-bridge-previews"
        bridge_report = (_read_report(bridge_output / "bridge-preview-report.json")
                         if resume and bridge_output.exists() else
                         build_ai_bridge_previews(
                             sparse, coverage, bridge_output,
                             workflow=settings.ai_bridge_workflow,
                             asset_id=asset_id,
                             require_mps=settings.sam2_device != "cpu"))
        report["ai_bridge_previews"] = "ai-bridge-previews/bridge-preview-report.json"
        report["ai_bridge_preview_status"] = bridge_report["status"]
        emit("done", "ai_bridges")
    if camera.get("unregistered_filenames"):
        # Before asking for more pictures, analyze existing verified match
        # graphs for independent-cohort and seeded-camera recovery.
        try:
            from .sfm_cohort_recovery import plan_cohort_recovery
            recovery = plan_cohort_recovery(sparse)
            recovery_file = coverage_dir / "cohort-recovery-plan.json"
            recovery_file.write_text(json.dumps(recovery, indent=2) + "\n", encoding="utf-8")
            report["camera_recovery_plan"] = str(recovery_file.relative_to(target))
            report["camera_recovery_status"] = recovery["status"]
        except (OSError, ValueError, ImportError) as error:
            report["camera_recovery_status"] = "diagnostic-unavailable"
            report["camera_recovery_error"] = str(error)[:250]
    if front["confidence"] in ("tentative", "none"):
        report["orientation_warning"] = (
            "Automatically selected front is unverified; side labels are provisional" if front["filename"]
            else "No reliable frontal photo hint; true left/right geometry remains unverified")
    if issues:
        # A verified recovered candidate can clear the numeric registration
        # threshold while still failing viewpoint coverage. Report its actual
        # remaining blockers; do not imply the original 47-camera model is
        # promoted or repeat an outdated 78.3% registration warning.
        candidate_count = report.get("verified_camera_candidate_registered", 0)
        if candidate_count > registered and "camera_recovery_coverage_warnings" in report:
            report["initial_coverage_warnings"] = issues
            issues = report["camera_recovery_coverage_warnings"]
        return checkpoint("blocked / missing-photo-coverage", "coverage", issues)

    if settings.masks_dir is None:
        mask_args = executable + ["prepare-dense-masks", "--reconstruction", str(sparse),
            "--output", str(mask_dir), "--segmenter", "sam2",
            "--sam2-size", settings.sam2_size, "--sam2-device", settings.sam2_device,
            "--sam2-checkpoint", str(checkpoint_file),
            "--cache-dir", str(shared_cache)]
        if settings.mask_guides:
            mask_args += ["--guides", str(settings.mask_guides.resolve())]
        if resume and mask_dir.exists() and "masks" not in report["completed_stages"]:
            mask_args.append("--resume")
        run_stage("masks", mask_args, mask_dir / "mask-report.json")
        mask_record = _read_report(mask_dir / "mask-report.json")
        report["mask_cache"] = mask_record.get("cache")
    else:
        mask_record = {"images": []}
        emit("skip", "masks")

    if not mask_source.is_dir():
        return checkpoint("blocked / mask-review-required", "masks", ["Expected mask directory is missing"])
    if settings.masks_dir is not None:
        _verify_mask_revision(mask_source, sparse)
    mask_files = sorted(mask_source.glob("*.mask.png"))
    if not mask_files:
        return checkpoint("blocked / mask-review-required", "masks", ["Mask directory contains no rectified PNG masks"])
    # Distinct edited mask revisions must receive distinct audit receipts, even
    # when the user resumes the very same sparse camera reconstruction.
    fingerprint = hashlib.sha256()
    for path in mask_files:
        if path.is_symlink() or not path.is_file():
            raise ReferenceError(f"Mask is not a regular file: {path}")
        fingerprint.update(path.name.encode())
        with path.open("rb") as stream:
            fingerprint.update(hashlib.file_digest(stream, "sha256").digest())
    mask_key = fingerprint.hexdigest()[:16]
    audit_stage = f"mask-audit-{mask_key}"
    mask_audit = target / audit_stage
    audit_report = run_stage(audit_stage, executable + ["audit-dense-masks",
        "--reconstruction", str(sparse), "--rectified-masks", str(mask_source),
        "--output", str(mask_audit)], mask_audit / "mask-consistency-report.json")
    report["mask_contact_sheet"] = ("masks/mask-contact-sheet.jpg" if settings.masks_dir is None else None)
    report["mask_audit_contact_sheet"] = f"{audit_stage}/mask-consistency-contact-sheet.jpg"
    if settings.masks_dir is None:
        issues = _mask_warnings(mask_record, audit_report, settings.max_mixed_mask_fraction)
    else:
        issues = ([f"Cross-view mask disagreement {audit_report.get('mixed_label_fraction')} exceeds "
                   f"{settings.max_mixed_mask_fraction:.0%}"]
                  if audit_report.get("mixed_label_fraction", 1) > settings.max_mixed_mask_fraction else [])
        generated = mask_dir / "mask-report.json"
        if generated.is_file():
            original = _read_report(generated)
            original_hashes = {item.get("filename"): item.get("mask_sha256")
                               for item in original.get("images", [])}
            revised = _read_report(mask_source.parent / "mask-report.json")
            revised_hashes = {item.get("filename"): item.get("mask_sha256")
                              for item in revised.get("images", [])}
            if original_hashes == revised_hashes:
                issues.append("Reviewed masks are byte-identical to the rejected automatic candidates")
    issues.extend(_possible_blue_sky_leaks(sparse, mask_source, [item["filename"] for item in audit_report["images"]]))
    if issues:
        return checkpoint("blocked / mask-review-required", audit_stage, issues)

    dense_argv = executable + ["build-dense-photos",
        "--reconstruction", str(sparse), "--rectified-masks", str(mask_source),
        "--output", str(dense), "--backend", "openmvs", "--resolution-level", "0",
        "--max-resolution", "3200", "--number-views", "5", "--no-refine",
        "--no-texture", "--timeout-seconds", str(settings.timeout_seconds)]
    if resume and dense.exists() and "dense" not in report["completed_stages"]:
        dense_argv.append("--resume")
    mesh_record = run_stage("dense", dense_argv,
        dense / "dense-report.json")
    if not str(mesh_record.get("status", "")).startswith("dense-mesh-candidate"):
        return checkpoint("blocked / dense-mesh-review-required", "dense", ["No validated dense mesh"])
    record = run_stage("review", executable + ["render-dense-blender-review",
        "--dense-revision", str(dense), "--output", str(review),
        "--blender", settings.blender, "--no-renders"],
        review / "blender-review-report.json")
    name = record.get("review_glb")
    if not str(record.get("status", "")).startswith("private-Blender-mesh-candidate") or name != "private-dense-review.glb":
        return checkpoint("blocked / blender-review-required", "review", ["Invalid Blender review GLB"])
    glb = review / name
    if glb.is_symlink() or not glb.is_file():
        return checkpoint("blocked / blender-review-required", "review", ["GLB is missing or a symbolic link"])
    emit("start", "export", "Auditing GLB structure and writing final model")
    try:
        report["glb_audit"] = audit_glb(glb)
    except AuditError as error:
        return checkpoint("blocked / blender-review-required", "export", [str(error)])
    shutil.copyfile(glb, target / "model.glb")
    report["model_glb"] = "model.glb"
    report["model_sha256"] = _sha256_file(target / "model.glb")
    completed = checkpoint("private-GLB-candidate / review-required")
    emit("done", "export")
    emit("finished", "export")
    return completed
