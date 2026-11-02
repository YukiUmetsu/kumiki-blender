"""GPU-rasterized Gaussian splat camera tours and private MP4 revisions.

SplatTransform is an external PlayCanvas tool: the frames are rendered from
actual Gaussian attributes, then FFmpeg encodes the resulting WebP sequence.
This module does not turn splats into polygon meshes or claim visual fidelity.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .dense import _sha256
from .references import ReferenceError
from .splats import audit_splat_artifact


_PLY_TYPES = {"char": "b", "uchar": "B", "short": "h", "ushort": "H",
              "int": "i", "uint": "I", "float": "f", "float32": "f",
              "double": "d", "float64": "d"}


@dataclass(frozen=True)
class SplatVideoSettings:
    fps: int = 30
    seconds: int = 6
    width: int = 1920
    height: int = 1080
    fov: float = 50.0
    orbit_degrees: float = 115.0
    start_degrees: float = -65.0
    elevation_degrees: float = 12.0
    target: tuple[float, float, float] | None = None
    distance: float | None = None
    shutter: float = 0.0
    splat_transform: str = "splat-transform"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    timeout_seconds: int = 7200
    viewer: bool = False

    def validate(self) -> "SplatVideoSettings":
        if type(self.fps) is not int or self.fps not in (24, 25, 30, 60):
            raise ReferenceError("Video FPS must be 24, 25, 30 or 60")
        if type(self.seconds) is not int or not 2 <= self.seconds <= 30:
            raise ReferenceError("Video duration must be 2–30 seconds")
        if (type(self.width) is not int or type(self.height) is not int or
                not 256 <= self.width <= 4096 or not 256 <= self.height <= 4096 or
                self.width % 2 or self.height % 2):
            raise ReferenceError("Video dimensions must be even numbers in 256–4096")
        for name, value, minimum, maximum in (
            ("FOV", self.fov, 15, 100), ("orbit angle", self.orbit_degrees, -360, 360),
            ("start angle", self.start_degrees, -360, 360),
            ("elevation", self.elevation_degrees, -80, 80),
            ("shutter", self.shutter, 0, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not minimum <= value <= maximum:
                raise ReferenceError(f"Video {name} must be finite and in {minimum}–{maximum}")
        if self.distance is not None and (isinstance(self.distance, bool) or
               not isinstance(self.distance, (float, int)) or
               not math.isfinite(self.distance) or self.distance <= 0):
            raise ReferenceError("Video camera distance must be positive and finite")
        if self.target is not None and (len(self.target) != 3 or
               any(not math.isfinite(c) for c in self.target)):
            raise ReferenceError("Camera target must contain three finite coordinates")
        if type(self.timeout_seconds) is not int or not 30 <= self.timeout_seconds <= 86_400:
            raise ReferenceError("Video timeout must be 30–86,400 seconds")
        for binary in (self.splat_transform, self.ffmpeg, self.ffprobe):
            if not isinstance(binary, str) or not binary.strip() or "\0" in binary:
                raise ReferenceError("Video executables must be nonempty paths or commands")
        return self


def _find_splat(source: str | Path) -> Path:
    entry = Path(source)
    if entry.is_symlink():
        raise ReferenceError("Splat input must not be a symlink")
    if entry.is_dir():
        report_path = entry / "splat-report.json"
        if report_path.is_symlink() or not report_path.is_file():
            raise ReferenceError("Splat revision is missing splat-report.json")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (ValueError, KeyError, TypeError) as error:
            raise ReferenceError("Invalid splat revision report") from error
        if not isinstance(report, dict) or not isinstance(report.get("status"), str):
            raise ReferenceError("Invalid splat revision report")
        if not report["status"].startswith("candidate"):
            raise ReferenceError("Splat revision has not passed training and artifact audit")
        matches = [entry / name for name in ("splat.ply", "splat.spz")
                   if (entry / name).exists() or (entry / name).is_symlink()]
        if len(matches) != 1:
            raise ReferenceError("Splat revision must have exactly one splat.ply or splat.spz")
        entry = matches[0]
        if entry.is_symlink() or not entry.is_file():
            raise ReferenceError("Splat revision must contain a regular, non-linked artifact")
        recorded_artifact = report.get("artifact")
        recorded_hash = (recorded_artifact.get("sha256")
                         if isinstance(recorded_artifact, dict) else None)
        if not isinstance(recorded_hash, str) or recorded_hash != _sha256(entry):
            raise ReferenceError("Splat revision artifact has changed since training")
    if entry.is_symlink() or not entry.is_file():
        raise ReferenceError("Splat input must be a real file")
    return entry.resolve()


def _position_bounds(ply: Path) -> tuple[tuple[float, float, float], float]:
    """Estimate the central object region from a bounded, distributed PLY sample.

    Trim each axis to the central 90% to reduce sparse floating outliers. The
    resulting camera is only an initial composition; it needs visual review.
    """
    with ply.open("rb") as stream:
        header = stream.read(65_536)
    end = header.find(b"end_header\n")
    if end < 0:
        raise ReferenceError("Cannot read Gaussian PLY header")
    header_size = end + len(b"end_header\n")
    count = None
    fields: list[tuple[str, str]] = []
    in_vertex = False
    for line in header[:end].decode("ascii").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
        elif in_vertex and len(parts) == 3 and parts[0] == "property":
            fields.append((parts[2], parts[1]))
    if count is None or count < 1:
        raise ReferenceError("Gaussian PLY has no vertices")
    fmt = "<" + "".join(_PLY_TYPES[t] for _, t in fields)
    stride = struct.calcsize(fmt)
    offsets = {name: index for index, (name, _) in enumerate(fields)}
    if not all(name in offsets for name in ("x", "y", "z")):
        raise ReferenceError("Gaussian PLY is missing XYZ positions")
    sample_count = min(count, 2048)
    axes: tuple[list[float], list[float], list[float]] = ([], [], [])
    with ply.open("rb") as stream:
        for n in range(sample_count):
            index = n * (count - 1) // max(1, sample_count - 1)
            stream.seek(header_size + index * stride)
            values = struct.unpack(fmt, stream.read(stride))
            for axis, name in enumerate(("x", "y", "z")):
                v = float(values[offsets[name]])
                if not math.isfinite(v):
                    raise ReferenceError("Non-finite Gaussian position in camera sample")
                axes[axis].append(v)
    centers = []
    extents = []
    for values in axes:
        values.sort()
        low = values[int((len(values) - 1) * 0.05)]
        high = values[int((len(values) - 1) * 0.95)]
        centers.append((low + high) / 2)
        extents.append(max(0, high - low))
    extent = max(extents)
    if extent < 1e-6:
        raise ReferenceError("Gaussian sample is degenerate; specify a camera track instead")
    return (tuple(centers), extent)


def _ease(t: float) -> float:
    """Soft start, purposeful orbit, and a steady ending hero view."""
    x = min(1.0, max(0.0, (t - 0.10) / 0.78))
    return x * x * (3 - 2 * x)


def make_orbit_track(center: tuple[float, float, float], distance: float,
                     settings: SplatVideoSettings) -> dict:
    frames = []
    phi = math.radians(settings.elevation_degrees)
    for frame in range(settings.fps * settings.seconds):
        t = frame / max(1, settings.fps * settings.seconds - 1)
        azimuth = math.radians(settings.start_degrees + settings.orbit_degrees * _ease(t))
        frames.append({"position": [round(center[0] + distance * math.cos(phi) * math.sin(azimuth), 8),
                                     round(center[1] + distance * math.sin(phi), 8),
                                     round(center[2] + distance * math.cos(phi) * math.cos(azimuth), 8)],
                       "target": list(center), "up": [0, 1, 0], "fov": settings.fov})
    return {"frameRate": settings.fps, "frames": frames}


def _video_output_plan(source: str | Path, output: str | Path,
                       settings: SplatVideoSettings, camera_track: str | Path | None) -> dict:
    settings.validate()
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists():
        raise ReferenceError(f"Video revision already exists: {target}")
    file = _find_splat(source)
    artifact = audit_splat_artifact(file)
    executables = {}
    for key, command in (("splat_transform", settings.splat_transform),
                         ("ffmpeg", settings.ffmpeg), ("ffprobe", settings.ffprobe)):
        found = shutil.which(command)
        if found is None:
            raise ReferenceError(f"Video executable not found: {command}")
        executables[key] = str(Path(found).resolve())
    track = None
    camera = None
    if camera_track is not None:
        track = Path(camera_track)
        if track.is_symlink() or not track.exists() or not (track.is_file() or track.is_dir()):
            raise ReferenceError("Camera track must be an existing, non-linked file or directory")
        track = track.resolve()
        if track.is_file() and track.suffix == ".json":
            try:
                data = json.loads(track.read_text(encoding="utf-8"))
            except (ValueError, OSError) as error:
                raise ReferenceError("Camera track JSON is invalid") from error
            if isinstance(data, dict) and "frames" in data:
                # Never silently encode an MP4 at a different frame rate.
                if (not isinstance(data["frames"], list) or
                        data.get("frameRate", 30) != settings.fps or
                        len(data["frames"]) != settings.fps * settings.seconds):
                    raise ReferenceError("Camera track frameRate/frame count must match --fps and --seconds")
                for frame in data["frames"]:
                    if not isinstance(frame, dict):
                        raise ReferenceError("Camera track frames must be objects")
                    for name in ("position", "target"):
                        vector = frame.get(name)
                        if (not isinstance(vector, list) or len(vector) != 3 or
                                any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                                    not math.isfinite(x) for x in vector)):
                            raise ReferenceError(f"Camera track {name} must be a finite XYZ vector")
                    if frame["position"] == frame["target"]:
                        raise ReferenceError("Camera track position cannot equal target")
    else:
        if file.suffix == ".spz" and (settings.target is None or settings.distance is None):
            raise ReferenceError("SPZ orbit needs --target X Y Z and --distance, or use --camera-track")
        center, extent = _position_bounds(file) if file.suffix == ".ply" else (settings.target, None)
        center = settings.target or center
        distance = settings.distance or max(0.1, extent * 1.45)
        camera = {"target": center, "distance": distance, "kind": "orbit",
                  "start_degrees": settings.start_degrees, "arc_degrees": settings.orbit_degrees,
                  "elevation_degrees": settings.elevation_degrees, "fov": settings.fov}
    return {"schema_version": 1, "source": str(file), "source_artifact": artifact,
            "output": str(target), "fps": settings.fps, "seconds": settings.seconds,
            "resolution": [settings.width, settings.height], "camera": camera,
            "external_track": str(track) if track else None, "executables": executables,
            "shutter": settings.shutter, "viewer_requested": settings.viewer,
            "status": "preflight-only / GPU-render-and-review-required",
            "distribution_allowed": False}


def plan_splat_video(source: str | Path, output: str | Path,
                     settings: SplatVideoSettings, *,
                     camera_track: str | Path | None = None) -> dict:
    return _video_output_plan(source, output, settings, camera_track)


def _run(command: list[str], log_file: Path, timeout: int) -> int:
    with log_file.open("w", encoding="utf-8") as log:
        try:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            return 124
        except OSError as error:
            log.write(f"\nUnable to launch renderer: {error}\n")
            return 127


def _probe_mp4(path: Path, ffprobe: str, expected_frames: int,
               settings: SplatVideoSettings) -> dict:
    completed = subprocess.run([
        ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_read_frames:format=duration",
        "-of", "json", str(path)], capture_output=True, text=True,
        timeout=30, check=False)
    if completed.returncode:
        raise ReferenceError("ffprobe could not decode MP4 metadata")
    try:
        data = json.loads(completed.stdout)
        stream = data["streams"][0]
        ratio = stream["avg_frame_rate"].split("/")
        fps = float(ratio[0]) / float(ratio[1])
        duration = float(data["format"]["duration"])
        decoded_frames = int(stream["nb_read_frames"])
        if (stream["codec_name"] != "h264" or stream["width"] != settings.width or
                stream["height"] != settings.height or abs(fps - settings.fps) > 0.01 or
                abs(duration - settings.seconds) > 1 / settings.fps + 0.01 or
                decoded_frames != expected_frames):
            raise ReferenceError("Encoded MP4 does not match requested FPS, size, length or frame count")
        return {"codec": "h264", "width": settings.width, "height": settings.height,
                "fps": fps, "duration_seconds": duration,
                "decoded_frames": decoded_frames, "encoded_frames": decoded_frames}
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError) as error:
        raise ReferenceError("MP4 metadata is invalid or incomplete") from error


def render_splat_video(source: str | Path, output: str | Path,
                       settings: SplatVideoSettings, *,
                       camera_track: str | Path | None = None) -> dict:
    """Render real Gaussian splat frames, verify contiguity, encode an MP4."""
    plan = _video_output_plan(source, output, settings, camera_track)
    target = Path(plan["output"])
    target.mkdir(parents=True, exist_ok=False)
    frames_dir = target / "frames"
    frames_dir.mkdir()
    logs = target / "logs"
    logs.mkdir()
    report = dict(plan)
    report.update(status="in-progress / GPU-render-required", failed_stage=None,
                  exit_code=None)

    def checkpoint() -> dict:
        (target / "video-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                  encoding="utf-8")
        return report

    checkpoint()
    track = plan["external_track"]
    if track is None:
        camera = plan["camera"]
        track_file = target / "camera-track.json"
        track_file.write_text(json.dumps(make_orbit_track(tuple(camera["target"]),
                            camera["distance"], settings), indent=2) + "\n",
                              encoding="utf-8")
        track = str(track_file)
    else:
        if Path(track).is_file():
            recorded = _sha256(Path(track))
            report["external_track_sha256"] = recorded
            # JSON tracks are copied into the private revision so a later edit
            # does not silently alter the camera path of this export.
            if Path(track).suffix == ".json":
                copied = target / "camera-track.json"
                shutil.copyfile(track, copied)
                if _sha256(copied) != recorded:
                    report.update(status="failed / review-required", failed_stage="camera-track-change",
                                  exit_code=3)
                    return checkpoint()
                track = str(copied)
        else:
            report["external_track_warning"] = "SuperSplat project directory is an external dependency and was not snapshotted"
    render_cmd = [plan["executables"]["splat_transform"], plan["source"],
                  str(frames_dir / "frame.webp"), "--camera-track", track,
                  "--resolution", f"{settings.width}x{settings.height}"]
    if settings.shutter:
        render_cmd.extend(("--shutter", str(settings.shutter)))
    report["commands"] = {"render": render_cmd}
    checkpoint()
    code = _run(render_cmd, logs / "render.log", settings.timeout_seconds)
    if code:
        report.update(status="failed / review-required", failed_stage="splat-rasterizer",
                      exit_code=code)
        return checkpoint()
    total = settings.fps * settings.seconds
    expected = {f"frame.{i:04d}.webp" for i in range(total)}
    produced = {p.name for p in frames_dir.iterdir()}
    if (produced != expected or
            any((frames_dir / name).is_symlink() or not (frames_dir / name).is_file() or
                (frames_dir / name).stat().st_size < 16 for name in expected)):
        report.update(status="failed / review-required", failed_stage="frame-sequence",
                      exit_code=3, expected_frames=total, actual_frames=len(produced))
        return checkpoint()
    command = [plan["executables"]["ffmpeg"], "-hide_banner", "-nostdin", "-loglevel", "error",
               "-framerate", str(settings.fps), "-start_number", "0", "-i",
               str(frames_dir / "frame.%04d.webp"), "-frames:v", str(total),
               "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               "-movflags", "+faststart", str(target / "video.mp4")]
    report["commands"]["encode"] = command
    code = _run(command, logs / "ffmpeg.log", settings.timeout_seconds)
    if code:
        report.update(status="failed / review-required", failed_stage="ffmpeg",
                      exit_code=code)
        return checkpoint()
    try:
        report["video"] = _probe_mp4(target / "video.mp4", plan["executables"]["ffprobe"],
                                     total, settings)
        report["video"].update(bytes=(target / "video.mp4").stat().st_size,
                               sha256=_sha256(target / "video.mp4"))
    except (ReferenceError, OSError, subprocess.TimeoutExpired) as error:
        report.update(status="failed / review-required", failed_stage="mp4-audit",
                      exit_code=3, error=str(error))
        return checkpoint()
    decode_cmd = [plan["executables"]["ffmpeg"], "-hide_banner", "-nostdin",
                  "-v", "error", "-xerror", "-i", str(target / "video.mp4"),
                  "-f", "null", "-"]
    report["commands"]["decode_audit"] = decode_cmd
    code = _run(decode_cmd, logs / "decode-audit.log", settings.timeout_seconds)
    if code:
        report.update(status="failed / review-required", failed_stage="mp4-decode",
                      exit_code=code)
        return checkpoint()
    sheet_cmd = [plan["executables"]["ffmpeg"], "-hide_banner", "-nostdin",
                 "-loglevel", "error", "-i", str(target / "video.mp4"),
                 "-vf", f"fps=6/{settings.seconds},scale=320:-1:flags=lanczos,tile=3x2",
                 "-frames:v", "1", "-q:v", "2", str(target / "contact-sheet.jpg")]
    report["commands"]["contact_sheet"] = sheet_cmd
    code = _run(sheet_cmd, logs / "contact-sheet.log", settings.timeout_seconds)
    if code or not (target / "contact-sheet.jpg").is_file():
        report.update(status="failed / review-required", failed_stage="contact-sheet",
                      exit_code=code or 3)
        return checkpoint()
    if _sha256(Path(plan["source"])) != plan["source_artifact"]["sha256"]:
        report.update(status="failed / review-required", failed_stage="source-changed-during-render",
                      exit_code=3)
        return checkpoint()
    if settings.viewer:
        command = [plan["executables"]["splat_transform"], plan["source"],
                   str(target / "viewer.html")]
        report["commands"]["viewer"] = command
        code = _run(command, logs / "viewer.log", settings.timeout_seconds)
        if code or not (target / "viewer.html").is_file() or (target / "viewer.html").stat().st_size < 16:
            report.update(status="failed / video-created-but-viewer-required",
                          failed_stage="viewer-export", exit_code=code or 3)
            return checkpoint()
        report["viewer"] = {"file": "viewer.html", "sha256": _sha256(target / "viewer.html")}
    if settings.viewer and _sha256(Path(plan["source"])) != plan["source_artifact"]["sha256"]:
        report.update(status="failed / review-required", failed_stage="source-changed-during-viewer-export",
                      exit_code=3)
        return checkpoint()
    report.update(status="candidate / visual-and-licensing-review-required",
                  exit_code=0, files={"video": "video.mp4", "frames": "frames",
                                      "contact_sheet": "contact-sheet.jpg",
                                      "camera_track": "camera-track.json" if (target / "camera-track.json").is_file() else None,
                                      "render_log": "logs/render.log", "ffmpeg_log": "logs/ffmpeg.log",
                                      "decode_audit_log": "logs/decode-audit.log",
                                      "contact_sheet_log": "logs/contact-sheet.log"},
                  limitations="MP4 technical checks passed; actual GPU playback, composition and source rights still require human review.")
    return checkpoint()
