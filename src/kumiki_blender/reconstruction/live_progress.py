"""Live, honest CLI progress for multi-hour photo reconstruction.

The main bar counts *completed phases*, not estimated elapsed work. Worker
subprogress is shown separately only when a real receipt records its count.
JSON results stay on stdout; all progress lives on stderr.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, TextIO


_PHASES = (
    ("preflight", "Checking dependencies"),
    ("photo_preprocess", "Preparing and caching source photographs"),
    ("cameras", "Matching photos and solving cameras"),
    ("coverage", "Checking camera coverage"),
    ("camera_recovery", "Recovering cameras from verified photo tracks"),
    ("recovery_coverage", "Rechecking camera coverage"),
    ("ai_bridges", "AI bridge reference previews"),
    ("masks", "Segmenting statue with SAM2"),
    ("mask_audit", "Checking mask consistency"),
    ("dense", "Reconstructing dense 3D geometry"),
    ("review", "Preparing Blender review"),
    ("export", "Verifying and exporting GLB"),
)
_SHORT_NAMES = {"preflight": "Prep", "photo_preprocess": "Photo Cache", "cameras": "Cameras", "coverage": "Coverage",
                "camera_recovery": "Recover", "recovery_coverage": "Recheck",
                "ai_bridges": "AI Bridges", "masks": "SAM2 Masks",
                "mask_audit": "Mask QA", "dense": "OpenMVS", "review": "Blender",
                "export": "GLB Export"}


def _safe(text: Any, limit: int = 110) -> str:
    """Never render terminal control sequences emitted by external workers."""
    value = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", str(text))
    value = "".join(char if char.isprintable() else " " for char in value)
    return " ".join(value.split())[:limit]


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            return {}
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _stage_detail(root: Path, stage: str) -> str | None:
    """Read partial receipts written by existing workers; never invent a %."""
    if stage == "photo_preprocess":
        report = _read_receipt(root / "photo-preprocess/photo-preprocess-report.json")
        entries = report.get("images")
        if isinstance(entries, list):
            done = len(entries)
            hits = report.get("normalized_hits", 0)
            total = report.get("source_images")
            count = f"{done}/{total} ({done * 100 // total}%)" if (
                type(total) is int and total > 0 and done <= total) else str(done)
            return f"Source photos: {count} · {hits} normalization cache hits"
    if stage == "cameras":
        report = _read_receipt(root / "cameras/reconstruction-report.json")
        pairs = report.get("matching_progress") or {}
        done, total = pairs.get("completed_pairs"), pairs.get("total_pairs")
        if (type(done) is int and type(total) is int and total > 0
                and 0 <= done <= total):
            return f"Feature matching: {done}/{total} pairs ({done * 100 // total}%)"
        commands = report.get("commands") or []
        if commands and isinstance(commands[-1], dict):
            name = str(commands[-1].get("stage", ""))
            return "Last completed camera step: " + _safe(name, 38)
    if stage == "masks":
        report = _read_receipt(root / "masks/mask-report.json")
        cameras = _read_receipt(root / "cameras/reconstruction-report.json")
        images = report.get("images")
        total = cameras.get("undistorted_images")
        if isinstance(images, list) and type(total) is int and total > 0:
            done = len(images)
            if done <= total:
                hits = (report.get("cache") or {}).get("hits", 0)
                return f"SAM2 masks: {done}/{total} photos ({done * 100 // total}%) · {hits} cached"
    if stage == "dense":
        report = _read_receipt(root / "dense/dense-report.json")
        commands = report.get("commands") or []
        if isinstance(commands, list) and commands:
            current = commands[-1].get("stage") if isinstance(commands[-1], dict) else None
            if current in ("interface", "densify", "mesh"):
                return f"OpenMVS: completed {current} · {len(commands)}/3 steps"
    return None


class LiveProgress:
    """TTY redraw or sparse, timestamped plain-text phase updates."""

    def __init__(self, asset_id: str, *, ai_bridges: bool = False,
                 stream: TextIO | None = None, interactive: bool | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.stream = stream if stream is not None else sys.stderr
        self.interactive = (self.stream.isatty() if interactive is None else interactive)
        self.phases = [(key, label) for key, label in _PHASES if ai_bridges or key != "ai_bridges"]
        self.names = dict(self.phases)
        self.asset_id = _safe(asset_id, 64)
        self.clock = clock
        self.started_at = clock()
        self.current: str | None = None
        self.current_since = self.started_at
        self.completed: set[str] = set()
        self.detail = ""
        self._last_draw = 0.0
        self._last_plain = 0.0
        self._last_plain_detail = ""
        self._closed = False
        self._introduced = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def _heartbeat(self) -> None:
        while not self._stop.wait(1):
            with self._lock:
                if self.current and not self._closed:
                    self._draw()

    def _introduce(self) -> None:
        if self._introduced:
            return
        self._introduced = True
        self._write(f"\nKUMIKI · {self.asset_id} · live reconstruction\n")
        for start in range(0, len(self.phases), 4):
            phases = self.phases[start:start + 4]
            self._write("  " + "  ›  ".join(
                f"{start + offset + 1:02} {_SHORT_NAMES[key]}"
                for offset, (key, _) in enumerate(phases)) + "\n")

    def _elapsed(self, since: float) -> str:
        seconds = max(0, int(self.clock() - since))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"

    def _line(self) -> str:
        done = len(self.completed)
        total = len(self.phases)
        filled = round(16 * done / total)
        bar = "━" * filled + "·" * (16 - filled)
        title = self.names.get(self.current, "Waiting")
        number = next((i for i, (key, _) in enumerate(self.phases, 1)
                       if key == self.current), total)
        phase = f"{number}/{total}" if self.current else f"{done}/{total}"
        return (f"KUMIKI {self.asset_id} [{bar}] {done * 100 // total}% phases "
                f"· {phase} {title} · {self._elapsed(self.current_since)}"
                + (f" · {self.detail}" if self.detail else ""))

    def _write(self, message: str) -> None:
        self.stream.write(message)
        self.stream.flush()

    def _draw(self, *, force: bool = False) -> None:
        if self._closed:
            return
        now = self.clock()
        if self.interactive:
            if not force and now - self._last_draw < 0.3:
                return
            width = max(40, shutil.get_terminal_size((110, 24)).columns)
            self._write("\r\x1b[2K" + self._line()[:width - 1])
            self._last_draw = now
        elif force or now - self._last_plain >= 15:
            self._write(self._line() + "\n")
            self._last_plain = now
            self._last_plain_detail = self.detail

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._emit_locked(event)

    def _emit_locked(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        phase = event.get("phase")
        if self._closed or (phase is not None and phase not in self.names):
            return
        if kind == "start":
            self._introduce()
            if self._heartbeat_thread is None:
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat, name="kumiki-cli-progress", daemon=True)
                self._heartbeat_thread.start()
            if self.interactive and self.current is not None:
                self._write("\r\x1b[2K")
            self.current = phase
            self.current_since = self.clock()
            self.detail = _safe(event.get("detail", ""))
            self._draw(force=True)
        elif kind == "tick":
            detail = _safe(event.get("detail", ""))
            if detail:
                self.detail = detail
            self._draw()
        elif kind in ("done", "skip"):
            if phase in self.completed:
                return
            self.completed.add(phase)
            self.current = phase
            self.detail = "Skipped (using existing data)" if kind == "skip" else "Complete"
            self._draw(force=True)
            if self.interactive:
                self._write("\n")
            self.current = None
        elif kind in ("blocked", "failed"):
            self.current = phase or self.current
            self.detail = _safe(event.get("detail", "Blocked"))
            self._draw(force=True)
            if self.interactive:
                self._write("\n")
            self._closed = True
            self._stop.set()
        elif kind == "finished":
            self.current = None
            self.detail = "Finished"
            self._draw(force=True)
            if self.interactive:
                self._write("\n")
            self._closed = True
            self._stop.set()


def run_monitored_worker(argv: list[str], log: Path, environment: dict[str, str],
                         timeout_seconds: int, *, stage: str, root: Path,
                         emit: Callable[[dict[str, Any]], None] | None = None,
                         clock: Callable[[], float] = time.monotonic) -> int:
    """Preserve complete logs while polling native workers without blocking UI."""
    started = clock()
    process = None
    try:
        with log.open("w", encoding="utf-8") as stream:
            try:
                process = subprocess.Popen(argv, env=environment, stdout=stream,
                                           stderr=subprocess.STDOUT)
            except OSError as error:
                stream.write(f"\nCould not start stage: {error}\n")
                return 127
            last_offset = 0
            last_activity = ""
            while True:
                status = process.poll()
                try:
                    size = log.stat().st_size
                    if size > last_offset:
                        with log.open("rb") as reader:
                            reader.seek(max(last_offset, size - 8192))
                            payload = reader.read(8192)
                        last_offset = size
                        lines = [_safe(part) for part in payload.decode("utf-8", "replace").splitlines()]
                        meaningful = [part for part in lines if part and not part.startswith("{")]
                        if meaningful:
                            last_activity = meaningful[-1]
                except OSError:
                    pass
                if emit is not None:
                    emit({"kind": "tick", "phase": stage,
                          "detail": _stage_detail(root, stage) or last_activity or "Worker running"})
                if status is not None:
                    return status
                if clock() - started >= timeout_seconds:
                    process.kill()
                    process.wait()
                    stream.write("\nStage timed out\n")
                    return 124
                time.sleep(0.4)
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise
